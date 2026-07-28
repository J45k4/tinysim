# Compiler-memory reduction plan

## Goal

Reduce the host memory and first-call compilation cost of contact-heavy TinySim
models without weakening the physics model, introducing model-specific kernels,
or making TinySim manage tinygrad's private global caches.

The primary acceptance workload is the 10-level Jenga model:

- 30 independent free bodies;
- 30 box-plane pairs;
- 27 box-box support pairs;
- 256 parallel worlds;
- CUDA TinyJit execution.

Recording is not part of this problem. The trajectory path already keeps its
incremental host-memory overhead bounded. Raw simulation tensors are also not
the cause: their device-resident allocation is only a few MiB in the full
workload.

## Implementation result

The independent free-body smooth-contact path now implements the high-leverage
parts of this plan:

- model compilation canonicalizes collision pairs into primitive-kind groups;
- one tensor call evaluates every box-plane pair in its group;
- one tensor call evaluates every box-box pair in its group;
- endpoint point velocities and generalized wrenches are batched;
- primitive groups use fixed 1/2/4-slot contact manifolds;
- a padded incident-endpoint table gathers and reduces forces per body;
- output is restored to generalized-coordinate order even when joint and body
  authoring orders differ;
- the compiler-memory benchmark enforces the full gate in a fresh process.

The acceptance command is:

```bash
PYTHONPATH=.:tinygrad DEV=CUDA \
  python3 -m benchmarks.bench_compiler_memory --acceptance
```

On the 10-level, 256-world CUDA support workload it reports:

| Measurement | Result |
| --- | ---: |
| Peak resident host memory | 3,025,481,728 bytes |
| Resident host memory after capture | 975,769,600 bytes |
| CUDA allocator-resident memory after capture | 6,914,768 bytes |
| Live UOps after capture | 156,717 |
| Captured calls | 70 |
| First call | 55.72 seconds |
| Capture | 29.47 seconds |
| Warm replay | 5.78 milliseconds |

The strict acceptance limit is 6,000,000,000 bytes of peak RSS, so the workload
uses about 51% of the allowed memory. The process virtual-address-space peak is
not used as a physical-memory measurement.

For the controlled 10-level, one-world probe, batching reduced peak RSS from
about 9.24 GB to 2.23 GB, captured calls from 312 to 77, and warm replay from
roughly 15 ms to 5 ms. The earlier 256-world recording warmup held about
12.29 GB current RSS; the isolated acceptance process now settles below
0.98 GB after capture.

All 187 native CPU tests pass, with four expected optional-reference skips.
The equivalence suite includes mixed box-plane/box-box contact, reversed pair
orientation, reversed joint order, differentiability, TinyJit replay, and the
general contact-Jacobian reference path.

The compact SAT rewrite, custom UOps, cache-clearing workarounds, and upstream
TinyJit compaction are not required for the 6 GB gate. They remain conditional
follow-up work and require new profiling evidence.

## Investigation result

The dominant cost is the expression graph produced by independently compiling
every box-box pair. The first eager TinyJit call also creates large temporary
scheduler, rewrite, and code-generation structures. Some of those pages remain
in the process allocator after compilation, but cache and allocator cleanup
does not remove the captured executable's graph.

The useful distinction is:

```text
small batched state
        |
        v
Python loops construct one expression graph per pair and body
        |
        v
tinygrad scheduling, rewriting, and code generation       <- transient peak
        |
        v
TinyJit captured calls and their compiled UOp graphs       <- retained memory
```

Increasing the number of worlds changes tensor shapes but does not multiply the
Python graph by the number of worlds. Increasing collision pairs does.

### Measurement method

Each meaningful case ran in a fresh process. The probe sampled:

- current RSS from `/proc/self/status`;
- lifetime peak RSS from `resource.getrusage`;
- elapsed first-call, capture, and replay time;
- live `UOp` count;
- schedule, program, and runtime cache sizes;
- captured TinyJit call count;
- tinygrad device allocator bytes.

TinyJit was called four times:

1. eager execution and compilation;
2. capture and lowering;
3. first replay;
4. steady replay.

A diagnostic-only cleanup then cleared tinygrad's global schedule, program,
runtime, first-run, and local-size caches, ran Python garbage collection, and
called glibc `malloc_trim`. Replay was tested again to determine which objects
were merely cached and which were required by the captured executable.

This cleanup is an investigative instrument, not a proposed TinySim runtime
feature. It is process-global, depends on private tinygrad details, is unsafe
for independently owned simulations, and `malloc_trim` is Linux/glibc-specific.

### Phase baseline

The following are fresh-process CUDA results with one world. MiB values are
RSS divided by 1,048,576.

| 10-level contact set | Pairs | Captured calls | Programs after compile | UOps after cleanup | RSS after cleanup |
| --- | ---: | ---: | ---: | ---: | ---: |
| no contacts | 0 | 6 | 587 | 42,111 | 424 MiB |
| floor only | 30 box-plane | 68 | 900 | 73,683 | 605 MiB |
| support | 30 box-plane + 27 box-box | 312 | 1,910 | 672,040 | 2,105 MiB |

Relative to floor-only contact, the 27 box-box pairs add:

- 598,357 live UOps after cleanup;
- 244 captured calls;
- 1,010 compiled program-cache entries before cleanup;
- about 1.46 GiB of retained RSS after cleanup.

The full support case took about 415 seconds for its first call and 61 seconds
for capture. Its lifetime peak reached about 8.61 GiB and it settled at about
2.90 GiB before diagnostic cleanup. Cleanup released about 865 MiB, but replay
still retained about 2.05 GiB and 672k UOps. Warm replay itself took roughly
15 ms.

No-contact dynamics with 256 worlds retained about 491 MiB and 52k UOps after
cleanup, close to the one-world result. This confirms that batch state is a
secondary contributor.

The earlier full recording process showed approximately 11.45 GiB current RSS
after warmup. That number is a valid end-to-end process observation, but it is
not all live simulation state or captured graph. The fresh probes show that
temporary compiler allocations and allocator retention vary with shapes and
process history. Future reports must give both current and peak RSS and must
not label either as tensor memory.

### Object-retention evidence

At two Jenga levels, the captured process contained roughly:

- 124k live UOps;
- 384k tuples;
- 189k dictionaries;
- 127k weak references;
- 122k sets;
- 41k frozen sets.

The shallow size of the UOp objects themselves was only about 10 MiB. Their
supporting dictionaries, source tuples, sets, recursive-property caches, and
compiler metadata dominate the Python heap.

Common memoized fields on compiled UOps include:

- source-operation sets;
- recursive range and ended-range sets;
- device, shape, key, and address-space properties;
- tuple and boolean-slice rewrites;
- min/max analysis.

Therefore, reducing the number and size of independently lowered UOp graphs is
the first-order fix. Micro-optimizing the Python `UOp` object or the trajectory
format is not.

## Root causes in TinySim

### One collision graph per pair

`smooth_free_body_generalized_force` iterates over every compiled collision
pair and calls `_collide` independently. For a box-box pair, `box_box` builds
15 separating-axis tests with Python loops, repeated reductions, and chained
`where` expressions.

The pair forces are then accumulated into one expression chain per body. This
duplicates structurally similar graph fragments and gives the scheduler a much
larger graph than the numeric problem requires.

### Scalarized independent-body dynamics

The independent free-body path still loops over bodies. Bias force construction
and each 3x3 angular solve are separately expressed. This explains much of the
42k-UOp no-contact floor, but it is not the dominant Jenga regression.

### Compiler intermediates outlive the useful peak

tinygrad's schedule and program caches retain some compilation products.
Lowered UOps also retain memoized analysis fields. Clearing global caches
releases meaningful memory, yet the captured `TinyJit` object still needs its
compiled calls and retains most of the support-case graph.

TinySim cannot safely solve this part by clearing private global dictionaries.
Any final compaction of a captured program belongs in a public tinygrad
abstraction or upstream change.

## Design principles

The implementation must follow these constraints:

- batch equivalent work in tensors; do not generate a Python graph per pair;
- preserve leading batch dimensions so the same path handles one or many
  worlds;
- specialize by collision algorithm, not by example name or Jenga topology;
- use precomputed integer topology as model metadata;
- avoid a dense pair-by-body incidence matrix, whose cost grows as
  `pair_count * body_count`;
- do not insert `realize()` calls per pair as a way to hide compiler pressure;
- do not clear tinygrad global caches from library code;
- do not add a custom UOp until measurements show a remaining operation that
  cannot be expressed compactly with existing tensor operations;
- keep eager, TinyJit, differentiation, and backend behavior equivalent.

## Target architecture

Compile topology once and execute each collision kind once:

```text
CompiledModel
  collision groups
    box-plane: pair indices, geometry indices, parameters
    box-box:   pair indices, geometry indices, parameters
  incident endpoints per body
    pair index, endpoint side, mask
             |
             v
gather transforms and velocities as [world, pair, ...]
             |
             v
one batched contact primitive per collision kind
             |
             v
endpoint forces as [world, pair, endpoint, 6]
             |
             v
gather bounded incident endpoints and reduce per body
             |
             v
generalized force as [world, nv]
```

Collision primitives already accept leading dimensions in many operations.
The desired representation makes `pair` another leading tensor dimension
rather than another Python invocation.

## Implementation phases

### 1. Land a reproducible compiler-memory benchmark

Convert the investigation probe into
`benchmarks/bench_compiler_memory.py`. Its parent process must start one child
process per case so peak RSS and global caches never leak between cases.

Cover:

- levels 1, 2, 4, and 10;
- no-contact, floor-only, and support pair sets;
- one and 256 worlds;
- CPU quick cases and CUDA acceptance cases.

Report:

- model body, geometry, joint, pair, and world counts;
- pair counts by collision kind;
- current and peak RSS at import, model compilation, first call, capture,
  replay, and completion;
- first-call, capture, and synchronized replay time;
- live UOps, captured calls, kernels/programs, and device allocator bytes;
- exact git revision, tinygrad revision, backend, dtype, and environment.

Diagnostic modes may separately measure Python GC, each tinygrad cache, and
allocator trimming. The default acceptance result must not clear caches or trim
the allocator, because users do not do that between simulations.

Store only reports explicitly requested by `--output`; generated reports remain
ignored by git.

Review gate: reproduce the ordering `support >> floor > none` and preserve the
baseline above before changing physics graph construction.

### 2. Compile collision groups

Extend model compilation to group collision pairs by primitive signature. At a
minimum, the key includes the two geometry kinds and any static mode that
changes the algorithm. Each group stores compact integer tensors or immutable
tuples for:

- pair indices;
- geometry A and B indices;
- body A and B indices;
- material/solver parameter indices;
- endpoint signs needed during force accumulation.

Keep authoring order available for diagnostics, but execution order may be
canonicalized by group. Empty groups must not manufacture zero-width kernels.

This metadata is generic to all models. No code may test the model name,
specific body count, tower level, or support-pair pattern.

Review gate: compilation tests prove that heterogeneous pair lists are grouped
deterministically and can be mapped back to authoring order.

### 3. Batch contact primitives by collision kind

Gather positions, quaternions, sizes, velocities, and material properties for
all pairs in a group. Shapes should follow:

```text
positions:   [world, pair, 3]
quaternions: [world, pair, 4]
velocities:  [world, pair, 6]
parameters:  broadcastable over [world, pair]
```

Call `box_plane` once for all box-plane pairs and `box_box` once for all
box-box pairs. Return contact position, normal, signed distance, relative
velocity, and force with the pair dimension intact.

Do not stack results produced by an existing per-pair loop; that preserves the
graph duplication. The primitive itself must receive and operate on the pair
axis.

Add a benchmark switch that compares:

- fully fused group evaluation;
- a realized group transform boundary;
- a realized group contact-force boundary.

Select a boundary only when it reduces compile peak materially and its extra
device traffic and replay kernels stay within the performance gate. Never
realize each pair separately.

Initial review target for the 10-level support model:

- captured calls at or below 100, down from 312;
- compiled programs at or below 1,000, down from 1,910;
- post-cleanup live UOps at or below 150k, down from 672k;
- post-cleanup RSS at or below 900 MiB, down from about 2.05 GiB;
- warm replay no more than 10% slower.

These are directional prototype gates. Tighten them after the first batched
implementation establishes reproducible CPU and CUDA results; do not relax
them without attaching evidence.

### 4. Replace pair-to-body expression chains

Precompute a padded incident-endpoint table:

```text
incident_pair[body, slot]
incident_side[body, slot]
incident_mask[body, slot]
```

`slot` is the maximum number of incident endpoints for this compiled model.
At runtime, gather endpoint wrenches with these indices, mask padding, and
reduce over `slot`. This scales with the actual contact graph degree rather
than constructing a dense `[pair, body]` incidence tensor.

Confirm the sign convention for both endpoints and ensure fixed/world
geometries do not create invalid body gathers. Preserve differentiability with
respect to state and supported parameters.

Review gate: there is no Python loop over collision pairs in the runtime force
path and no sequential tensor addition chain per body.

### 5. Compact the box-box SAT expression

Only begin this phase after pair batching is measured. Pair batching may be
sufficient.

If box-box still dominates, express the oriented-box SAT in matrix form:

- compute the relative rotation and translation once;
- build the three A-face, three B-face, and nine cross-axis candidate
  separations in tensors;
- build matching candidate normals in the same axis dimension;
- select the maximum separating axis with one reduction/gather operation;
- compute the contact approximation from the selected axis without a
  14-element chain of `where` nodes.

Handle nearly parallel cross axes with an explicit validity mask and stable
epsilon policy. Do not change the documented smooth-contact semantics merely
to obtain a smaller graph.

Review gate: primitive values and gradients match the existing implementation
on separated, touching, penetrating, rotated, nearly parallel, and degenerate
test cases.

### 6. Vectorize independent free-body dynamics

After collision graph work, reshape independent root-free bodies into one
body-batched path:

- qpos as `[world, body, 7]`;
- qvel and force as `[world, body, 6]`;
- inertia and mass broadcast over `[world, body]`;
- one batched world/body kinematics and bias calculation;
- one batched 3x3 angular solve over `world * body`.

This optimization must be selected from joint topology, not example identity.
The general articulated path remains the fallback.

Initial review target for the 10-level no-contact case:

- at or below 20k post-cleanup UOps, down from 42k;
- at or below 300 MiB post-cleanup RSS, down from 424 MiB;
- at or below four captured calls, down from six;
- no replay-time regression greater than 10%.

### 7. Investigate public TinyJit compaction upstream

Re-measure after the TinySim graph is compact. If retained compiler metadata is
still material, prototype an upstream tinygrad API rather than adding private
cache manipulation to TinySim.

A safe design could allow a captured `TinyJit` to discard lowering-only data
while retaining:

- executable runners/programs;
- input and output buffer slots;
- symbolic variable bindings;
- graph dependencies required for replay;
- debugging metadata explicitly requested by the caller.

The API must:

- preserve replay and graph execution;
- be scoped to one captured object;
- coexist with other TinyJit instances;
- define whether later recapture or introspection remains possible;
- avoid process-global cache eviction;
- include tinygrad-side memory and correctness tests.

Do not make this phase a prerequisite for landing the generic collision
batching. TinySim currently creates far more graph than it should, and upstream
compaction should not conceal that.

## Validation plan

### Numerical equivalence

Compare the old and new paths before deleting the old implementation:

- contact distance, normal, position, and wrench for every primitive case;
- generalized force before integration;
- acceleration and next state for one eager step;
- TinyJit output after capture and replay;
- 10-, 100-, and 1,000-step Jenga trajectories.

Use explicit tolerances appropriate to dtype. Near contact-mode boundaries,
compare invariants and bounded error rather than requiring an unstable contact
axis to have identical sign.

### Topology and invariance

Test:

- no pairs and one pair;
- mixed box-plane and box-box lists;
- disconnected bodies and bodies with many incident pairs;
- fixed geometry on either endpoint;
- pair-order permutation;
- body-order permutation with state remapping;
- one and multiple worlds;
- float32 and supported float64 backends.

Pair-order permutation must not materially change forces or trajectories.

### Gradient checks

For small deterministic scenes:

- compare autodiff with central finite differences for position, orientation,
  velocity, and selected material parameters;
- test separated and penetrating box-box configurations;
- exclude exact nonsmooth switching points from scalar finite-difference
  equality and test bounded directional behavior there;
- run both eager and captured paths where TinyJit's differentiation contract
  applies.

### Memory and performance

Every memory acceptance case runs in a fresh subprocess. Report results rather
than only a pass/fail boolean.

Final 10-level, 256-world CUDA targets:

- first-call/capture lifetime peak RSS at or below 4 GiB;
- current RSS after capture at or below 1.5 GiB without cache clearing or
  allocator trimming;
- captured calls and program count materially below the original baseline;
- device allocator memory reported separately;
- warm step throughput no more than 10% slower than baseline;
- recording adds no more than its existing bounded-memory gate.

Also record kernel count and global-memory traffic. A low-RSS result obtained by
splitting the graph into hundreds of device round trips does not pass.

### Failure and ownership behavior

- compiling two simulations in the same process must not invalidate either;
- deleting one simulation must not corrupt the other's replay;
- repeated construction/destruction must not grow live graph counts without
  bound;
- exceptions during capture leave subsequent eager execution usable;
- benchmark diagnostics that clear caches remain opt-in and isolated.

## Patch sequence

Keep review units small and independently measurable:

1. compiler-memory benchmark and baseline artifact schema;
2. collision-group model metadata;
3. batched box-plane path;
4. batched box-box path;
5. incident-endpoint body reduction;
6. optional compact SAT rewrite, if justified;
7. vectorized independent-body dynamics;
8. optional upstream TinyJit compaction proposal.

Each performance patch must include the before/after report from the same
revision pair and environment. Do not combine an algorithm rewrite, new
topology representation, and cache experiment into one unreviewable change.

## Critic rejection rules

Return the patch to its implementer if it contains any of the following:

- a runtime Python loop that creates one tensor graph per collision pair;
- Jenga-, tower-, level-, or body-count-specific branches;
- process-global tinygrad cache clearing in TinySim;
- `malloc_trim` or GC calls in the simulation API;
- a dense pair-by-body incidence allocation without measured justification;
- per-pair `realize()` calls;
- hidden changes to timestep, worlds, pair set, dtype, or contact parameters in
  a claimed benchmark improvement;
- memory claims based only on device allocation or only on lifetime peak RSS;
- deleted gradient or physics assertions to make equivalence tests pass;
- a custom UOp without a minimized benchmark proving existing UOps are the
  remaining bottleneck;
- a broad abstraction whose only real caller is the Jenga example.

The desired result is shorter graph construction, not a larger framework around
the same duplicated graph.

## Completion criteria

This plan is complete when:

- collision work is compiled per primitive kind rather than per pair;
- free-body work is batched where topology permits;
- numerical, gradient, topology, and long-rollout tests pass;
- the fresh-process CUDA acceptance case meets the final memory and throughput
  gates;
- memory accounting clearly separates current RSS, peak RSS, captured compiler
  graph, process allocator retention, and device tensors;
- any remaining tinygrad retention issue has a minimized upstream reproducer
  rather than a private TinySim workaround.

## Non-goals

- reducing trajectory memory again;
- changing Jenga into a simpler validation scene;
- replacing TinyJit;
- adding MuJoCo, JAX, Bullet, or Isaac as runtime dependencies;
- introducing a C/C++ contact backend;
- creating a custom UOp before ordinary tensor batching is exhausted;
- treating allocator trimming as memory management.
