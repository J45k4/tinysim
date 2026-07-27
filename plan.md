# TinySim: Research and Implementation Plan

Status: research MVP implemented for the supported subset; remaining gates tracked below
Date: 2026-07-26

## 1. Purpose

TinySim is an experimental robotics simulation engine whose numerical computation is expressed with tinygrad tensors and lowered through tinygrad's UOp intermediate representation.

The project will investigate whether a small, understandable physics engine can provide:

- High-throughput batched rigid-body simulation on accelerators.
- A differentiable simulation path for controls and selected model parameters.
- A portable execution path across tinygrad-supported CPU, NVIDIA, AMD, and Metal backends.
- A compact codebase suitable for physics, compiler, and reinforcement-learning research.
- Compatibility with useful portions of existing robot-description ecosystems without making an external simulator part of the runtime.

TinySim is not initially intended to match all MuJoCo, Bullet, Isaac Sim, or MJX features. The first objective is a correct and measurable vertical slice that establishes whether tinygrad and UOps are a good foundation for robotics simulation.

## 2. Core hypothesis

A robotics simulator can map effectively to tinygrad when:

1. The robot topology and tensor shapes are compiled ahead of execution.
2. Parallel worlds are represented by an explicit leading batch dimension.
3. Tree traversals and solver iterations are scheduled statically.
4. Runtime branching is represented with masks where practical.
5. The Tensor API is used for most implementation work.
6. Direct UOp kernels are introduced only for measured bottlenecks.

The initial research question is not whether TinySim can reproduce every feature of MJX. It is whether a small set of articulated-body dynamics and contact operations can be made correct, differentiable, and efficient through tinygrad's compiler stack.

## 3. Project principles

### 3.1 Tensor-first implementation

Physics algorithms should first be written using `tinygrad.Tensor`. Tensor expressions already construct UOp graphs, so this provides automatic differentiation, scheduling, device portability, and faster iteration.

Handwritten UOps should be reserved for operations that cannot be expressed efficiently through the Tensor frontend, such as:

- Static-topology segmented accumulation.
- Specialized small batched factorization or solve kernels.
- Fused collision narrow-phase kernels.
- Operations that require a custom backward definition.

### 3.2 Static model, dynamic state

Robot structure is treated as compile-time information. Position, velocity, control, contact activity, and randomized parameters are runtime tensors.

Each compiled simulator is specialized by at least:

- Model topology.
- Maximum contact layout.
- Batch size, unless symbolic batching proves reliable and efficient.
- Device.
- Numeric dtype.
- Enabled physics features.
- Solver and integration iteration counts.

### 3.3 Correctness before breadth

Every subsystem must be validated independently before it is integrated into a complete step. Supporting one joint or shape correctly is more valuable than nominally accepting many unsupported combinations.

### 3.4 Explicit limitations

Importers must reject unsupported features with actionable errors. TinySim must not silently reinterpret model semantics.

### 3.5 Benchmark complete workloads

Kernel microbenchmarks are useful, but the main performance metrics are simulation steps per second, compilation latency, kernel launches per step, memory usage, and scaling across numbers of parallel worlds.

## 4. Goals and non-goals

### 4.1 Initial goals

- Batched simulation of pendulum, cart-pole, and small articulated chains.
- Reduced-coordinate rigid-body dynamics.
- Hinge and slide joints, followed by ball and free joints.
- Semi-implicit Euler integration.
- Primitive collision detection.
- Smooth compliant contact with differentiable forces.
- Fixed-iteration constraint solving.
- Gradient calculation with respect to controls and selected physical parameters.
- A direct Python modeling API.
- A useful subset of MJCF import.
- Numerical comparison against analytical systems and CPU MuJoCo.
- CPU and at least one GPU execution path.

### 4.2 Later goals

- More complete MJCF support.
- URDF import.
- Joint limits and equality constraints.
- Coulomb-like friction constraints.
- Mesh and convex collision.
- Sensors.
- Tendons and more actuator types.
- Reinforcement-learning environment integration.
- Multi-device execution.
- Native scene serialization.

### 4.3 Explicit early non-goals

- Photorealistic rendering.
- Full Isaac Sim or USD scene support.
- General deformable-body simulation.
- Fluids.
- Soft bodies.
- Complete MuJoCo compatibility.
- Bit-identical trajectories across engines.
- Dynamic allocation inside compiled simulation steps.
- Arbitrary topology changes without recompilation.
- Production safety or real-time control certification.

## 5. Success criteria

The project is viable when the following evidence exists:

### 5.1 Correctness

- Pendulum and cart-pole agree with analytical equations.
- One-step articulated dynamics agree with MuJoCo within documented float32 and float64 tolerances.
- Collision distances, normals, and contact points pass geometric unit tests.
- Conserved quantities behave as expected in undamped, unforced, contact-free systems.
- Long rollouts remain stable for chosen reference models.

### 5.2 Differentiability

- Gradients of one-step and short-horizon losses agree with central finite differences.
- Gradients remain finite for selected smooth-contact scenarios.
- Batched gradients have the same results as independently evaluated worlds.
- Custom kernels, if introduced, have dedicated backward tests.

### 5.3 Performance

- `TinyJit` removes repeated Python and compilation overhead after warm-up.
- Throughput grows substantially as the number of parallel worlds increases.
- Simulation state remains on the accelerator during rollouts.
- Profiling identifies no accidental host synchronization in the step loop.
- Performance reports include compilation time rather than hiding it.

### 5.4 Portability

- The core correctness suite passes on CPU.
- At least one supported GPU backend passes the core suite.
- Backend-specific kernels have portable fallbacks.

## 6. High-level architecture

```text
Python Model API        MJCF importer        Future URDF importer
       \                     |                       /
        +---------------- TinySpec ----------------+
                              |
                       model compilation
                              |
                       CompiledModel
             static arrays, schedules, capacities
                              |
          +-------------------+--------------------+
          |                                        |
       State tensors                          Runtime parameters
 qpos, qvel, act, ctrl, time          masses, friction, gains, etc.
          |                                        |
          +---------------- simulation step -------+
                              |
       kinematics -> dynamics -> collision -> constraints
                              |
                    solve -> integrate -> sensors
                              |
                       tinygrad Tensor graph
                              |
                          TinyJit capture
                              |
                           UOp kernels
                              |
                      CPU / NV / AMD / Metal
```

MuJoCo may be used as a development dependency for model parsing and numerical comparison. It is not part of TinySim's physics runtime.

## 7. Model representations

TinySim should keep authoring, compiled model data, and dynamic state separate.

### 7.1 `ModelSpec`

`ModelSpec` is the engine-independent authoring structure. It should be easy to construct directly from Python and serve as the target of file importers.

Proposed content:

- Bodies and body-local frames.
- Inertial properties.
- Joints and limits.
- Collision and visual shapes.
- Collision filter information.
- Actuators.
- Sensors.
- Equality constraints.
- Global simulation options.
- Named initial states or keyframes.
- Source metadata for error reporting.

`ModelSpec` values may be symbolic or incomplete when the compiler can infer them. For example, mass and inertia may optionally be inferred from collision geometry.

### 7.2 `CompiledModel`

`CompiledModel` is a flat, validated, device-oriented representation. It contains only data required by the enabled runtime features.

Representative arrays:

```text
body_parent[nbody]
body_depth[nbody]
body_mass[nbody]
body_inertia[nbody, 3, 3]
body_pos[nbody, 3]
body_quat[nbody, 4]

joint_type[njoint]
joint_body[njoint]
joint_axis[njoint, 3]
joint_qpos_address[njoint]
joint_dof_address[njoint]

dof_body[nv]
dof_parent[nv]
dof_damping[nv]

geom_type[ngeom]
geom_body[ngeom]
geom_size[ngeom, max_size]
geom_pos[ngeom, 3]
geom_quat[ngeom, 4]
geom_friction[ngeom, friction_width]

collision_pair[npair, 2]
pair_contact_address[npair]

actuator_dof[nu]
actuator_gain[nu]
actuator_limit[nu, 2]
```

It also contains Python-side static schedules:

- Bodies grouped by tree depth.
- Reverse-depth groups.
- Joint-type groups.
- Static parent and child gather indices.
- Collision pairs grouped by shape combination.
- Constraint row layout.
- Sparse adjacency or CSR metadata.
- Maximum contacts and constraints.

### 7.3 Dynamic `State`

Dynamic data uses structure-of-arrays layouts with worlds as the leading dimension:

```text
qpos[B, nq]
qvel[B, nv]
act[B, na]
ctrl[B, nu]
time[B]

body_xpos[B, nbody, 3]
body_xquat[B, nbody, 4]
body_xmat[B, nbody, 3, 3]

contact_distance[B, ncontact]
contact_position[B, ncontact, 3]
contact_normal[B, ncontact, 3]
contact_active[B, ncontact]
```

Derived fields should not all be permanently stored. Their materialization should be guided by reuse, scheduling behavior, memory cost, and debugging needs.

### 7.4 Runtime-randomized model parameters

Domain randomization should be possible without recompiling topology. Runtime parameters may have either shared or per-world forms:

```text
body_mass[nbody] or body_mass[B, nbody]
geom_friction[ngeom, 3] or geom_friction[B, ngeom, 3]
actuator_gain[nu] or actuator_gain[B, nu]
```

The compiler must distinguish structural values from runtime numeric values.

## 8. Coordinate and numerical conventions

These conventions must be fixed before implementing dynamics:

- Right-handed coordinate system.
- SI units.
- Quaternion layout, likely `(w, x, y, z)`.
- Spatial vector layout, either angular-linear or linear-angular.
- Body-to-world versus world-to-body rotation convention.
- Row-vector versus column-vector transform semantics.
- `qpos` and `qvel` layouts for each joint type.
- Contact normal direction.
- Positive constraint-force direction.
- Float32 as the performance target.
- Float64 on supported CPU paths as the high-accuracy reference.

All spatial algebra functions must document input and output frames. Many rigid-body bugs are frame-convention bugs rather than algebra bugs.

## 9. Physics pipeline

### 9.1 Spatial algebra

Implement and test:

- Quaternion multiplication, inversion, normalization, and integration.
- Quaternion-to-matrix conversion.
- Rotation composition.
- Cross products.
- Spatial motion and force cross operators.
- Spatial transforms.
- Inertia transformation.
- Point velocity and Jacobian helpers.

These functions should be small pure Tensor expressions and receive exhaustive unit tests.

### 9.2 Forward kinematics

Forward kinematics traverses the body tree from roots to leaves.

Execution strategy:

1. Compile bodies into depth buckets.
2. Gather each body's parent transform.
3. Compute joint transforms in parallel within a depth.
4. Compose parent and local transforms.
5. Continue to the next depth.

The number of depth buckets is static, so Python can construct the graph without a runtime tree loop.

### 9.3 Dynamics

Begin with dense reduced-coordinate dynamics:

```text
M(q) qacc + C(q, qvel) = tau + external_forces + constraint_forces
```

Initial algorithms:

- Composite rigid-body algorithm for `M(q)`.
- Recursive Newton-Euler algorithm for bias forces.
- Dense batched factorization or iterative solve.
- Actuator force mapping.

Dense matrices are appropriate for initial correctness and common small robots. Sparse or articulated-body algorithms should be added only after profiling representative models.

### 9.4 Actuation

Initial actuator support:

- Direct torque motor.
- Position servo.
- Velocity servo.
- Control clamping.
- Force clamping.

Later:

- Stateful filters.
- Gear and transmission variants.
- Tendon actuation.
- Muscle models.

### 9.5 Integration

Initial integrator:

- Semi-implicit Euler.

Required details:

- Update velocity from acceleration.
- Integrate scalar joints.
- Integrate quaternion joints using angular velocity.
- Normalize quaternions safely.
- Advance actuator state and time.

Later:

- Explicit RK4 for smooth contact-free validation.
- Implicit damping.
- Fully or semi-implicit methods if stability requires them.

### 9.6 Collision detection

Initial primitive pairs:

- Sphere-plane.
- Sphere-sphere.
- Capsule-plane.
- Capsule-sphere.
- Capsule-capsule.
- Box-plane.

Collision pair generation is performed at model compilation. Runtime collision detection evaluates a fixed set of candidate pairs and returns fixed-size contact slots with activity masks.

Later broadphase options:

- Explicit user-provided collision pairs.
- Sweep-and-prune with fixed capacities.
- Bounding-volume hierarchy with refit.
- Spatial hashing with bounded buckets.

Dynamic allocation and variable-length contact arrays are excluded from compiled steps.

### 9.7 Smooth contact

The first contact model should prioritize stability and useful gradients:

```text
penetration = smooth_positive(-distance)
normal_force = stiffness * penetration - damping * normal_velocity
friction_force = -mu * normal_force * smooth_direction(tangent_velocity)
```

Candidate smooth functions:

- Softplus or a piecewise smooth approximation for penetration.
- `tanh(v / epsilon)` for tangential direction.
- Safe normalization with a configurable epsilon.

Parameters and smoothing scales must be exposed and tested because they affect stability, realism, and gradient quality.

### 9.8 Constraint-based contact

After smooth contact is working, introduce a constraint formulation:

```text
A = J M^-1 J^T + R
A lambda = b
qacc = qacc_free + M^-1 J^T lambda
```

Start with:

- Frictionless unilateral normal constraints.
- Fixed solver iterations.
- Warm starting.
- Masked inactive rows.

Then add:

- Joint limits.
- Pyramidal friction.
- Equality constraints.
- Elliptic friction if justified.
- Projected Gauss-Seidel, conjugate-gradient, or Newton-like methods.

Exact non-smooth forward simulation and smooth differentiable simulation may ultimately be separate modes.

## 10. Tinygrad and UOp execution strategy

### 10.1 Explicit batching

TinySim should not require a `vmap` equivalent. Every public numerical function accepts an explicit world dimension:

```python
def step(model, qpos, qvel, ctrl):
    # qpos: [B, nq]
    ...
```

Static model tensors broadcast across the leading world dimension. Randomized model fields may already include it.

### 10.2 JIT capture

`TinyJit` should capture stable step functions after inputs are realized. The first call establishes buffers, the capture call records execution, and later calls replay the compiled graph.

The initial benchmark must report:

- First-call time.
- Capture/compile time.
- Warm execution time.
- Number of compiled calls or kernels.

### 10.3 Indexing

Generic gather and scatter implementations may construct large one-hot intermediates. Preferred strategies:

- Contiguous slices.
- Linear advanced indexing when supported efficiently.
- Precomputed dense incidence matrices for small models.
- Depth grouping to avoid conflicting writes.
- Static CSR traversal kernels for larger models.

### 10.4 Reductions and child accumulation

Three implementations should be compared:

1. Dense incidence matrix multiplication.
2. Masked dense reduction.
3. Custom static CSR UOp kernel.

The CSR kernel should assign one output lane to each destination and loop over its incident sources. This avoids portable atomic-add requirements.

### 10.5 Batched linear solves

Implementation stages:

1. Pure Tensor, statically unrolled Cholesky or LDL transpose.
2. Fixed-iteration conjugate gradient for positive-definite systems.
3. Specialized custom UOp kernel if profiling justifies it.

The Tensor version provides automatic differentiation and a reference for custom kernels.

For a custom solve, implement an analytic backward using transposed solves rather than differentiating through every factorization instruction.

### 10.6 Runtime control flow

Use:

- Python graph-construction loops for static iteration counts.
- Masks and `where` for per-world conditions.
- Fixed capacities for contacts and constraints.
- Fixed solver iteration counts during initial JIT work.

Avoid:

- Data-dependent Python branches.
- Host reads inside `step`.
- Shape changes based on active contacts.
- Per-step tensor creation on the host.

### 10.7 Graph size management

Long differentiable rollouts can create large graphs. Investigate:

- JIT replay for inference rollouts.
- Chunked horizon differentiation.
- Checkpointing or recomputation.
- A differentiable custom step function.
- Custom backward through a rollout.
- Truncated gradients.
- Implicit differentiation for converged solvers.

## 11. Differentiability policy

TinySim should distinguish three levels:

### Level 1: naturally differentiable

- Kinematics.
- Smooth forces.
- Dense dynamics.
- Integration.
- Smooth contact.

Use tinygrad automatic differentiation directly.

### Level 2: differentiable with fixed algorithmic choices

- Unrolled iterative solvers.
- Masked collision candidates.
- Fixed contact capacities.

Gradients follow the executed fixed iteration graph.

### Level 3: custom or approximate gradients

- Hard contact activation.
- Projection.
- Exact Coulomb friction.
- Sorting and broadphase.
- Specialized UOp solvers.

Each Level 3 operation must document whether its backward is:

- Exact for the implemented forward.
- An implicit derivative.
- A straight-through estimator.
- A deliberately smooth surrogate.
- Unsupported.

No operation should claim differentiability merely because a gradient tensor can be produced.

## 12. Model input strategy

### 12.1 Native Python API

The Python API is the first authoring interface:

```python
model = ModelSpec(
    bodies=[...],
    joints=[...],
    geoms=[...],
    actuators=[...],
    options=Options(timestep=0.002),
)
compiled = compile_model(model, worlds=4096, device="NV")
```

This avoids blocking physics research on a file parser.

### 12.2 MJCF importer

MJCF is the first file importer because it represents dynamics, contacts, actuators, and articulated robots compactly.

Initial supported subset:

- `option`
- `default`, with a deliberately limited inheritance implementation
- `asset/mesh`, initially metadata-only or deferred
- `worldbody/body`
- `inertial`
- `joint`
- Primitive `geom`
- Basic `contact` exclusions and explicit pairs
- `actuator/motor`
- `actuator/position`
- Keyframe positions

Initial supported joints:

- Fixed body relationship.
- Hinge.
- Slide.
- Ball.
- Free.

Initial supported geometry:

- Plane.
- Sphere.
- Capsule.
- Box.

Unsupported elements must report their XML path and source location when possible.

The first importer may optionally use MuJoCo to parse and normalize MJCF. A native importer can replace it after the TinySim semantics stabilize.

### 12.3 URDF importer

URDF follows MJCF to support existing ROS robot descriptions. Simulation-specific data may require:

- Gazebo extensions.
- A TinySim sidecar configuration.
- Explicit default contact and actuator parameters.

### 12.4 Native serialization

Do not choose a permanent native file syntax early. First stabilize `ModelSpec`.

Possible later representation:

- Versioned JSON or YAML for structure and metadata.
- NPZ or another typed tensor container for compiled arrays and large assets.
- A cache key derived from source, compiler version, backend, dtype, and enabled features.

Raw compiled data must not be the only maintained source format because compiled schemas will evolve.

## 13. Public API sketch

```python
from tinysim import ModelSpec, Simulator, State

spec = ModelSpec.from_mjcf("robot.xml")
sim = Simulator.compile(
    spec,
    worlds=4096,
    device="NV",
    dtype="float32",
    contact="smooth",
    solver_iterations=4,
)

state = sim.make_state()
control = sim.zeros_control()

state = sim.step(state, control)
observation = sim.observe(state)
```

Functional core:

```python
qpos_next, qvel_next, act_next = step(
    compiled_model,
    qpos,
    qvel,
    act,
    ctrl,
)
```

Gradient example:

```python
control = control.requires_grad_()
state_next = sim.step(state, control)
loss = state_next.qpos.square().sum()
loss.backward()
```

The class API should remain a thin wrapper over the functional core.

### 13.1 Executable pendulum vertical slice

The first implementation should be a point-mass pendulum with an actuated hinge. It deliberately bypasses `ModelSpec`, articulated-body algorithms, collision, and file import so the tinygrad execution hypothesis can be tested in isolation.

Conventions for this example:

- `theta = 0` is the stable downward configuration.
- Positive `theta` rotates away from the downward vertical.
- `omega` is angular velocity.
- `torque` is the applied hinge torque.
- The pendulum is a point mass at distance `length` from the hinge.
- Its moment of inertia is `mass * length**2`.
- All runtime arrays have shape `[worlds, 1]`.

The continuous dynamics are:

```text
I * angular_acceleration =
    torque
    - damping * angular_velocity
    - mass * gravity * length * sin(angle)
```

The integrator is semi-implicit Euler:

```text
omega_next = omega + dt * angular_acceleration
theta_next = theta + dt * omega_next
```

The following is intended to become `examples/pendulum.py` during Phase 1:

```python
from tinygrad import Tensor, TinyJit


def pendulum_step(
    theta: Tensor,
    omega: Tensor,
    torque: Tensor,
    *,
    mass: float = 1.0,
    length: float = 1.0,
    damping: float = 0.05,
    gravity: float = 9.81,
    dt: float = 0.01,
) -> tuple[Tensor, Tensor]:
    """Advances a batch of independent point-mass pendulums by one step.

    theta, omega, and torque all have shape [worlds, 1].
    """
    inertia = mass * length * length
    gravity_torque = mass * gravity * length * theta.sin()
    acceleration = (
        torque - damping * omega - gravity_torque
    ) / inertia

    # Semi-implicit Euler updates velocity before position.
    omega_next = omega + dt * acceleration
    theta_next = theta + dt * omega_next
    return theta_next, omega_next


@TinyJit
def jitted_pendulum_step(
    theta: Tensor,
    omega: Tensor,
    torque: Tensor,
) -> tuple[Tensor, Tensor]:
    # Realization makes the outputs concrete JIT-captured buffers.
    return tuple(
        output.realize()
        for output in pendulum_step(theta, omega, torque)
    )


def check_control_gradient() -> None:
    """Compares autodiff with a finite-difference directional derivative."""
    worlds = 32
    theta = Tensor.full((worlds, 1), 0.2)
    omega = Tensor.zeros(worlds, 1)
    torque = Tensor.zeros(worlds, 1)

    theta_next, _ = pendulum_step(theta, omega, torque)
    loss = theta_next.mean()
    loss.backward()

    assert torque.grad is not None

    # This is d(loss)/d(epsilon) when epsilon is added to every torque.
    autodiff = torque.grad.sum().item()

    # A relatively large epsilon is intentional: the one-step torque effect
    # is proportional to dt**2 and would otherwise be lost in float32 noise.
    epsilon = 1e-2
    detached_torque = torque.detach()
    loss_plus = pendulum_step(
        theta, omega, detached_torque + epsilon
    )[0].mean().item()
    loss_minus = pendulum_step(
        theta, omega, detached_torque - epsilon
    )[0].mean().item()
    finite_difference = (loss_plus - loss_minus) / (2 * epsilon)

    relative_error = abs(autodiff - finite_difference) / max(
        abs(autodiff),
        abs(finite_difference),
        1e-12,
    )
    print(
        f"control gradient: autodiff={autodiff:.8e}, "
        f"finite_difference={finite_difference:.8e}, "
        f"relative_error={relative_error:.3e}"
    )
    assert relative_error < 1e-2


def main() -> None:
    worlds = 4096
    steps = 1000

    theta = Tensor.full((worlds, 1), 0.2).realize()
    omega = Tensor.zeros(worlds, 1).realize()
    torque = Tensor.zeros(worlds, 1).realize()

    # TinyJit executes normally, captures, and then replays on successive
    # calls. No state is read back to the host inside the rollout.
    for _ in range(steps):
        theta, omega = jitted_pendulum_step(theta, omega, torque)

    # item() synchronizes only after the rollout is complete.
    print(
        f"world 0 after {steps} steps: "
        f"theta={theta[0, 0].item():.6f}, "
        f"omega={omega[0, 0].item():.6f}"
    )

    check_control_gradient()


if __name__ == "__main__":
    main()
```

When the example is saved under `examples/pendulum.py`, it can be run against the pinned tinygrad submodule with:

```bash
PYTHONPATH=tinygrad DEV=CPU python3 examples/pendulum.py
```

An accelerator can be selected using an appropriate tinygrad device, for example:

```bash
PYTHONPATH=tinygrad DEV=NV python3 examples/pendulum.py
```

The rollout intentionally does not call `.numpy()`, `.item()`, or another host-reading operation inside the step loop. Reading state within the loop would synchronize the device and invalidate the intended accelerator benchmark.

This example establishes five requirements before articulated dynamics work begins:

1. A physics step can be expressed entirely with Tensor operations.
2. The world batch dimension is explicit.
3. The state remains on-device across steps.
4. `TinyJit` can replay the simulation step.
5. A control gradient agrees with finite differences.

## 14. Proposed repository layout

```text
tinysim/
  pyproject.toml
  README.md
  plan.md
  tinysim/
    __init__.py
    model.py
    compile.py
    state.py
    spatial.py
    math.py
    kinematics.py
    dynamics.py
    actuator.py
    collision/
      __init__.py
      types.py
      primitive.py
      broadphase.py
    contact.py
    constraint.py
    solver.py
    integrator.py
    simulation.py
    importers/
      __init__.py
      mjcf.py
      urdf.py
    uops/
      __init__.py
      segment.py
      solve.py
  tests/
    unit/
    dynamics/
    collision/
    gradients/
    reference/
    integration/
  benchmarks/
    bench_step.py
    bench_batch_scaling.py
    bench_compile.py
    models/
  examples/
    pendulum.py
    cartpole.py
    bouncing_ball.py
```

Modules should remain small and algorithms should be exposed independently for testing.

## 15. Validation strategy

### 15.1 Analytical references

Use systems with known equations:

- Free particle.
- Ballistic rigid body.
- Simple pendulum.
- Double pendulum where practical.
- Cart-pole.
- Point mass with spring.
- Elastic sphere-plane impact approximations.

### 15.2 MuJoCo comparison

MuJoCo is a development oracle, not a runtime dependency.

Compare intermediate values:

- Body transforms.
- Center-of-mass positions.
- Mass matrix.
- Bias forces.
- Actuator forces.
- Contact distance and normal.
- Constraint Jacobian.
- Acceleration.
- Integrated state.

One-step comparison is more diagnostic than comparing only long trajectories.

### 15.3 Physical invariants

Test:

- Linear momentum without external force.
- Angular momentum without external torque.
- Energy behavior for selected integrators.
- Quaternion norm.
- Equal and opposite contact forces.
- Static equilibrium.
- Symmetry and positive definiteness of the mass matrix.

### 15.4 Gradient validation

For scalar losses, compare automatic gradients with central finite differences for:

- Initial position.
- Initial velocity.
- Control.
- Mass.
- Inertia parameters where safely parameterized.
- Actuator gain.
- Contact stiffness.
- Friction smoothing parameters.

Run gradient tests away from discontinuities first, then add documented near-contact cases.

### 15.5 Tolerances

Tolerances must be defined per subsystem and dtype. Initial targets, subject to calibration:

- Float64 algebraic unit tests: relative error around `1e-8` to `1e-6`.
- Float32 one-step dynamics: relative error around `1e-5` to `1e-3`.
- Float64 finite-difference gradients: relative error around `1e-5`.
- Float32 gradients: looser, operation-specific tolerances.

Trajectory comparisons require separate drift envelopes and must not reuse one-step tolerances.

### 15.6 Final visual and video verification

Every release candidate should produce a deterministic recorded simulation as a final
human-facing verification artifact. This is a smoke test and demonstration layer, not
a replacement for analytical, invariant, gradient, and MuJoCo comparisons: a plausible
video can still hide incorrect physics.

The final verification runner should:

1. Load a versioned model and configuration.
2. Run from a recorded random seed with fixed initial state, controls, timestep,
   integrator, dtype, backend, and solver settings.
3. Save the numerical trajectory before rendering so physics can be inspected without
   rerunning or depending on the renderer.
4. Render frames from the saved trajectory through an optional visualization package;
   rendering and video encoding must not be dependencies of the physics core.
5. Encode an MP4 with the commit, configuration, backend, seed, simulated duration,
   wall-clock duration, and realtime factor stored beside it in a machine-readable
   manifest.
6. Run the same trajectory through automated acceptance checks and fail the release if
   they fail, regardless of how convincing the video looks.

Required final recordings:

- Pendulum energy and phase behavior.
- Cart-pole under a recorded control sequence.
- Free rigid body or articulated robot without contact.
- Bouncing body showing contact, restitution, and settling.
- Representative imported MJCF robot.
- Batched rollout visualized from selected world indices.

Automated checks associated with each recording should cover finite state values,
quaternion normalization, joint-limit violations, penetration bounds, energy or
momentum envelopes where applicable, deterministic trajectory hashes within the
declared tolerance policy, and agreement between eager and `TinyJit` execution.
Maintain one short golden clip for presentation, but compare numerical trajectories or
selected rendered frames with explicit tolerances rather than requiring byte-identical
MP4 files. Codec and driver differences make encoded-video hashes unsuitable as a
correctness oracle.

Suggested command and API:

```bash
python -m tinysim.verify --scenario pendulum --record artifacts/pendulum.mp4
```

```python
report = verify(
    scenario="pendulum",
    seed=0,
    steps=600,
    record="artifacts/pendulum.mp4",
)
assert report.passed
```

## 16. Performance plan

### 16.1 Workloads

Benchmark:

- Pendulum.
- Cart-pole.
- Small quadruped-like tree.
- Humanoid-scale articulated model.
- Bouncing spheres.
- Contact-heavy box stack once supported.

### 16.2 Batch sizes

At minimum:

```text
1, 16, 64, 256, 1024, 4096, 16384
```

Stop increasing when memory or device limits make the result uninformative.

### 16.3 Metrics

- Compilation and capture time.
- Warm step latency.
- Aggregate simulation steps per second.
- Per-world latency.
- Kernel launches per step.
- Peak and steady-state memory.
- Device utilization.
- Forward and backward throughput.
- Time per subsystem.
- Scaling efficiency.

### 16.4 Baselines

Use comparisons carefully because engines may implement different physics:

- Pure Python or NumPy analytical implementation.
- CPU MuJoCo for compatible models.
- MJX-JAX for compatible models and hardware.
- TinySim Tensor implementation versus specialized UOp implementation.

Report model settings, batch size, solver iterations, contact capacity, device, precision, warm-up, and synchronization methodology.

## 17. Testing matrix

Every major feature should be tested across relevant dimensions:

| Dimension | Values |
| --- | --- |
| Device | CPU, one GPU backend initially |
| Dtype | float32, float64 where supported |
| Batch | 1, small batch, large batch |
| Joint | hinge, slide, ball, free |
| Contact | none, inactive candidates, active contact |
| Gradient | forward only, one-step backward, short rollout |
| Solver | direct, fixed-iteration iterative |
| Import | direct Python, MJCF |

Tests should be categorized so CPU-only development remains fast while accelerator and reference suites can run separately.

## 18. Phased roadmap

Progress is controlled by exit criteria rather than dates.

### Phase 0: repository and experiment harness

Deliverables:

- Python package skeleton.
- Test configuration.
- Benchmark harness.
- Device and dtype reporting.
- Reproducible environment instructions.

Exit criteria:

- A Tensor function runs under `TinyJit` on CPU.
- The same test can be selected for an available GPU.
- Benchmark output distinguishes compile and warm execution time.

### Phase 1: batched analytical systems

Deliverables:

- Pendulum.
- Cart-pole.
- Semi-implicit Euler.
- Explicit world batching.
- Control gradients.

Exit criteria:

- Analytical one-step tests pass.
- Finite-difference gradients pass.
- JIT replay works for multiple batch sizes.
- Batch scaling report is recorded.

This phase is the first go/no-go gate for tinygrad suitability.

### Phase 2: spatial algebra and articulated kinematics

Deliverables:

- Spatial math module.
- Model compiler for body trees.
- Hinge and slide joints.
- Depth-grouped forward kinematics.
- Direct Python model API.

Exit criteria:

- Transform conventions are fully documented.
- Kinematics agree with reference calculations.
- Batched and single-world results match.

### Phase 3: contact-free articulated dynamics

Deliverables:

- Composite rigid-body mass matrix.
- Recursive Newton-Euler bias forces.
- Actuator force mapping.
- Batched dense solve.
- Two-link and small-tree examples.

Exit criteria:

- Mass matrices are symmetric and positive definite for valid models.
- Accelerations agree with MuJoCo or analytical references.
- Energy and momentum tests behave as expected.
- Gradients through dynamics pass finite differences.

### Phase 4: primitive collision and smooth contact

Deliverables:

- Fixed collision-pair compiler.
- Sphere and plane collision.
- Capsule collision.
- Activity masks.
- Smooth normal and friction forces.
- Bouncing ball example.

Exit criteria:

- Geometry tests cover separated, touching, penetrating, and degenerate cases.
- Contact rollouts remain stable over defined parameter ranges.
- Contact gradients pass away from activation boundaries.
- Throughput scaling is measured with active contacts.

### Phase 5: model import

Deliverables:

- Initial MJCF subset importer.
- Clear unsupported-feature diagnostics.
- Imported pendulum, cart-pole, and articulated examples.
- Optional MuJoCo-assisted normalization path, used only if the native strict
  importer cannot normalize an otherwise supported model.

Exit criteria:

- Directly authored and imported equivalent models compile to equivalent arrays.
- Importer tests cover defaults, frames, inertials, and actuators.
- Runtime simulation does not require MuJoCo.

### Phase 6: constraint solver

Deliverables:

- Contact Jacobians.
- Frictionless unilateral constraints.
- Fixed-iteration solver.
- Warm starting.
- Joint limits.

Exit criteria:

- Static support and impact tests pass.
- Solver convergence is measured rather than assumed.
- Forward and gradient semantics are documented.
- Solver iteration count is part of compilation configuration.

### Phase 7: targeted UOp optimization

Deliverables depend on profiling:

- CSR segmented reduction kernel.
- Specialized solve kernel.
- Fused collision kernel.
- Custom backward functions.

Exit criteria:

- Every custom kernel has a portable Tensor reference.
- Each optimization demonstrates a material end-to-end improvement.
- Correctness and gradient tests pass on supported backends.
- Backend limitations are documented.

### Phase 8: robotics and reinforcement-learning integration

Deliverables:

- Gym-like environment interface.
- Batched reset and observation functions.
- Domain randomization.
- Short on-device rollout example.
- Representative locomotion model.

Exit criteria:

- Simulation and policy computation can remain on the same device.
- Reset does not require per-world host synchronization.
- Training-oriented throughput and memory are measured.

### Phase 9: final demonstrator and release verification

Deliverables:

- Optional deterministic trajectory renderer with basic camera controls.
- Headless RGB frame generation.
- MP4 export through FFmpeg or a similarly isolated encoder adapter.
- `tinysim.verify` runner and machine-readable verification manifest.
- Recorded demonstrations listed in section 15.6.
- A release report linking numerical, gradient, performance, and visual evidence.

Exit criteria:

- A clean checkout can reproduce the pendulum verification artifact with one command.
- Rendering can consume a saved trajectory and does not alter simulation state.
- Physics and tests run without visualization or video dependencies installed.
- All automated scenario checks pass in eager and `TinyJit` modes.
- At least one supported accelerator produces a valid batched rollout recording.
- The Critic accepts the verification API and confirms it does not leak renderer
  concepts into the physics core.
- The manager reviews the videos for obvious frame, geometry, contact, or stability
  failures and signs the release report only after the numerical gates pass.

## 19. Multi-agent implementation strategy

TinySim should be implemented by one master manager agent coordinating a small set of specialized subagents. The manager owns the project-wide result; subagents own bounded tasks with explicit context, file scopes, interfaces, and acceptance tests.

The purpose of this structure is to gain useful parallelism without allowing architecture, physics conventions, or the shared worktree to diverge.

### 19.1 Master manager agent

The master manager is the only agent responsible for declaring a phase complete. Its responsibilities are:

- Maintain the roadmap, dependency graph, and current critical path.
- Select the next independently executable tasks.
- Prepare complete task packets for subagents.
- Own cross-cutting architecture and public interfaces.
- Assign disjoint file ownership for parallel mutations.
- Resolve interface questions and conflicting findings.
- Track decisions, assumptions, risks, and deferred work.
- Integrate subagent output into the shared design.
- Run phase-level validation after local tests pass.
- Reject incomplete work even when an individual agent reports success.
- Keep `plan.md`, decision records, and support matrices current.

The manager should continue useful local work while subagents execute independent tasks. It should not delegate the final integration judgment.

### 19.2 Recommended subagent roles

Roles represent context specialties, not permanent processes. The manager may instantiate only those needed for the current phase.

| Role | Primary ownership |
| --- | --- |
| Physics agent | Spatial algebra, kinematics, dynamics, integration, physical conventions |
| Tinygrad/UOp agent | Tensor lowering, JIT behavior, indexing, custom kernels, backend portability |
| Collision agent | Primitive geometry, contact generation, contact and constraint models |
| Model agent | `ModelSpec`, compilation, static schedules, MJCF and URDF import |
| Validation agent | Analytical references, MuJoCo comparisons, invariants, gradient checks |
| Performance agent | Benchmarks, synchronization audits, profiling, batch scaling |
| Review agent | Read-only adversarial review of risky mathematics, gradients, or shared APIs |
| Critic agent | Ruthless simplicity, abstraction integrity, code cleanliness, and AI-slop rejection |

With four concurrent agent slots, the default topology should be one manager plus at most three workers. Fewer workers are preferable when integration or shared-interface work is the bottleneck.

### 19.3 Stable shared context

Every agent must receive a compact project context containing:

- Current phase and its exit criteria.
- Pinned tinygrad revision.
- Coordinate, quaternion, spatial-vector, and contact conventions.
- Current public and internal interface definitions.
- Supported devices and dtypes.
- Model and state shape conventions.
- Known limitations and active architecture decisions.
- Relevant local source paths and required reference material.
- Commands for the applicable test suites.

Stable context belongs in repository documentation, not only in chat history. When an agent discovers a project-wide fact, the manager decides whether it becomes a documented invariant, a decision record, or a local implementation detail.

### 19.4 Required task packet

The manager must give each subagent a bounded task packet:

```text
Task ID and roadmap gate:
Task:
Why it matters:
Inputs and authoritative references:
Expected outputs:
Files owned:
Files that must not be modified:
Interface contract:
Non-goals:
Dependencies and assumptions:
Required tests:
Acceptance criteria:
Required completion report:
```

Tasks such as "implement physics" or "make it fast" are too broad. Suitable tasks include:

- Implement and test quaternion integration in `spatial.py`.
- Create an analytical pendulum reference test without editing runtime code.
- Benchmark eager versus `TinyJit` pendulum stepping at specified batch sizes.
- Implement sphere-plane distance and normal for fixed tensor shapes.

### 19.5 Shared-worktree ownership

All agents operate in one shared worktree, so coordination is based on file ownership rather than independent Git merges.

Rules:

1. Parallel mutation tasks must have disjoint owned files.
2. Shared interface files have one active owner at a time.
3. An agent may read any relevant file but may modify only its assigned scope.
4. An agent must notify the manager before changing an interface consumed by another task.
5. The manager serializes migrations that affect multiple subsystems.
6. Agents must preserve unrelated user or agent changes.
7. Destructive Git operations are prohibited.
8. Generated or formatted changes must not spill into unowned files.

Shared boundary files such as package exports, common model/state definitions, global configuration, the decision log, and `plan.md` remain manager-owned unless explicitly handed off. A handoff is complete only when the previous owner reports its final diff and stops editing the transferred scope.

When two tasks require the same file, they should run sequentially or one should produce a design/review report without editing.

### 19.6 Task lifecycle

Each task moves through:

```text
queued
  -> context-ready
  -> in-progress
  -> locally-validated
  -> independent-review
  -> critic-gate
  -> manager-integrated
  -> phase-validated
```

The Critic gate branches as follows:

```text
ACCEPT                 -> manager-integrated
RETURN TO IMPLEMENTER  -> in-progress with the same agent
BAN FROM INTEGRATION   -> terminate agent
                         -> revoke ownership
                         -> spawn replacement
                         -> context-ready with a fresh agent
```

A task is not complete at `locally-validated`. The manager must confirm the diff is in scope, inspect the evidence, and run integration tests. High-risk work requires an independent validation or review handoff.

An agent should report a blocker immediately when it affects an interface or another active task. It should not silently expand scope to work around the blocker.

Agents should send compact transition reports using `working`, `blocked`, or `ready-for-review`, including completed work, files changed, tests and results, contract deviations, risks, and the next action.

### 19.7 Completion report contract

Every implementation agent returns:

```text
Outcome:
Files changed:
Interfaces added or changed:
Algorithms and conventions used:
Tests run and exact results:
Benchmarks run and exact results:
Known limitations:
Risks or uncertain assumptions:
Recommended next task:
```

Claims such as "tests pass" must identify the commands and results. Performance claims must include device, dtype, batch size, warm-up, synchronization, and compile-time treatment.

### 19.8 Independent validation handoff

Risky functionality follows a three-party evidence chain:

1. The implementation agent supplies code and focused unit tests.
2. A validation or review agent checks it using an independent method.
3. The manager integrates it and runs the broader phase gate.

Examples:

- A dynamics implementation is checked against analytical equations or MuJoCo intermediates.
- A custom UOp kernel is checked against its Tensor reference on multiple shapes and devices.
- A custom backward is checked with central finite differences.
- A collision routine is checked with independently constructed geometric cases.

The validation agent should initially be read-only with respect to the implementation under review. This reduces the chance that the reference repeats the same assumption or bug.

### 19.9 Critic agent: ruthless code-quality gate

The Critic agent is a separate read-only gate inspired by the uncompromising, direct engineering standards commonly associated with Linus Torvalds and George Hotz. It does not imitate either person. Its job is to protect TinySim from complexity, leaky abstractions, cargo-cult patterns, and low-quality AI-generated code.

The Critic is intentionally blunt. It may complain loudly about the code, but criticism must remain technical. A ban is an execution-policy decision about the current implementer agent instance and its work product, not a personal judgment.

The Critic asks:

- Is this the shortest clear correct solution?
- Can an entire type, layer, wrapper, option, or helper be deleted?
- Does every abstraction hide a real stable boundary?
- Does the abstraction leak implementation details to its callers?
- Is there one obvious path through the code?
- Are names precise enough that comments are mostly unnecessary?
- Are shapes, frames, ownership, and mutation visible rather than magical?
- Does the implementation solve today's accepted requirement rather than imaginary future requirements?
- Is the Tensor reference simpler than the proposed custom UOp?
- Can a new contributor understand the implementation without reconstructing an agent's hidden reasoning?

The target is not code golf. Dense clever code is rejected alongside bloated code. The desired implementation has the fewest concepts and lines consistent with correctness, explicit semantics, numerical clarity, testability, and backend portability.

#### AI-slop indicators

The Critic treats the following as presumptive reasons to return work:

- Layers whose only purpose is forwarding arguments unchanged.
- `Manager`, `Factory`, `Adapter`, `Provider`, or `Registry` types without demonstrated need.
- Generic frameworks created before two concrete uses exist.
- Duplicate representations of the same state or configuration.
- Functions split so aggressively that the algorithm is impossible to read locally.
- Giant functions that mix model compilation, physics, device execution, and reporting.
- Comments and docstrings that merely restate the code.
- Verbose boilerplate hiding a small equation.
- Broad exception handling that conceals programmer errors.
- Silent fallbacks or compatibility paths without tests.
- Configuration options that are never exercised.
- Placeholder APIs, dead branches, speculative hooks, and abandoned TODO scaffolding.
- Copy-pasted implementations that should share one mathematical primitive.
- Tests that reproduce the implementation instead of checking independent behavior.
- Unnecessary mutation, global state, or hidden host-device synchronization.
- Type abstractions that obscure tensor shapes, frames, or ownership.
- Custom UOps without a clean Tensor reference and measured justification.
- Large generated-looking patches with inconsistent naming or no coherent design center.

#### Tinygrad pull-request evidence

The Critic's standards should be grounded in observed tinygrad maintenance practice, not invented aesthetics. This plan reviewed a representative sample of merged and closed-unmerged pull requests, their discussion, diff size, test demands, and maintainer feedback on 2026-07-26.

Tinygrad's official contribution guide explicitly states that:

- Low line count is a guide, but code golf is rejected; the real goals are lower complexity and higher readability.
- Claimed speedups require benchmarks and must justify their maintainability tradeoff.
- Large, complex, line-adding changes may not be reviewed.
- Prerequisite simplification should make later features small and obvious.
- Bug fixes and features need regression tests.
- APIs should generally match established Torch or NumPy behavior.
- Refactors must be clear wins and preserve generated behavior through process replay.
- Dead-code removal from the core is valuable.
- AI-looking contributions may be closed without review, and AI use must be disclosed.

Source: [tinygrad contribution guide](https://github.com/tinygrad/tinygrad#contributing).

Observed PR evidence:

| PR | Outcome | Maintainer signal |
| --- | --- | --- |
| [#15532](https://github.com/tinygrad/tinygrad/pull/15532) | Merged | A two-line removal of redundant traversal still required a representative benchmark before merge. |
| [#17017](https://github.com/tinygrad/tinygrad/pull/17017) | Merged | An O(n-squared) fix supplied timing and memory evidence, explained cache validity from immutable UOps, and added the pathological case as a regression test. |
| [#17049](https://github.com/tinygrad/tinygrad/pull/17049) | Merged after revision | The reviewer rejected locally reimplementing `backward_slice_with_self`; the change was revised to use the existing abstraction even though the local direct form benchmarked faster. |
| [#15925](https://github.com/tinygrad/tinygrad/pull/15925) | Merged after revision | The gradient rule had to match Torch and use the existing operation-test framework instead of a hyper-explicit bespoke test. |
| [#16122](https://github.com/tinygrad/tinygrad/pull/16122) | Merged | A coherent RMSNorm behavior change matched Torch, updated the API and tests together, and produced only a small net line increase after reuse. |
| [#15963](https://github.com/tinygrad/tinygrad/pull/15963) | Closed/superseded | Splitting an inseparable RMSNorm change across dependent PRs made review harder; the maintainer asked for one coherent change. |
| [#16070](https://github.com/tinygrad/tinygrad/pull/16070) | Closed | A speed-oriented reduction change added a new UOp across seven files; the maintainer stated the existing spec could express it and that complexity is never traded for speed. |
| [#17097](https://github.com/tinygrad/tinygrad/pull/17097) | Closed | A 1,835-line one-file speedup patch used simulated benchmark claims, omitted files, asked the maintainer to assemble attachments, and disclosed AI use only after challenge. |
| [#16899](https://github.com/tinygrad/tinygrad/pull/16899) | Closed | A generic 109-line scan API was closed with an explicit no-AI response, illustrating that polished prose and nominal tests do not overcome provenance and trust failures. |

These examples are evidence of recurring review preferences, not proof that every closed PR was technically wrong. Closure can also mean superseded work, wrong sequencing, or an untrusted review process. The Critic must cite the specific applicable rule rather than treating "closed upstream" as sufficient reasoning.

#### Tinygrad-derived Critic rules

The Critic enforces the following adapted rules for TinySim:

1. **Complexity, not characters, is the budget.** Reject code golf and reject abstraction bloat. Optimize for the fewest concepts needed to make the algorithm obvious.
2. **The burden of proof grows with the diff.** Three clear lines need modest justification; thirty need strong evidence; hundreds require decomposition or an exceptional whole-system payoff.
3. **Refactor toward the feature first.** If a feature requires a large awkward patch, look for a prerequisite simplification that makes the final semantic change small.
4. **Use the existing abstraction before inventing a local variant.** A locally faster duplicate helper is normally worse than one shared well-understood primitive. Improve the shared primitive if necessary.
5. **Do not add a new UOp merely for speed.** First prove the existing Tensor/UOp language cannot express the operation cleanly. Any new IR concept requires spec, renderer, backend, reference, and gradient consequences to be justified together.
6. **Never trade portable architectural simplicity for a marginal benchmark.** A faster special case that leaks across layers or disables untested backends is rejected.
7. **Every speed claim needs real reproducible measurement.** Require the benchmark command or script, device, dtype, shapes, warm-up, synchronization, compile-time handling, variance, baseline, and end-to-end effect. Simulated numbers are not benchmarks.
8. **Benchmark memory and invalidation risks as well as time.** Caches and memoization require lifetime, mutation, determinism, and peak-memory arguments.
9. **Turn the failure or complexity into a regression test.** Bug fixes test the bug; complexity fixes test the pathological scaling case; custom gradients test finite differences.
10. **Use the existing test framework.** Prefer concise parameterized reference checks over bespoke tests that mirror the implementation or hard-code excessive internals.
11. **Match a recognized semantic reference when the API claims compatibility.** For TinySim this may be analytical mechanics, NumPy conventions, MJCF meaning, or MuJoCo behavior. Intentional differences must be explicit.
12. **Make one coherent review unit.** Split independent prerequisite refactors, but keep inseparable API, implementation, and test changes together. Do not create a chain of PRs that forces the reviewer to reconstruct the final behavior mentally.
13. **The patch must be complete in the worktree.** No missing files, external attachments required for assembly, uncommitted generated pieces, or requests that the manager finish the implementation.
14. **Explain non-obvious correctness arguments.** Cache validity, ordering stability, frame transforms, masking, aliasing, and gradient semantics require evidence proportionate to risk.
15. **Deletion is a first-class improvement.** Prefer removing dead code, duplicated state, redundant passes, and obsolete options over adding another layer around them.
16. **Generated behavior must remain stable for behavior-preserving refactors.** TinySim's equivalent of process replay should compare eager and `TinyJit` results, kernel structure where relevant, and benchmark behavior before and after.
17. **No opaque AI output.** TinySim is intentionally agent-built, so a literal no-AI rule is impossible. Instead, every agent must disclose its role, explain every non-obvious choice, provide reproducible evidence, and accept immediate rejection of generated-looking code it cannot defend.
18. **Polished explanation does not rescue bad code.** Judge the diff, tests, and evidence. Long summaries, confident claims, and exhaustive bullet lists carry zero weight without a clean implementation.
19. **A speedup does not excuse a leaky abstraction.** If performance requires exposing backend details through physics or public model APIs, redesign the boundary.
20. **The manager should never assemble an agent's partial patch.** Incomplete work is returned or banned; implementation ownership stays with the implementer until an accepted handoff.

#### Critic verdicts

Every Critic report begins with exactly one verdict:

```text
ACCEPT
RETURN TO IMPLEMENTER
BAN FROM INTEGRATION
```

`ACCEPT` means no blocking cleanliness or abstraction finding remains.

`RETURN TO IMPLEMENTER` sends the work back to its original implementer with concrete blocking findings. The implementer revises the same task; the manager does not quietly repair it on the implementer's behalf.

`BAN FROM INTEGRATION` is a hard rejection when a patch or approach is irredeemably over-engineered, abstraction-leaking, generated-looking, or inconsistent with TinySim's design. It has mandatory agent-lifecycle consequences:

1. The manager immediately revokes the implementer's file and task ownership.
2. The old implementer agent is stopped and terminated.
3. The terminated agent may not continue editing, respond to findings, or be reused for the same task or phase.
4. The rejected patch is not integrated. Only diagnostic evidence, tests, or independently verified facts may be retained.
5. The manager spawns a fresh implementer agent.
6. The replacement receives the original frozen contract, authoritative context, the Critic report, and any verified evidence.
7. The replacement must produce a clean implementation and may not merely patch over the rejected abstraction.
8. The replacement passes independent validation and a new Critic review from the beginning.

The purpose of termination is to reset implementation context after the old agent has demonstrated a strong bias toward the rejected design. It prevents an agent from cosmetically rearranging the same abstraction while preserving its underlying mistakes.

Critic findings must include:

```text
Verdict:
Blocking findings with file and line:
Abstractions that leak or should not exist:
Code that should be deleted:
Simpler concrete design:
Non-blocking cleanup:
Evidence needed for reconsideration:
```

Volume is not evidence. Each complaint must identify a concrete defect, maintenance cost, leaked boundary, redundant concept, or simpler alternative. Personal attacks, vague aesthetic preferences, and demands to remove code required for correctness are invalid findings.

#### Critic workflow

1. The implementer completes focused tests and its completion report.
2. Independent numerical validation runs where required.
3. The Critic reviews the diff, surrounding interfaces, tests, and generated kernel implications.
4. A rejected task returns to the original implementer.
5. The same Critic or the manager verifies that every blocking finding is resolved.
6. Only an accepted task may proceed to manager integration.

The Critic does not edit the implementation while judging it. If it becomes the implementer of a replacement, another agent must perform the next Critic pass.

For `RETURN TO IMPLEMENTER`, step 4 applies normally. For `BAN FROM INTEGRATION`, step 4 is replaced by the termination-and-replacement procedure above. Before termination, the manager records the owned files, current diff, test evidence, and Critic findings so abandoned shared-worktree changes cannot be mistaken for accepted work. The fresh agent becomes the sole new owner of that scope.

The Critic gate is mandatory for:

- New shared abstractions or public APIs.
- Model and state representation changes.
- Spatial algebra and dynamics implementations.
- Collision and solver code.
- Custom UOps and custom backward functions.
- Patches with broad file scope.
- Phase exit candidates.
- Any patch the manager suspects contains AI slop.

### 19.10 Interface and decision control

Project-wide decisions require manager approval and documentation. These include:

- Coordinate and frame conventions.
- State and model tensor layouts.
- Public API changes.
- Contact sign and normal conventions.
- Solver semantics.
- Gradient semantics.
- Backend-specific fallbacks.
- Changes to phase acceptance criteria.

Small architecture decision records should state the decision, alternatives, evidence, consequences, and revision date. Subagents may propose decisions but must not unilaterally establish cross-project conventions.

### 19.11 Parallelization rules

Good parallel tasks:

- Implementation and independent reference-test construction.
- Spatial math work and benchmark-harness work in disjoint files.
- Separate collision primitive implementations after their common interface is frozen.
- Documentation research and an unrelated bounded implementation.

Tasks that should be sequential:

- Public API definition followed by consumers of that API.
- Coordinate-convention decisions followed by spatial algebra.
- Model layout changes followed by dynamics kernels.
- Tensor reference implementation followed by custom UOp optimization.
- Solver semantics followed by gradient implementation.

The manager should avoid spawning more work than it can inspect and integrate. Agent utilization is not itself a success metric.

### 19.12 Phase-oriented team allocation

| Phase | Suggested parallel allocation |
| --- | --- |
| 0 | Manager: skeleton; validation agent: test layout; performance agent: benchmark protocol |
| 1 | Physics agent: pendulum/cart-pole; validation agent: analytical and gradient references; performance agent: JIT scaling |
| 2 | Physics agent: spatial algebra; model agent: topology compiler; validation agent: frame and batch tests |
| 3 | Physics agent: CRBA/RNEA; Tinygrad agent: batched solve; validation agent: MuJoCo intermediates |
| 4 | Collision agent: primitives; physics agent: smooth contact; validation agent: geometry and gradient cases |
| 5 | Model agent: MJCF importer; validation agent: equivalent-model tests; manager: supported-subset contract |
| 6 | Physics agent: constraint formulation; Tinygrad agent: solver; validation agent: convergence and finite differences |
| 7 | Tinygrad agent: measured optimization; review agent: portability and backward audit; performance agent: end-to-end proof |
| 8 | Environment agent: RL API; performance agent: rollout profiling; validation agent: reset and batching semantics |
| 9 | Visualization agent: renderer and encoder adapter; validation agent: deterministic scenarios and manifests; manager: final release review |

Allocation changes when tasks are coupled. The manager should prefer a smaller coherent team over nominal parallelism.

Every phase candidate receives a Critic pass before the manager declares its exit criteria satisfied. The Critic normally replaces one worker slot temporarily after implementation work has yielded.

### 19.13 Multi-agent definition of done

A multi-agent work item is done only when:

- Its task packet acceptance criteria are satisfied.
- The diff remains within assigned scope.
- Focused tests pass.
- Independent evidence exists where required.
- Shared interfaces and documentation are updated.
- No active agent is unknowingly relying on an obsolete contract.
- Eager and `TinyJit` results agree when JIT support is claimed.
- Performance-sensitive work has no unexplained benchmark regression.
- Independent review has no unresolved blocking finding.
- The Critic verdict is `ACCEPT`.
- The manager runs integration tests and accepts the result.
- For a release candidate, the final numerical verification report passes and its
  deterministic demonstration videos and manifests are generated and reviewed.

### 19.14 Anti-patterns

Avoid:

- Multiple agents editing the same core file concurrently.
- Giving agents broad goals without file or interface boundaries.
- Treating an agent's self-report as integration evidence.
- Asking an optimizer to work before a correct reference exists.
- Allowing imported-engine behavior to define undocumented TinySim semantics.
- Creating a custom UOp without a portable Tensor reference.
- Letting each subsystem invent its own frame or shape conventions.
- Starting many tasks while review and integration are backlogged.
- Repeating large context dumps instead of maintaining stable repository documentation.

## 20. Risks and mitigations

### 20.1 Tinygrad API and compiler evolution

Risk: tinygrad is actively evolving and internal UOp APIs may change.

Mitigation:

- Pin a known commit during experiments.
- Keep direct UOp usage isolated under `tinysim/uops`.
- Prefer public Tensor operations.
- Maintain Tensor reference implementations.

### 20.2 Graph explosion

Risk: unrolled tree traversals, solvers, and long differentiable horizons can create very large graphs.

Mitigation:

- Group work by depth and operation type.
- Start with short horizons.
- Measure graph construction and compilation separately.
- Add custom function boundaries or checkpointing.
- Consider implicit solver derivatives.

### 20.3 Inefficient indexed operations

Risk: generic gather or scatter can generate dense one-hot work.

Mitigation:

- Inspect generated graphs.
- Use contiguous and linear indexing.
- Precompute incidence matrices for small models.
- Implement static CSR accumulation when justified.

### 20.4 Contact instability

Risk: penalty contact can require small timesteps or stiff parameters.

Mitigation:

- Begin with controlled test scenes.
- Expose stiffness, damping, and smoothing.
- Add adaptive test sweeps.
- Implement constraint-based contact as a later mode.

### 20.5 Misleading gradients

Risk: contact gradients may be discontinuous or depend heavily on smoothing.

Mitigation:

- Label gradient semantics.
- Validate with finite differences.
- Test both away from and near mode boundaries.
- Separate exact-forward and smooth-gradient modes if needed.

### 20.6 Reference mismatch

Risk: TinySim and MuJoCo may differ because of solver, contact, or integration semantics rather than implementation bugs.

Mitigation:

- Compare contact-free subsystems first.
- Match conventions and options explicitly.
- Compare intermediates.
- Use analytical references where possible.
- Document expected semantic differences.

### 20.7 Premature format work

Risk: implementing full MJCF, URDF, or USD support delays the compiler and physics experiment.

Mitigation:

- Use direct Python models in early phases.
- Implement only importer subsets required by validated examples.
- Keep importers separate from runtime structures.

## 21. Dependencies

Required initially:

- Python.
- The checked-out tinygrad source.
- NumPy for host-side model compilation and reference utilities.
- Pytest or the chosen minimal test runner.

Optional development dependencies:

- MuJoCo for MJCF parsing and numerical reference.
- Matplotlib for local diagnostics.
- Existing checked-out engines for algorithm and behavior study.

Runtime physics must not depend on:

- MuJoCo.
- JAX.
- PyTorch.
- Bullet.
- Isaac Sim.

## 22. Documentation requirements

Maintain:

- Coordinate and frame conventions.
- Supported model features.
- Unsupported-feature behavior.
- Contact model equations.
- Solver equations and termination semantics.
- Gradient semantics.
- Benchmark methodology.
- Backend support.
- Known numerical limitations.

Each example should state:

- Model parameters.
- Timestep.
- Integrator.
- Contact mode.
- Solver settings.
- Dtype.
- Expected behavior.

## 23. Initial decision log

Decisions made:

1. Use tinygrad Tensor operations before direct UOps.
2. Represent worlds with an explicit leading batch dimension.
3. Specialize compiled graphs to static model topology.
4. Use reduced coordinates.
5. Start with dense mass matrices and solves.
6. Start with semi-implicit Euler.
7. Implement smooth compliant contact before hard constraints.
8. Use Python as the first native model-authoring API.
9. Add an MJCF subset as the first file importer.
10. Keep MuJoCo optional and development-only.
11. Use fixed contact capacities and activity masks.
12. Require Tensor references and gradient tests for custom UOp kernels.

Open research questions:

- What batch sizes are required for acceptable GPU utilization?
- Does tinygrad fuse the spatial algebra workloads effectively?
- How much graph growth results from depth and solver unrolling?
- Which gather patterns lower efficiently?
- When does dense incidence multiplication outperform a CSR kernel?
- Does dense CRBA or an articulated-body algorithm perform better for common robot sizes?
- Which solve method provides the best correctness/performance trade-off?
- How should long-horizon gradients be represented without excessive memory?
- Should exact forward contact and differentiable contact be separate modes?
- Which model parameters can vary per world without causing recompilation?

## 24. Immediate next actions

1. Create the package and test skeleton.
2. Pin and record the tinygrad revision used for the first experiment.
3. Implement quaternion and small-vector math tests.
4. Implement a batched pendulum with analytical dynamics.
5. Implement semi-implicit Euler.
6. Wrap the step in `TinyJit`.
7. Test batches of 1, 256, and 4096 on CPU and an available accelerator.
8. Validate control gradients with finite differences.
9. Record compilation time, warm latency, kernels per step, and throughput.
10. Use the results as the Phase 1 go/no-go decision before implementing articulated dynamics.

## 25. References and local research sources

- Tinygrad developer architecture: <https://docs.tinygrad.org/developer/developer/>
- Tinygrad UOp reference: <https://docs.tinygrad.org/developer/uop/>
- Tinygrad runtime support: <https://docs.tinygrad.org/runtime/>
- MuJoCo modeling and MJCF: <https://mujoco.readthedocs.io/en/stable/modeling.html>
- MuJoCo model representations: <https://mujoco.readthedocs.io/en/stable/overview.html>
- MJX documentation: <https://mujoco.readthedocs.io/en/stable/mjx.html>
- SDFormat specification: <https://sdformat.org/>
- Brax differentiable physics paper: <https://arxiv.org/abs/2106.13281>
- Local tinygrad source: `tinygrad` git submodule
- Local MuJoCo and MJX source: `../mujoco` and `../mjx`
- Local Bullet source: `../bullet3`
- Local Isaac Lab and Isaac Sim source: `../IsaacLab` and `../IsaacSim`

## 26. Implementation status

Implementation began on 2026-07-26 against the revisions recorded in
`research-lock.json`. Status is evidence-based; a phase is not marked complete
when an acceptance criterion cannot be exercised in the current environment.

| Phase | Status | Evidence and remaining gap |
| --- | --- | --- |
| 0 | CPU complete; CUDA evidence in progress | Package, tests, reproducible lock, capability probe, and first/capture/warm benchmark matrix are implemented; the host RTX 2070 is available through tinygrad's CUDA backend when GPU device access is granted |
| 1 | Complete | Batched pendulum/cart-pole, semi-implicit Euler, finite-difference control gradients, and eager/`TinyJit` agreement |
| 2 | Complete | Fully documented spatial conventions, fixed/hinge/slide/ball/free layouts, compiler depth schedules, depth-grouped kinematics, batching, and float32/float64 tests |
| 3 | Complete | World-coordinate CRBA, independent body-Jacobian mass reference, recursive bias, Cholesky solve, actuation, analytical references, invariants, finite-difference gradients, and genuine MuJoCo pendulum/two-link intermediate comparison at `1e-9` |
| 4 | Complete | Fixed compiler-owned pairs; sphere-plane, sphere-sphere, sphere-capsule, capsule-plane, capsule-capsule, and box-plane queries; activity masks; integrated smooth normal/friction forces; gradients and bouncing rollout |
| 5 | Complete for the declared subset | Native MJCF options/defaults, translated trees, fixed/hinge/slide/ball/free joints, inertials, primitive geoms, friction, contact pairs/exclusions, actuators, keyframes, public `ModelSpec.from_mjcf`, and strict rejection paths; runtime remains MuJoCo-independent. The optional MuJoCo normalization fallback was not needed and is intentionally absent. |
| 6 | Complete for the phase deliverables | Articulated contact Jacobians, frictionless unilateral rows, compiled fixed iterations, warm starts, inactive masks, scalar joint limits, static support/impact tests, convergence measurements, and documented executed-graph gradient semantics |
| 7 | Profiling decision complete | No custom UOp is justified by current synchronized CPU evidence; portable Tensor references and quantitative reopening criteria are recorded in `docs/uop-decision.md` |
| 8 | CPU complete; CUDA evidence in progress | Gym-like batched environment, tensor-masked resets, per-world mass/friction/gain randomization, same-device rollout, representative free-base quadruped, and performance workloads are implemented |
| 9 | Implementation complete; release evidence in progress | Six deterministic scenario recordings, saved-trajectory reload before rendering, numerical checks, eager/`TinyJit` agreement, playback timing, rollout wall time/realtime factor, source/config manifests, strict `release_ready` gates, release report, PPM output, and isolated FFmpeg/GStreamer MP4 adapters are implemented. `release_ready=false` until the accelerator recording, Critic acceptance, MP4, and manager review evidence all pass. |

Canonical MuJoCo 3.10 Python bindings were installed only in `/tmp` for the
development-oracle run; the clean runtime remains dependency-free and the
reference suite skips explicitly when bindings are absent. The host RTX 2070 is
usable through tinygrad's CUDA backend when the process is granted GPU device
access; the driverless NV backend does not support its Turing architecture.
FFmpeg availability and the final accelerator recording are measured during
the source-frozen evidence run rather than assumed or silently waived.
