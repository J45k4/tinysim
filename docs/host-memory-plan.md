# Host-memory reduction plan

## Goal

Make simulation and recording use bounded host memory. A longer rollout or a
larger saved trajectory may use more disk, but it must not retain every sampled
state as Python objects.

This work targets CPU RAM. Device allocator usage is measured separately and is
not hidden inside the host-memory result.

## Current behavior

The unrecorded `Simulation.run` path does not call `Tensor.tolist`, and its
TinyJit inference outputs are detached each step. That path should remain
host-read-free.

Recording currently does all of the following:

1. Copies selected `qpos`, `qvel`, and `ctrl` tensors to nested Python lists at
   every captured step.
2. Retains all those lists until the rollout finishes.
3. Calls `dataclasses.asdict`, which deep-copies the retained trajectory.
4. Calls `json.dumps`, which creates a complete text representation in memory.
5. Reads and parses the complete JSON file again before rendering.

The renderer already yields frames to the encoder one at a time. Frame encoding
is therefore not the unbounded part of the pipeline.

The existing 10-level Jenga recording provides a useful baseline:

- 256 simulated worlds, 64 recorded worlds
- 1,000 simulation steps, sampled every 10 steps
- 101 states
- 13,440 `qpos` and 11,520 `qvel` values per state
- 2,520,960 total stored float values
- 9.6 MiB of raw float32 data
- 64.4 MiB JSON trajectory
- about 192 MiB process peak RSS when only loading and validating that JSON
- about 424 MiB process peak RSS when loading and saving a copy through the
  current implementation
- about 31 MiB process peak RSS for the import-only baseline used in the same
  measurement environment

These are high-water RSS measurements from fresh processes. They demonstrate
the representation problem, not an exact decomposition of every allocation in
an end-to-end CUDA run.

## Target architecture

```text
device simulation state
        |
        | one selected, packed sample
        v
compact host memoryview
        |
        v
versioned trajectory stream on disk
        |
        | sequential or fixed-offset reads
        v
one qpos sample -> one RGB frame -> encoder stdin
```

At any time the host should retain only:

- one packed trajectory sample;
- one trajectory read buffer;
- one rendered RGB frame;
- fixed-size encoder and metadata state.

No component in this path may keep a Python list per recorded frame.

## Public behavior

Keep the existing entry points:

```python
simulation.run(steps=2000, record="rollout.mp4")
simulation.record("rollout.mp4", steps=2000)
```

A recorded run will write the video and a compact
`rollout.trajectory.tstraj` sidecar. The old in-memory `Trajectory` and JSON
load/save functions remain available for small verification artifacts and
explicit interchange, but `Simulation.record` will no longer use them as its
working representation.

The binary sidecar is a product artifact, not an internal cache. It must be
versioned, deterministic, validated on read, and documented. A later explicit
JSON export can stream from this file without loading it completely.

## Implementation phases

### 1. Establish a recording-memory benchmark

Add a benchmark that runs each case in a fresh subprocess and reports:

- process peak RSS from `resource.getrusage`;
- current process RSS from `/proc/self/status` when available;
- tinygrad allocator-resident bytes;
- trajectory and video file sizes;
- capture count, selected world count, `nq`, `nv`, and `nu`;
- elapsed simulation, trajectory writing, rendering, and encoding time.

Measure these cases:

- unrecorded bouncing ball, 256 worlds;
- one-world and 64-world bouncing-ball recording;
- 10-level Jenga, 256 simulated worlds and 64 recorded worlds;
- a synthetic trajectory length sweep that isolates storage at 101, 1,001,
  and 10,001 samples.

Sample RSS at phase boundaries: after import, compile, warmup, rollout, sidecar
close, render, and completion. Do not use one process for several cases because
`ru_maxrss` is a lifetime high-water value.

### 2. Add a minimal streaming trajectory format

Introduce a small `TrajectoryWriter` and `TrajectoryReader` in
`tinysim/trajectory.py`.

The file contains:

- magic bytes and a format version;
- little-endian numeric dtype (`float32` or `float64`);
- frame count and fixed widths for `qpos`, `qvel`, and `ctrl`;
- timestep and selected-world metadata;
- a canonical JSON metadata header;
- fixed-width frame records.

The writer must:

- write one frame immediately;
- accept compact bytes or a `memoryview`, not nested lists;
- update the final frame count when closed;
- write to a same-directory temporary file and atomically replace the target;
- remove its partial file after an exception.

The reader must:

- validate magic, version, dtype, dimensions, and exact file length;
- expose metadata without loading frame data;
- read or seek to one fixed-width frame at a time;
- yield qpos as a `memoryview` usable by the existing host renderer;
- reject truncated, oversized, or non-finite trajectories with clear errors.

Do not add compression initially. Fixed-width uncompressed records are shorter,
faster, deterministic, seekable, and sufficient to remove the Python-object
overhead.

### 3. Pack and transfer one sample

Replace the three per-capture `tolist` calls in `Simulation.run` with one
helper that:

1. selects only the requested worlds;
2. packs the requested state fields into one contiguous tensor;
3. realizes that tensor once;
4. obtains its compact host representation through `Tensor.data()`;
5. writes it immediately and releases the host buffer.

Video rendering only needs `qpos`. The working spool used for a video-only
recording should therefore contain only `qpos`. If the retained trajectory
sidecar promises full state, the packed record includes `qpos`, `qvel`, and
`ctrl` in the same transfer.

Keep this as a recording concern. Do not alter physics kernels or introduce a
recording-specific UOp.

### 4. Render from the stream

After simulation:

- close the trajectory writer;
- iterate playback indices lazily;
- read only the qpos record needed for the next video frame;
- render it;
- write the resulting RGB bytes directly to the encoder.

Replace the allocating `playback_indices` list in this path with an iterator so
very long videos also remain bounded. Repeated playback indices may reuse one
read sample; skipped indices may seek over fixed-width records.

The existing renderer and `encode_mp4` streaming contract stay intact. Do not
add a producer thread or queue in this phase.

### 5. Separate retained artifacts from temporary spooling

Use one implementation for both modes:

- retained trajectory: atomically finalize the `.tstraj` sidecar;
- video without a retained trajectory, if exposed later: render from a
  temporary `.tstraj` spool and delete it after successful encoding.

The public API should specify artifact retention directly rather than exposing
buffer sizes, chunks, queues, or other storage mechanics.

### 6. Stream optional JSON interchange

Once the binary path is stable, add an explicit streaming exporter from
`.tstraj` to the existing JSON schema. It should write arrays incrementally and
never call `asdict` or build the complete JSON string.

JSON import may remain an in-memory compatibility operation initially. If large
JSON imports are required later, address them separately rather than adding a
JSON parser to the simulation loop.

### 7. Consider bounded overlap only after profiling

If sequential simulation followed by rendering is too slow, allow a producer
and consumer to overlap through a queue with a fixed capacity of one or two
samples. The queue must apply backpressure and preserve deterministic ordering.

This is optional. It is justified only if benchmarks show material end-to-end
improvement; it is not needed to solve memory growth.

## Validation

### Correctness

- Final simulation state is bit-identical with recording disabled and enabled.
- Small trajectories round-trip through `.tstraj` for float32 and float64.
- Selected-world ordering is preserved.
- Empty control width and distinct `nq`/`nv` widths work.
- Frame count, timestep, and metadata survive round-trip.
- Truncated files, bad versions, wrong lengths, and non-finite values fail.
- Renderer frame hashes match the current implementation for the same qpos
  samples.
- Playback duration and sampled indices remain unchanged.
- Encoder failures do not leave a finalized corrupt sidecar.

### Memory gates

- Unrecorded runs continue to perform no state host read.
- Peak RSS attributable to recording is no more than 32 MiB above the matching
  warmed-up unrecorded process for the 10-level Jenga case.
- Increasing the synthetic trajectory from 101 to 10,001 samples increases
  peak RSS by no more than 8 MiB.
- The full Jenga `.tstraj` is no larger than raw numeric payload plus the
  greater of 1% or 64 KiB of format overhead.
- The video-only path never stores qvel or ctrl samples.
- Each captured full-state sample performs one packed device-to-host transfer,
  not one transfer per field or world.

### Performance gates

- Unrecorded step time and tinygrad resident memory regress by less than 5%.
- Recording throughput is reported, not hidden; memory improvements must not
  silently change `record_every`, selected worlds, resolution, or frame rate.
- The benchmark records simulation, storage, and rendering time separately so
  a faster-looking result cannot come from omitting work.

## Implementation order and review gates

1. Land the benchmark and reproduce the current baseline.
2. Land `TrajectoryWriter`/`TrajectoryReader` with format tests.
3. Integrate the writer into `Simulation.run`.
4. Render from the reader and remove the save/reload JSON path.
5. Update examples and documentation.
6. Run the full test suite and CPU/CUDA memory matrix.
7. Accept the change only when all correctness and memory gates pass.

Every phase should remain independently testable. If a patch mixes the binary
format, simulation transfer logic, rendering, and concurrency in one change, it
should be split before review.

## Non-goals

- changing rigid-body dynamics;
- reducing GPU allocator memory in the same patch;
- GPU rendering;
- trajectory compression;
- background threads before the sequential bounded-memory path is measured;
- custom kernels whose only purpose is artifact recording.
