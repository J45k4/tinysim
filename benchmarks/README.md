# TinySim performance harness

The matrix runner benchmarks six workloads: analytical pendulum, analytical
cart-pole, an 8-DoF quadruped-like tree, a 12-DoF humanoid-scale tree, and
smooth sphere-plane contact, plus a representative free-base
Environment/policy/contact step. The default batch matrix is
`1, 16, 64, 256, 1024, 4096, 16384`.

From the TinySim repository:

```bash
CACHEDB=/tmp/tinysim-bench.db PYTHONPATH=.:tinygrad DEV=CPU \
  python3 -m benchmarks.bench_matrix \
  --output artifacts/benchmarks/cpu-full.json
```

Use bounded quick mode for CI and smoke testing. It runs every workload at one
world, one warm replay, and omits the expensive backward/subsystem sections:

```bash
CACHEDB=/tmp/tinysim-bench-quick.db PYTHONPATH=.:tinygrad DEV=CPU \
  python3 -m benchmarks.bench_matrix --quick \
  --output artifacts/benchmarks/cpu-quick.json
```

Every case separates the first uncaptured call, the capture/compile call, and
warm TinyJit replay. Wall time is bracketed by backend synchronization. The JSON
also records executed kernels, device-kernel time where supplied by tinygrad,
operation and memory-access counters, allocator-resident bytes, process
high-water RSS, forward and backward throughput, subsystem timings, scalar
Python baselines where comparable, model dimensions, solver/contact settings,
dtype, revisions, warm-up protocol, and scaling efficiency against batch 1.

The memory numbers do not pretend to be exact per-call peaks:
`GlobalCounters.mem_used_per_device` is a current allocator snapshot and
`resource.getrusage` is a process-lifetime high-water mark. Device utilization
is marked unavailable because tinygrad does not expose a portable utilization
counter here. Canonical MuJoCo `mj_step` and JIT+VMAP MJX baselines are
available for compatible contact-free articulated workloads when their optional
bindings are installed. Specialized-UOp results remain explicitly unavailable
until profiling justifies such a kernel.

For focused investigation, select workloads or batches and disable an expensive
section by setting its sample count to zero:

```bash
python3 -m benchmarks.bench_matrix \
  --workloads pendulum cartpole \
  --worlds 1 64 4096 \
  --warm-steps 50 --backward-steps 5 \
  --subsystem-steps 0 --baseline-steps 100
```

`python3 -m benchmarks.bench_compile` is a one-world compile/capture shortcut;
`python3 -m benchmarks.bench_batch_scaling` runs forward scaling without
backward or subsystem samples. Both accept the matrix runner's CLI options.

The recording-memory benchmark runs every case in a fresh subprocess so
process high-water RSS remains attributable to one workload:

```bash
PYTHONPATH=.:tinygrad DEV=CPU \
  python3 -m benchmarks.bench_recording_memory \
  --output artifacts/benchmarks/recording-memory-quick.json
```

The full acceptance run includes the 10-level Jenga workload with 256 simulated
worlds and a 64-world recording grid:

```bash
PYTHONPATH=.:tinygrad DEV=CUDA \
  python3 -m benchmarks.bench_recording_memory --full \
  --output artifacts/benchmarks/recording-memory-cuda.json
```

It reports current and peak process RSS, tinygrad allocator-resident bytes,
trajectory/video sizes, and elapsed time. Synthetic 101-, 1,001-, and
10,001-frame cases verify that trajectory length changes disk usage without
causing proportional host-memory growth.
