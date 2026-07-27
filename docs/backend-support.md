# Backend and optional-tool support

CPU float32 and float64 are the validated development paths. Tests select a
backend with `DEV`; the package does not hard-code CPU tensors.

Run the capability probe:

```sh
PYTHONPATH=.:tinygrad DEV=CPU python3 tools/check_capabilities.py
```

Capabilities are measured in the source-frozen evidence run because device
nodes can differ between sandboxed and host execution. This host's RTX 2070 has
been exercised through tinygrad's CUDA backend; the driverless NV backend does
not support its Turing architecture. The verification runner also produces
saved trajectories, manifests, release reports, and deterministic PPM frames.
MP4 export uses either FFmpeg or an isolated GStreamer/OpenH264 pipeline without
changing the physics dependencies.

MuJoCo is optional validation software. With canonical bindings installed, the
reference suite compares transforms, COM positions, mass matrices, bias,
actuation, acceleration, and one integrated step. The benchmark harness also
offers an optional JIT+VMAP MJX baseline for compatible articulated workloads.
MuJoCo, JAX/MJX, Bullet, and Isaac are never imported by the TinySim runtime.
