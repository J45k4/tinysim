# UOp optimization decision

Decision: keep the physics implementation in portable Tensor expressions for
this release. No custom UOp kernel is accepted yet.

The CPU evidence in `artifacts/benchmarks/cpu-evidence.json` records first call,
capture, warm replay, kernels, memory counters, backward cost, and subsystem
timings for analytical, articulated, and contact workloads. Warm captured
steps are already small fixed graphs (the recorded quick matrix reports 3–8
kernels per step). The dominant articulated cost in that evidence is initial
graph construction/capture and eager backward, not proof that a CSR reduction
or specialized solve kernel materially improves end-to-end warm rollout.

A custom kernel would currently add three risks without the required evidence:

- no accelerator result demonstrates a portable bottleneck;
- no like-for-like specialized-UOp baseline demonstrates material end-to-end
  improvement;
- solve and reduction sizes vary by compiled model, increasing maintenance and
  backward complexity.

The Tensor references remain explicit:

- `mass_matrix_jacobian` independently checks the CRBA result;
- `batched_solve` is the differentiable dense-solve reference;
- `projected_jacobi` is the fixed-iteration constraint reference;
- collision primitives are standalone Tensor functions.

Reopen this decision only when a synchronized representative benchmark shows a
subsystem consumes a material share of warm step time on a supported
accelerator. A proposal must include device, dtype, shapes, batch sizes,
warm-up, capture treatment, kernel counts, memory, forward/backward results,
the portable reference comparison, and an end-to-end improvement. Backend
details must remain behind the subsystem boundary.
