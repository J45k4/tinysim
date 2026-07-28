# Physics and gradient contract

TinySim uses right-handed SI coordinates and column-vector transforms.
Quaternions are `(w, x, y, z)` and rotate local vectors into parent/world
coordinates. Spatial vectors are angular-linear. Body inertia is diagonal in
the body frame and is interpreted about the center of mass.

## Generalized coordinates

| Joint | `qpos` | `qvel` |
|---|---|---|
| fixed | — | — |
| hinge | angle | angular speed along the joint axis |
| slide | displacement | linear speed along the joint axis |
| ball | unit quaternion | three parent-joint-frame angular components |
| free | translation, unit quaternion | three linear then three angular components |

Ball/free quaternions are normalized by kinematics and after exponential-map
integration. The compiler gives `qpos` and `qvel` separate addresses; code must
not assume `nq == nv`.

The mass matrix uses a world-coordinate composite-rigid-body recursion. The
retained body-Jacobian construction is an independent test oracle. Bias forces
use recursive Newton–Euler expressions. Dense Cholesky is the reference solve.

## Collision and contact

The compiler fixes collision pairs and shape-specific manifold capacity:
sphere pairs use one slot, capsule-plane and parallel capsule-capsule pairs use
up to two, and box pairs use up to four. Supported narrow-phase pairs are
sphere-plane, sphere-sphere, sphere-capsule, capsule-plane, capsule-capsule,
box-plane, and box-box. Signed distance is positive when separated. The normal
points from shape A to shape B; forces on A and B are equal and opposite.

Box manifolds use the SAT normal, support-face vertices, and projection onto
the reference face. Duplicate or geometrically inactive slots are masked.
Smooth contact divides a pair's compliant load across its active slots so
adding a manifold does not multiply stiffness. Constraint mode reserves one
fixed unilateral row per possible slot.

`smooth` mode uses a smooth positive penetration, a damped normal load, and a
regularized Coulomb direction. Stiffness, damping, friction, and smoothing
scales are compile-time semantics, while per-world geom friction may be
randomized without changing topology.

`constraint` mode builds fixed frictionless unilateral rows:

```text
A = J M^-1 J^T + regularization
b = -normal_velocity + stabilization * penetration / timestep
lambda = projected_jacobi(A, b)
```

Inactive rows are masked, the iteration count is compiled into the graph, and
the previous impulse is the next step's warm start. Scalar hinge/slide limits
use two additional unilateral rows. Constraint friction, equality constraints,
mesh collision, restitution controls, and general external wrenches are not
silently approximated; they remain unsupported.

## Gradient semantics

- Kinematics, CRBA/RNEA dynamics, dense solve, integration, and smooth contact
  use tinygrad automatic differentiation.
- Fixed-iteration constraint gradients, when requested, differentiate the
  executed unrolled iteration graph.
- Contact activation and nonnegative projection are piecewise operations.
  Their derivative is the derivative of the executed branch, not an implicit
  derivative of a converged complementarity problem.
- `Simulator.step` is the differentiable path.
  `Simulator.inference_step` is forward-only `TinyJit` replay and deliberately
  returns detached dynamic state.

Tests cover float32/float64 algebra, finite differences for control, initial
state, mass, inertia, actuator gain, stiffness, and friction, plus documented
contact tests away from activation boundaries.
