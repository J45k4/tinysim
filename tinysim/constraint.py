"""Fixed-capacity frictionless contact constraint helpers."""

import math

from tinygrad import Tensor, dtypes

from .compile import CompiledModel
from .solver import projected_jacobi


def joint_limit_rows(
    model: CompiledModel,
    qpos: Tensor,
    qvel: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Builds fixed lower/upper unilateral rows for scalar joint limits."""

    batch = qpos.shape[0]
    identity = model.dof_identity
    jacobians: list[Tensor] = []
    velocities: list[Tensor] = []
    penetrations: list[Tensor] = []
    activities: list[Tensor] = []
    for joint, spec in enumerate(model.joints):
        if spec.limit is None:
            continue
        qadr, dof = model.joint_qpos[joint], model.joint_dof[joint]
        lower, upper = spec.limit
        coordinate = qpos[:, qadr]
        velocity = qvel[:, dof]
        unit = identity[dof].unsqueeze(0).expand(batch, model.nv)
        jacobians.extend((unit, -unit))
        velocities.extend((velocity, -velocity))
        penetrations.extend((lower - coordinate, coordinate - upper))
        activities.extend(
            (
                coordinate <= lower + model.contact.margin,
                coordinate >= upper - model.contact.margin,
            )
        )
    count = len(jacobians)
    if count == 0:
        return (
            Tensor.zeros(
                batch, 0, model.nv, dtype=qpos.dtype, device=qpos.device
            ),
            Tensor.zeros(batch, 0, dtype=qpos.dtype, device=qpos.device),
            Tensor.zeros(batch, 0, dtype=qpos.dtype, device=qpos.device),
            Tensor.zeros(
                batch, 0, dtype=dtypes.bool, device=qpos.device
            ),
        )
    return (
        Tensor.stack(*jacobians, dim=1),
        Tensor.stack(*velocities, dim=1),
        Tensor.stack(*penetrations, dim=1),
        Tensor.stack(*activities, dim=1),
    )


def contact_system(
    inverse_mass: Tensor,
    jacobian: Tensor,
    normal_velocity: Tensor,
    penetration: Tensor,
    active: Tensor,
    *,
    timestep: float,
    stabilization: float = 0.2,
    regularization: float = 1e-6,
) -> tuple[Tensor, Tensor]:
    """Builds ``A lambda = b`` for fixed frictionless normal rows.

    Shapes are ``inverse_mass [B, V, V]``, ``jacobian [B, C, V]`` and
    per-contact arrays ``[B, C]``. Inactive rows are masked but never removed.
    """
    if timestep <= 0 or not math.isfinite(timestep):
        raise ValueError("timestep must be positive")
    if (
        stabilization < 0
        or regularization < 0
        or not math.isfinite(stabilization)
        or not math.isfinite(regularization)
    ):
        raise ValueError("stabilization and regularization must be finite and nonnegative")
    if inverse_mass.ndim != 3 or jacobian.ndim != 3:
        raise ValueError("inverse_mass and jacobian must be batched matrices")
    if inverse_mass.shape[1] != inverse_mass.shape[2]:
        raise ValueError("inverse_mass must be square")
    if jacobian.shape[0] != inverse_mass.shape[0]:
        raise ValueError("world dimensions differ")
    if jacobian.shape[-1] != inverse_mass.shape[-1]:
        raise ValueError("jacobian velocity dimension differs from inverse mass")
    contact_shape = (jacobian.shape[0], jacobian.shape[1])
    if (
        normal_velocity.shape != contact_shape
        or penetration.shape != contact_shape
        or active.shape != contact_shape
    ):
        raise ValueError(f"contact arrays must have shape {contact_shape}")
    if active.dtype != dtypes.bool:
        raise TypeError("active contact mask must have boolean dtype")
    numeric = (inverse_mass, jacobian, normal_velocity, penetration)
    if any(tensor.device != jacobian.device for tensor in numeric):
        raise ValueError("constraint tensors must share a device")
    if any(tensor.dtype != jacobian.dtype for tensor in numeric):
        raise TypeError("constraint numeric tensors must share a dtype")

    mask = active.cast(jacobian.dtype)
    masked_jacobian = jacobian * mask.unsqueeze(-1)
    effective_mass = masked_jacobian @ inverse_mass @ masked_jacobian.transpose(-1, -2)
    identity = Tensor.eye(jacobian.shape[1], dtype=jacobian.dtype).to(
        jacobian.device
    ).unsqueeze(0)
    matrix = effective_mass + identity * regularization
    correction = stabilization * penetration.maximum(0.0) / timestep
    rhs = mask * (-normal_velocity + correction)

    # Make inactive rows a harmless identity equation with zero right-hand side.
    inactive = 1.0 - mask
    matrix = (
        matrix * mask.unsqueeze(-1) * mask.unsqueeze(-2)
        + identity * inactive.unsqueeze(-1)
    )
    return matrix, rhs


def solve_contact_impulses(
    inverse_mass: Tensor,
    jacobian: Tensor,
    normal_velocity: Tensor,
    penetration: Tensor,
    active: Tensor,
    *,
    timestep: float,
    iterations: int = 8,
    stabilization: float = 0.2,
    regularization: float = 1e-6,
    initial_impulse: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Returns nonnegative impulses and the resulting generalized velocity change."""
    matrix, rhs = contact_system(
        inverse_mass,
        jacobian,
        normal_velocity,
        penetration,
        active,
        timestep=timestep,
        stabilization=stabilization,
        regularization=regularization,
    )
    impulses = projected_jacobi(
        matrix, rhs, iterations=iterations, initial=initial_impulse
    )
    impulses = impulses * active.cast(impulses.dtype)
    delta_velocity = (
        inverse_mass @ jacobian.transpose(-1, -2) @ impulses.unsqueeze(-1)
    ).squeeze(-1)
    return impulses, delta_velocity
