"""Differentiable compliant contact forces for fixed collision candidates."""

from tinygrad import Tensor

from .collision.types import ContactForces, ContactGeometry, ContactParams, column


def _smooth_positive(value: Tensor, epsilon: float | int | Tensor) -> Tensor:
    """Smooth ReLU with bounded behavior for large positive or negative input."""

    return (value + (value.square() + epsilon * epsilon).sqrt()) * 0.5


def smooth_contact(
    geometry: ContactGeometry,
    velocity_a: Tensor,
    velocity_b: Tensor,
    params: ContactParams = ContactParams(),
) -> ContactForces:
    """Computes equal-and-opposite forces at the two contact witnesses.

    Velocities are Cartesian point velocities. Since the geometry normal points
    A-to-B, negative normal relative velocity means closing motion and the
    repulsive normal force on A points along ``-normal``.
    """

    if velocity_a.shape != geometry.normal.shape or velocity_b.shape != geometry.normal.shape:
        raise ValueError("contact velocities must match the normal shape")

    relative_velocity = velocity_b - velocity_a
    normal_velocity = (relative_velocity * geometry.normal).sum(axis=-1)
    tangent_velocity = relative_velocity - geometry.normal * normal_velocity.unsqueeze(-1)

    penetration = _smooth_positive(-geometry.distance, params.penetration_smoothing)
    unprojected_load = params.stiffness * penetration - params.damping * normal_velocity
    normal_force = _smooth_positive(unprojected_load, params.force_smoothing)
    normal_force = normal_force * geometry.active.cast(normal_force.dtype)

    tangent_speed = (
        tangent_velocity.square().sum(axis=-1) + params.velocity_smoothing**2
    ).sqrt()
    friction_force_a = (
        tangent_velocity
        * column(params.friction * normal_force / tangent_speed, tangent_velocity)
    )
    force_a = -geometry.normal * normal_force.unsqueeze(-1) + friction_force_a
    return ContactForces(
        force_a=force_a,
        force_b=-force_a,
        normal_force=normal_force,
        friction_force_a=friction_force_a,
        normal_velocity=normal_velocity,
        tangent_velocity=tangent_velocity,
    )
