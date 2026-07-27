"""Small end-to-end systems built from TinySim's reusable physics pieces."""

import math

from tinygrad import Tensor, TinyJit

from .collision import ContactParams, sphere_plane
from .contact import smooth_contact


DEFAULT_BALL_CONTACT = ContactParams(
    stiffness=5_000.0,
    damping=100.0,
    friction=0.5,
    penetration_smoothing=1e-4,
    force_smoothing=1e-6,
    velocity_smoothing=1e-3,
)


def sphere_plane_step(
    position: Tensor,
    velocity: Tensor,
    *,
    mass: float = 1.0,
    radius: float = 0.25,
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81),
    timestep: float = 0.002,
    contact_params: ContactParams = DEFAULT_BALL_CONTACT,
) -> tuple[Tensor, Tensor]:
    """Advance batched spheres above the fixed z=0 plane.

    Position and velocity have shape ``[worlds, 3]``. The collision query
    describes the sphere as shape A and the plane as shape B, so its normal
    points downward. ``smooth_contact(...).force_a`` consequently points
    upward and is the force integrated into the sphere.
    """

    if position.ndim != 2 or position.shape[1] != 3:
        raise ValueError(f"position must have shape [worlds, 3], got {position.shape}")
    if velocity.shape != position.shape:
        raise ValueError(f"velocity must have shape {position.shape}, got {velocity.shape}")
    if (
        mass <= 0
        or radius <= 0
        or timestep <= 0
        or not all(math.isfinite(value) for value in (mass, radius, timestep))
    ):
        raise ValueError("mass, radius, and timestep must be finite and positive")
    if len(gravity) != 3 or not all(math.isfinite(value) for value in gravity):
        raise ValueError("gravity must contain three finite values")
    if velocity.dtype != position.dtype or velocity.device != position.device:
        raise TypeError("position and velocity dtype/device must match")

    plane_point = Tensor((0.0, 0.0, 0.0), device=position.device, dtype=position.dtype)
    plane_normal = Tensor((0.0, 0.0, 1.0), device=position.device, dtype=position.dtype)
    gravity_vector = Tensor(gravity, device=position.device, dtype=position.dtype)
    geometry = sphere_plane(position, radius, plane_point, plane_normal)
    contact_force = smooth_contact(geometry, velocity, velocity.zeros_like(), contact_params).force_a
    next_velocity = velocity + timestep * (gravity_vector + contact_force / mass)
    return position + timestep * next_velocity, next_velocity


def make_jitted_sphere_plane_step(
    *,
    mass: float = 1.0,
    radius: float = 0.25,
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81),
    timestep: float = 0.002,
    contact_params: ContactParams = DEFAULT_BALL_CONTACT,
) -> TinyJit:
    """Build a forward-only JIT with the physical parameters frozen.

    TinyJit replay does not retain an autograd graph across the captured call
    boundary. Use :func:`sphere_plane_step` for gradients.
    """

    def run(position: Tensor, velocity: Tensor) -> tuple[Tensor, Tensor]:
        outputs = sphere_plane_step(
            position,
            velocity,
            mass=mass,
            radius=radius,
            gravity=gravity,
            timestep=timestep,
            contact_params=contact_params,
        )
        Tensor.realize(*outputs)
        return outputs

    return TinyJit(run)
