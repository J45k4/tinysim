"""Quaternion and spatial-vector algebra.

Conventions:

* quaternions are ``(w, x, y, z)`` and rotate local vectors into parent/world;
* spatial vectors are angular-linear ``[omega, velocity]``;
* transforms act on column vectors.
"""

from tinygrad import Tensor

from .math import cross, dot, matvec, skew


def quat_conjugate(quaternion: Tensor) -> Tensor:
    if quaternion.shape[-1] != 4:
        raise ValueError("quaternion must end in four components")
    return Tensor.stack(
        quaternion[..., 0],
        -quaternion[..., 1],
        -quaternion[..., 2],
        -quaternion[..., 3],
        dim=-1,
    )


def quat_mul(left: Tensor, right: Tensor) -> Tensor:
    """Hamilton product; applies ``right`` then ``left`` to a vector."""
    if left.shape[-1] != 4 or right.shape[-1] != 4:
        raise ValueError("quaternion operands must end in four components")
    lw, lx, ly, lz = (left[..., i] for i in range(4))
    rw, rx, ry, rz = (right[..., i] for i in range(4))
    return Tensor.stack(
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        dim=-1,
    )


def quat_normalize(quaternion: Tensor, epsilon: float = 1e-12) -> Tensor:
    length = dot(quaternion, quaternion).sqrt()
    safe = (length > epsilon).where(length, 1.0)
    normalized = quaternion / safe.unsqueeze(-1)
    identity = Tensor.stack(
        length * 0.0 + 1.0, length * 0.0, length * 0.0, length * 0.0, dim=-1
    )
    return (length > epsilon).unsqueeze(-1).where(normalized, identity)


def quat_inverse(quaternion: Tensor, epsilon: float = 1e-12) -> Tensor:
    squared_length = dot(quaternion, quaternion)
    safe = (squared_length > epsilon).where(squared_length, 1.0)
    return quat_conjugate(quaternion) / safe.unsqueeze(-1)


def axis_angle(axis: Tensor, angle: Tensor, epsilon: float = 1e-12) -> Tensor:
    if axis.shape[-1] != 3:
        raise ValueError("axis must end in three components")
    axis_length = dot(axis, axis).sqrt()
    safe = (axis_length > epsilon).where(axis_length, 1.0)
    unit = axis / safe.unsqueeze(-1)
    half = angle * 0.5
    return quat_normalize(
        Tensor.stack(
            half.cos(),
            unit[..., 0] * half.sin(),
            unit[..., 1] * half.sin(),
            unit[..., 2] * half.sin(),
            dim=-1,
        )
    )


def quat_to_matrix(quaternion: Tensor) -> Tensor:
    """Returns a local-to-parent rotation matrix."""
    q = quat_normalize(quaternion)
    w, x, y, z = (q[..., i] for i in range(4))
    return Tensor.stack(
        Tensor.stack(1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y), dim=-1),
        Tensor.stack(2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x), dim=-1),
        Tensor.stack(2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y), dim=-1),
        dim=-2,
    )


def rotate(quaternion: Tensor, vector: Tensor) -> Tensor:
    return matvec(quat_to_matrix(quaternion), vector)


def quat_integrate(quaternion: Tensor, angular_velocity: Tensor, timestep: float | Tensor) -> Tensor:
    """Integrates parent-frame angular velocity with an exponential map."""
    speed = dot(angular_velocity, angular_velocity).sqrt()
    half_angle = speed * timestep * 0.5
    safe_speed = (speed > 1e-12).where(speed, 1.0)
    scale = half_angle.sin() / safe_speed
    scale = (speed > 1e-12).where(scale, timestep * 0.5)
    delta = Tensor.stack(
        half_angle.cos(),
        angular_velocity[..., 0] * scale,
        angular_velocity[..., 1] * scale,
        angular_velocity[..., 2] * scale,
        dim=-1,
    )
    return quat_normalize(quat_mul(delta, quaternion))


def motion_cross(motion: Tensor) -> Tensor:
    """Matrix for the angular-linear spatial motion cross product."""
    if motion.shape[-1] != 6:
        raise ValueError("spatial motion must have six components")
    angular, linear = motion[..., :3], motion[..., 3:]
    zeros = skew(angular) * 0.0
    return Tensor.cat(
        Tensor.cat(skew(angular), zeros, dim=-1),
        Tensor.cat(skew(linear), skew(angular), dim=-1),
        dim=-2,
    )


def force_cross(motion: Tensor) -> Tensor:
    """Dual spatial cross product, equal to ``-motion_cross(motion).T``."""
    matrix = motion_cross(motion)
    return -matrix.transpose(-1, -2)


def spatial_motion_transform(rotation: Tensor, translation: Tensor) -> Tensor:
    """Maps local-origin motion into parent coordinates.

    ``rotation`` maps local vectors to parent coordinates and ``translation``
    points from the parent origin to the local origin in parent coordinates.
    """
    if rotation.shape[-2:] != (3, 3) or translation.shape[-1] != 3:
        raise ValueError("expected rotation [...,3,3] and translation [...,3]")
    zeros = rotation * 0.0
    return Tensor.cat(
        Tensor.cat(rotation, zeros, dim=-1),
        Tensor.cat(skew(translation) @ rotation, rotation, dim=-1),
        dim=-2,
    )


def point_velocity(linear_velocity: Tensor, angular_velocity: Tensor, offset: Tensor) -> Tensor:
    return linear_velocity + cross(angular_velocity, offset)
