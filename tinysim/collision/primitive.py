"""Batched, tensor-only narrow phase for the initial collision primitives."""

from tinygrad import Tensor

from ..math import cross, dot
from ..spatial import quat_conjugate, quat_to_matrix, rotate
from .types import ContactGeometry, Scalar, column


def _check_vector(name: str, value: Tensor) -> None:
    if value.ndim < 1 or value.shape[-1] != 3:
        raise ValueError(f"{name} must end in dimension 3")


def _unit(vector: Tensor, fallback: tuple[float, float, float], epsilon: float) -> tuple[Tensor, Tensor]:
    squared = dot(vector, vector)
    length = squared.sqrt()
    safe_length = squared.maximum(epsilon * epsilon).sqrt()
    default = Tensor(fallback, device=vector.device, dtype=vector.dtype)
    unit = (squared > epsilon * epsilon).unsqueeze(-1).where(
        vector / safe_length.unsqueeze(-1), default
    )
    return length, unit


def _result(
    distance: Tensor,
    normal: Tensor,
    point_a: Tensor,
    point_b: Tensor,
    margin: Scalar,
) -> ContactGeometry:
    vector_shape = distance.shape + (3,)
    normal = normal.expand(vector_shape)
    point_a = point_a.expand(vector_shape)
    point_b = point_b.expand(vector_shape)
    return ContactGeometry(distance, normal, point_a, point_b, distance <= margin)


def sphere_plane(
    center: Tensor,
    radius: Scalar,
    plane_point: Tensor,
    plane_normal: Tensor,
    *,
    margin: Scalar = 0.0,
    epsilon: float = 1e-12,
) -> ContactGeometry:
    """Collides sphere A with one-sided plane B.

    ``plane_normal`` points out of B's solid half-space. The returned contact
    normal therefore points in the opposite direction, from the sphere to the
    plane. Inputs may carry any broadcast-compatible leading dimensions.
    """

    _check_vector("center", center)
    _check_vector("plane_point", plane_point)
    _check_vector("plane_normal", plane_normal)
    _, outward = _unit(plane_normal, (0.0, 0.0, 1.0), epsilon)
    center_height = dot(center - plane_point, outward)
    radius_column = column(radius, center)
    point_a = center - outward * radius_column
    point_b = center - outward * center_height.unsqueeze(-1)
    return _result(center_height - radius, -outward, point_a, point_b, margin)


def sphere_sphere(
    center_a: Tensor,
    radius_a: Scalar,
    center_b: Tensor,
    radius_b: Scalar,
    *,
    margin: Scalar = 0.0,
    epsilon: float = 1e-12,
) -> ContactGeometry:
    """Collides spheres A and B with a deterministic normal when concentric."""

    _check_vector("center_a", center_a)
    _check_vector("center_b", center_b)
    center_distance, normal = _unit(center_b - center_a, (1.0, 0.0, 0.0), epsilon)
    point_a = center_a + normal * column(radius_a, normal)
    point_b = center_b - normal * column(radius_b, normal)
    return _result(center_distance - radius_a - radius_b, normal, point_a, point_b, margin)


def sphere_capsule(
    sphere_center: Tensor,
    sphere_radius: Scalar,
    capsule_start: Tensor,
    capsule_end: Tensor,
    capsule_radius: Scalar,
    *,
    margin: Scalar = 0.0,
    epsilon: float = 1e-12,
) -> ContactGeometry:
    """Collides sphere A with capsule B, including zero-length capsules."""

    _check_vector("sphere_center", sphere_center)
    _check_vector("capsule_start", capsule_start)
    _check_vector("capsule_end", capsule_end)
    segment = capsule_end - capsule_start
    segment_sq = dot(segment, segment)
    projection = dot(sphere_center - capsule_start, segment)
    parameter = projection / segment_sq.maximum(epsilon * epsilon)
    parameter = (segment_sq > epsilon * epsilon).where(parameter, 0.0).clamp(0.0, 1.0)
    axis_point = capsule_start + segment * parameter.unsqueeze(-1)
    axis_distance, normal = _unit(axis_point - sphere_center, (1.0, 0.0, 0.0), epsilon)
    point_a = sphere_center + normal * column(sphere_radius, normal)
    point_b = axis_point - normal * column(capsule_radius, normal)
    distance = axis_distance - sphere_radius - capsule_radius
    return _result(distance, normal, point_a, point_b, margin)


def capsule_plane(
    capsule_start: Tensor,
    capsule_end: Tensor,
    capsule_radius: Scalar,
    plane_point: Tensor,
    plane_normal: Tensor,
    *,
    margin: Scalar = 0.0,
    epsilon: float = 1e-12,
) -> ContactGeometry:
    """Collides capsule A with one-sided plane B using its lowest endpoint."""

    _check_vector("capsule_start", capsule_start)
    _check_vector("capsule_end", capsule_end)
    _check_vector("plane_point", plane_point)
    _check_vector("plane_normal", plane_normal)
    _, outward = _unit(plane_normal, (0.0, 0.0, 1.0), epsilon)
    start_height = dot(capsule_start - plane_point, outward)
    end_height = dot(capsule_end - plane_point, outward)
    use_start = start_height <= end_height
    axis_point = use_start.unsqueeze(-1).where(capsule_start, capsule_end)
    height = use_start.where(start_height, end_height)
    point_a = axis_point - outward * column(capsule_radius, outward)
    point_b = axis_point - outward * height.unsqueeze(-1)
    return _result(height - capsule_radius, -outward, point_a, point_b, margin)


def capsule_capsule(
    start_a: Tensor,
    end_a: Tensor,
    radius_a: Scalar,
    start_b: Tensor,
    end_b: Tensor,
    radius_b: Scalar,
    *,
    margin: Scalar = 0.0,
    epsilon: float = 1e-12,
) -> ContactGeometry:
    """Collides two capsules using robust closest points on their axis segments."""

    for name, value in (
        ("start_a", start_a), ("end_a", end_a),
        ("start_b", start_b), ("end_b", end_b),
    ):
        _check_vector(name, value)
    direction_a, direction_b = end_a - start_a, end_b - start_b
    relative = start_a - start_b
    aa, bb = dot(direction_a, direction_a), dot(direction_b, direction_b)
    ab = dot(direction_a, direction_b)
    ar, br = dot(direction_a, relative), dot(direction_b, relative)
    safe_aa, safe_bb = aa.maximum(epsilon), bb.maximum(epsilon)
    denominator = aa * bb - ab * ab
    parameter_a = (
        (denominator > epsilon).where((ab * br - ar * bb) / denominator.maximum(epsilon), 0.0)
    ).clamp(0.0, 1.0)
    parameter_b = ((ab * parameter_a + br) / safe_bb).clamp(0.0, 1.0)
    parameter_a = ((ab * parameter_b - ar) / safe_aa).clamp(0.0, 1.0)
    parameter_b = (bb > epsilon).where(parameter_b, 0.0)
    parameter_a = (aa > epsilon).where(parameter_a, 0.0)
    point_axis_a = start_a + direction_a * parameter_a.unsqueeze(-1)
    point_axis_b = start_b + direction_b * parameter_b.unsqueeze(-1)
    axis_distance, normal = _unit(point_axis_b - point_axis_a, (1.0, 0.0, 0.0), epsilon)
    point_a = point_axis_a + normal * column(radius_a, normal)
    point_b = point_axis_b - normal * column(radius_b, normal)
    return _result(axis_distance - radius_a - radius_b, normal, point_a, point_b, margin)


def box_plane(
    center: Tensor,
    half_size: Tensor,
    quaternion: Tensor,
    plane_point: Tensor,
    plane_normal: Tensor,
    *,
    margin: Scalar = 0.0,
    epsilon: float = 1e-12,
) -> ContactGeometry:
    """Collides oriented box A with one-sided plane B using its support point."""

    for name, value in (
        ("center", center), ("half_size", half_size),
        ("plane_point", plane_point), ("plane_normal", plane_normal),
    ):
        _check_vector(name, value)
    if quaternion.shape[-1] != 4:
        raise ValueError("quaternion must end in dimension 4")
    _, outward = _unit(plane_normal, (0.0, 0.0, 1.0), epsilon)
    local_normal = rotate(
        Tensor.stack(
            quaternion[..., 0],
            -quaternion[..., 1],
            -quaternion[..., 2],
            -quaternion[..., 3],
            dim=-1,
        ),
        outward,
    )
    local_support = (local_normal >= 0.0).where(-half_size, half_size)
    support = dot(local_normal.abs(), half_size)
    height = dot(center - plane_point, outward)
    distance = height - support
    point_a = center + rotate(quaternion, local_support)
    point_b = point_a - outward * distance.unsqueeze(-1)
    return _result(distance, -outward, point_a, point_b, margin)


def box_box(
    center_a: Tensor,
    half_size_a: Tensor,
    quaternion_a: Tensor,
    center_b: Tensor,
    half_size_b: Tensor,
    quaternion_b: Tensor,
    *,
    margin: Scalar = 0.0,
    epsilon: float = 1e-6,
) -> ContactGeometry:
    """Collide two oriented boxes with the 15-axis separating-axis test.

    The least-penetrating axis and one representative support witness are
    returned. ``box_box_manifold`` expands face contact into fixed slots.
    """

    for name, value in (
        ("center_a", center_a),
        ("half_size_a", half_size_a),
        ("center_b", center_b),
        ("half_size_b", half_size_b),
    ):
        _check_vector(name, value)
    if quaternion_a.shape[-1] != 4 or quaternion_b.shape[-1] != 4:
        raise ValueError("box quaternions must end in dimension 4")

    rotation_a = quat_to_matrix(quaternion_a)
    rotation_b = quat_to_matrix(quaternion_b)
    axes_a = tuple(rotation_a[..., :, index] for index in range(3))
    axes_b = tuple(rotation_b[..., :, index] for index in range(3))
    offset = center_b - center_a

    def radius(
        axis: Tensor,
        axes: tuple[Tensor, Tensor, Tensor],
        half_size: Tensor,
    ) -> Tensor:
        terms = (
            dot(axis, box_axis).abs() * half_size[..., index]
            for index, box_axis in enumerate(axes)
        )
        return Tensor.stack(*terms, dim=-1).sum(axis=-1)

    def candidate(axis: Tensor) -> tuple[Tensor, Tensor]:
        projection = dot(offset, axis)
        normal = (projection >= 0.0).unsqueeze(-1).where(axis, -axis)
        distance = (
            projection.abs()
            - radius(axis, axes_a, half_size_a)
            - radius(axis, axes_b, half_size_b)
        )
        return distance, normal

    best_distance, best_normal = candidate(axes_a[0])
    for axis in axes_a[1:] + axes_b:
        distance, normal = candidate(axis)
        replace = distance > best_distance
        best_distance = replace.where(distance, best_distance)
        best_normal = replace.unsqueeze(-1).where(normal, best_normal)
    for axis_a in axes_a:
        for axis_b in axes_b:
            length, axis = _unit(
                cross(axis_a, axis_b),
                (1.0, 0.0, 0.0),
                epsilon,
            )
            distance, normal = candidate(axis)
            replace = (length > epsilon) & (distance > best_distance)
            best_distance = replace.where(distance, best_distance)
            best_normal = replace.unsqueeze(-1).where(normal, best_normal)

    local_a = rotate(quat_conjugate(quaternion_a), best_normal)
    local_b = rotate(quat_conjugate(quaternion_b), best_normal)
    support_a = (local_a >= 0.0).where(half_size_a, -half_size_a)
    support_b = (local_b >= 0.0).where(-half_size_b, half_size_b)
    point_a = center_a + rotate(quaternion_a, support_a)
    point_b = center_b + rotate(quaternion_b, support_b)
    return _result(
        best_distance,
        best_normal,
        point_a,
        point_b,
        margin,
    )
