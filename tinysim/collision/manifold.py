"""Fixed-capacity contact manifolds for primitive shape pairs."""

from tinygrad import Tensor, dtypes

from ..math import cross, dot
from ..spatial import quat_conjugate, quat_to_matrix, rotate
from .primitive import (
    box_box,
    capsule_capsule,
    sphere_capsule,
    sphere_plane,
    sphere_sphere,
)
from .types import (
    ContactGeometry,
    ContactManifold,
    Scalar,
    column,
)


def _unit(
    vector: Tensor,
    fallback: tuple[float, float, float],
    epsilon: float,
) -> tuple[Tensor, Tensor]:
    squared = dot(vector, vector)
    length = squared.sqrt()
    safe = squared.maximum(epsilon * epsilon).sqrt()
    default = Tensor(
        fallback,
        dtype=vector.dtype,
        device=vector.device,
    )
    return length, (squared > epsilon * epsilon).unsqueeze(-1).where(
        vector / safe.unsqueeze(-1),
        default,
    )


def singleton_manifold(geometry: ContactGeometry) -> ContactManifold:
    """Adds one explicit contact-slot dimension to a point contact."""

    return ContactManifold(
        geometry.distance.unsqueeze(-1),
        geometry.normal.unsqueeze(-2),
        geometry.point_a.unsqueeze(-2),
        geometry.point_b.unsqueeze(-2),
        geometry.active.unsqueeze(-1),
    )


def sphere_plane_manifold(
    center: Tensor,
    radius: Scalar,
    plane_point: Tensor,
    plane_normal: Tensor,
    *,
    margin: Scalar = 0.0,
    epsilon: float = 1e-12,
) -> ContactManifold:
    return singleton_manifold(
        sphere_plane(
            center,
            radius,
            plane_point,
            plane_normal,
            margin=margin,
            epsilon=epsilon,
        )
    )


def sphere_sphere_manifold(
    center_a: Tensor,
    radius_a: Scalar,
    center_b: Tensor,
    radius_b: Scalar,
    *,
    margin: Scalar = 0.0,
    epsilon: float = 1e-12,
) -> ContactManifold:
    return singleton_manifold(
        sphere_sphere(
            center_a,
            radius_a,
            center_b,
            radius_b,
            margin=margin,
            epsilon=epsilon,
        )
    )


def sphere_capsule_manifold(
    sphere_center: Tensor,
    sphere_radius: Scalar,
    capsule_start: Tensor,
    capsule_end: Tensor,
    capsule_radius: Scalar,
    *,
    margin: Scalar = 0.0,
    epsilon: float = 1e-12,
) -> ContactManifold:
    return singleton_manifold(
        sphere_capsule(
            sphere_center,
            sphere_radius,
            capsule_start,
            capsule_end,
            capsule_radius,
            margin=margin,
            epsilon=epsilon,
        )
    )


def _padded_singleton(
    geometry: ContactGeometry,
    capacity: int,
) -> ContactManifold:
    if capacity < 1:
        raise ValueError("manifold capacity must be positive")
    if capacity == 1:
        return singleton_manifold(geometry)
    trailing = capacity - 1
    scalar_shape = geometry.distance.shape + (trailing,)
    vector_shape = geometry.distance.shape + (trailing, 3)
    return ContactManifold(
        Tensor.cat(
            geometry.distance.unsqueeze(-1),
            Tensor.ones(
                scalar_shape,
                dtype=geometry.distance.dtype,
                device=geometry.distance.device,
            ),
            dim=-1,
        ),
        Tensor.cat(
            geometry.normal.unsqueeze(-2),
            geometry.normal.unsqueeze(-2).expand(vector_shape),
            dim=-2,
        ),
        Tensor.cat(
            geometry.point_a.unsqueeze(-2),
            geometry.point_a.unsqueeze(-2).expand(vector_shape),
            dim=-2,
        ),
        Tensor.cat(
            geometry.point_b.unsqueeze(-2),
            geometry.point_b.unsqueeze(-2).expand(vector_shape),
            dim=-2,
        ),
        Tensor.cat(
            geometry.active.unsqueeze(-1),
            Tensor.zeros(
                scalar_shape,
                dtype=dtypes.bool,
                device=geometry.distance.device,
            ),
            dim=-1,
        ),
    )


def _slot_scalar(value: Scalar, slots: Tensor) -> Scalar:
    return (
        value.unsqueeze(-1)
        if isinstance(value, Tensor) and value.ndim == slots.ndim - 2
        else value
    )


def capsule_plane_manifold(
    capsule_start: Tensor,
    capsule_end: Tensor,
    capsule_radius: Scalar,
    plane_point: Tensor,
    plane_normal: Tensor,
    *,
    margin: Scalar = 0.0,
    epsilon: float = 1e-12,
) -> ContactManifold:
    """Returns the two spherical-end contacts of a capsule and plane."""

    _, outward = _unit(
        plane_normal,
        (0.0, 0.0, 1.0),
        epsilon,
    )
    axis_points = Tensor.stack(capsule_start, capsule_end, dim=-2)
    normal = outward.unsqueeze(-2).expand(axis_points.shape)
    origin = plane_point.unsqueeze(-2)
    height = dot(axis_points - origin, normal)
    radius = _slot_scalar(capsule_radius, axis_points)
    distance = height - radius
    point_a = axis_points - normal * column(radius, axis_points)
    point_b = axis_points - normal * height.unsqueeze(-1)
    return ContactManifold(
        distance,
        -normal,
        point_a,
        point_b,
        distance <= margin,
    )


def _closest_segment_point(
    start: Tensor,
    end: Tensor,
    point: Tensor,
    epsilon: float,
) -> Tensor:
    segment = end - start
    squared = dot(segment, segment)
    parameter = (
        dot(point - start, segment)
        / squared.maximum(epsilon)
    ).clamp(0.0, 1.0)
    parameter = (squared > epsilon).where(parameter, 0.0)
    return start + segment * parameter.unsqueeze(-1)


def _gather_slots(value: Tensor, indices: Tensor) -> Tensor:
    if value.ndim == indices.ndim:
        return value.gather(-1, indices)
    expanded = indices.unsqueeze(-1).expand(
        indices.shape + (value.shape[-1],)
    )
    return value.gather(-2, expanded)


def _unique_active(
    points: Tensor,
    active: Tensor,
    epsilon: float,
) -> Tensor:
    unique: list[Tensor] = []
    for slot in range(active.shape[-1]):
        keep = active[..., slot]
        for previous, previous_active in enumerate(unique):
            separated = (
                dot(
                    points[..., slot, :] - points[..., previous, :],
                    points[..., slot, :] - points[..., previous, :],
                )
                > epsilon * epsilon
            )
            keep = keep & (~previous_active | separated)
        unique.append(keep)
    return Tensor.stack(*unique, dim=-1)


def capsule_capsule_manifold(
    start_a: Tensor,
    end_a: Tensor,
    radius_a: Scalar,
    start_b: Tensor,
    end_b: Tensor,
    radius_b: Scalar,
    *,
    margin: Scalar = 0.0,
    epsilon: float = 1e-6,
) -> ContactManifold:
    """Returns one contact generally and two for parallel overlapping axes."""

    point = capsule_capsule(
        start_a,
        end_a,
        radius_a,
        start_b,
        end_b,
        radius_b,
        margin=margin,
        epsilon=epsilon,
    )
    fallback = _padded_singleton(point, 2)

    direction_a, direction_b = end_a - start_a, end_b - start_b
    aa, bb = dot(direction_a, direction_a), dot(direction_b, direction_b)
    parallel = (
        dot(cross(direction_a, direction_b), cross(direction_a, direction_b))
        <= epsilon * epsilon * aa * bb
    ) & (aa > epsilon) & (bb > epsilon)

    axis_a = Tensor.stack(
        start_a,
        end_a,
        _closest_segment_point(start_a, end_a, start_b, epsilon),
        _closest_segment_point(start_a, end_a, end_b, epsilon),
        dim=-2,
    )
    axis_b = Tensor.stack(
        _closest_segment_point(start_b, end_b, start_a, epsilon),
        _closest_segment_point(start_b, end_b, end_a, epsilon),
        start_b,
        end_b,
        dim=-2,
    )
    _, normals = _unit(
        axis_b - axis_a,
        (1.0, 0.0, 0.0),
        epsilon,
    )
    axis_distance = dot(axis_b - axis_a, normals)
    radius_a_slots = _slot_scalar(radius_a, axis_a)
    radius_b_slots = _slot_scalar(radius_b, axis_b)
    candidate_distance = axis_distance - radius_a_slots - radius_b_slots
    candidate_a = axis_a + normals * column(radius_a_slots, normals)
    candidate_b = axis_b - normals * column(radius_b_slots, normals)
    _, indices = candidate_distance.topk(
        2,
        dim=-1,
        largest=False,
    )
    distance = _gather_slots(candidate_distance, indices)
    normal = _gather_slots(normals, indices)
    point_a = _gather_slots(candidate_a, indices)
    point_b = _gather_slots(candidate_b, indices)
    active = (distance <= margin) & parallel.unsqueeze(-1)
    active = _unique_active((point_a + point_b) * 0.5, active, epsilon)
    use_parallel = parallel & active.any(axis=-1)
    return ContactManifold(
        use_parallel.unsqueeze(-1).where(
            distance,
            fallback.distance,
        ),
        use_parallel.unsqueeze(-1).unsqueeze(-1).where(
            normal,
            fallback.normal,
        ),
        use_parallel.unsqueeze(-1).unsqueeze(-1).where(
            point_a,
            fallback.point_a,
        ),
        use_parallel.unsqueeze(-1).unsqueeze(-1).where(
            point_b,
            fallback.point_b,
        ),
        use_parallel.unsqueeze(-1).where(
            active,
            fallback.active,
        ),
    )


def _support_face_vertices(
    center: Tensor,
    half_size: Tensor,
    quaternion: Tensor,
    direction: Tensor,
) -> Tensor:
    local_direction = rotate(
        quat_conjugate(quaternion),
        direction,
    )
    hx, hy, hz = (
        half_size[..., component]
        for component in range(3)
    )
    sx = (local_direction[..., 0] >= 0.0).where(hx, -hx)
    sy = (local_direction[..., 1] >= 0.0).where(hy, -hy)
    sz = (local_direction[..., 2] >= 0.0).where(hz, -hz)
    face_x = Tensor.stack(
        Tensor.stack(sx, -hy, -hz, dim=-1),
        Tensor.stack(sx, hy, -hz, dim=-1),
        Tensor.stack(sx, hy, hz, dim=-1),
        Tensor.stack(sx, -hy, hz, dim=-1),
        dim=-2,
    )
    face_y = Tensor.stack(
        Tensor.stack(-hx, sy, -hz, dim=-1),
        Tensor.stack(hx, sy, -hz, dim=-1),
        Tensor.stack(hx, sy, hz, dim=-1),
        Tensor.stack(-hx, sy, hz, dim=-1),
        dim=-2,
    )
    face_z = Tensor.stack(
        Tensor.stack(-hx, -hy, sz, dim=-1),
        Tensor.stack(hx, -hy, sz, dim=-1),
        Tensor.stack(hx, hy, sz, dim=-1),
        Tensor.stack(-hx, hy, sz, dim=-1),
        dim=-2,
    )
    absolute = local_direction.abs()
    use_x = (
        (absolute[..., 0] >= absolute[..., 1])
        & (absolute[..., 0] >= absolute[..., 2])
    )
    use_y = (~use_x) & (
        absolute[..., 1] >= absolute[..., 2]
    )
    local = use_x.unsqueeze(-1).unsqueeze(-1).where(
        face_x,
        use_y.unsqueeze(-1).unsqueeze(-1).where(
            face_y,
            face_z,
        ),
    )
    quaternion_slots = quaternion.unsqueeze(-2).expand(
        local.shape[:-1] + (4,)
    )
    return center.unsqueeze(-2) + rotate(
        quaternion_slots,
        local,
    )


def _closest_face_points(
    center: Tensor,
    half_size: Tensor,
    quaternion: Tensor,
    outward: Tensor,
    points: Tensor,
) -> Tensor:
    quaternion_slots = quaternion.unsqueeze(-2).expand(
        points.shape[:-1] + (4,)
    )
    local = rotate(
        quat_conjugate(quaternion_slots),
        points - center.unsqueeze(-2),
    )
    local_outward = rotate(
        quat_conjugate(quaternion),
        outward,
    )
    hx, hy, hz = (
        half_size[..., component]
        for component in range(3)
    )
    sx = (local_outward[..., 0] >= 0.0).where(hx, -hx)
    sy = (local_outward[..., 1] >= 0.0).where(hy, -hy)
    sz = (local_outward[..., 2] >= 0.0).where(hz, -hz)
    face_x = Tensor.stack(
        sx.unsqueeze(-1).expand(local.shape[:-1]),
        local[..., 1].clamp(-hy.unsqueeze(-1), hy.unsqueeze(-1)),
        local[..., 2].clamp(-hz.unsqueeze(-1), hz.unsqueeze(-1)),
        dim=-1,
    )
    face_y = Tensor.stack(
        local[..., 0].clamp(-hx.unsqueeze(-1), hx.unsqueeze(-1)),
        sy.unsqueeze(-1).expand(local.shape[:-1]),
        local[..., 2].clamp(-hz.unsqueeze(-1), hz.unsqueeze(-1)),
        dim=-1,
    )
    face_z = Tensor.stack(
        local[..., 0].clamp(-hx.unsqueeze(-1), hx.unsqueeze(-1)),
        local[..., 1].clamp(-hy.unsqueeze(-1), hy.unsqueeze(-1)),
        sz.unsqueeze(-1).expand(local.shape[:-1]),
        dim=-1,
    )
    absolute = local_outward.abs()
    use_x = (
        (absolute[..., 0] >= absolute[..., 1])
        & (absolute[..., 0] >= absolute[..., 2])
    )
    use_y = (~use_x) & (
        absolute[..., 1] >= absolute[..., 2]
    )
    projected = use_x.unsqueeze(-1).unsqueeze(-1).where(
        face_x,
        use_y.unsqueeze(-1).unsqueeze(-1).where(
            face_y,
            face_z,
        ),
    )
    return center.unsqueeze(-2) + rotate(
        quaternion_slots,
        projected,
    )


def box_plane_manifold(
    center: Tensor,
    half_size: Tensor,
    quaternion: Tensor,
    plane_point: Tensor,
    plane_normal: Tensor,
    *,
    margin: Scalar = 0.0,
    epsilon: float = 1e-12,
) -> ContactManifold:
    """Returns up to four vertices from the box face nearest the plane."""

    _, outward = _unit(
        plane_normal,
        (0.0, 0.0, 1.0),
        epsilon,
    )
    point_a = _support_face_vertices(
        center,
        half_size,
        quaternion,
        -outward,
    )
    normal = outward.unsqueeze(-2).expand(point_a.shape)
    distance = dot(
        point_a - plane_point.unsqueeze(-2),
        normal,
    )
    point_b = point_a - normal * distance.unsqueeze(-1)
    return ContactManifold(
        distance,
        -normal,
        point_a,
        point_b,
        distance <= margin,
    )


def box_box_manifold(
    center_a: Tensor,
    half_size_a: Tensor,
    quaternion_a: Tensor,
    center_b: Tensor,
    half_size_b: Tensor,
    quaternion_b: Tensor,
    *,
    margin: Scalar = 0.0,
    epsilon: float = 1e-6,
) -> ContactManifold:
    """Returns four face slots or one edge slot from a box-box SAT result."""

    point = box_box(
        center_a,
        half_size_a,
        quaternion_a,
        center_b,
        half_size_b,
        quaternion_b,
        margin=margin,
        epsilon=epsilon,
    )
    fallback = _padded_singleton(point, 4)
    rotation_a = quat_to_matrix(quaternion_a)
    rotation_b = quat_to_matrix(quaternion_b)
    alignment_a = Tensor.stack(
        *(
            dot(point.normal, rotation_a[..., :, axis]).abs()
            for axis in range(3)
        ),
        dim=-1,
    ).max(axis=-1)
    alignment_b = Tensor.stack(
        *(
            dot(point.normal, rotation_b[..., :, axis]).abs()
            for axis in range(3)
        ),
        dim=-1,
    ).max(axis=-1)
    face_axis = alignment_a.maximum(alignment_b) >= 1.0 - epsilon
    reference_a = face_axis & (alignment_a >= alignment_b)
    reference_b = face_axis & ~reference_a

    incident_b = _support_face_vertices(
        center_b,
        half_size_b,
        quaternion_b,
        -point.normal,
    )
    reference_points_a = _closest_face_points(
        center_a,
        half_size_a,
        quaternion_a,
        point.normal,
        incident_b,
    )
    distance_a = dot(
        incident_b - reference_points_a,
        point.normal.unsqueeze(-2),
    )
    active_a = (
        (distance_a <= margin)
        & point.active.unsqueeze(-1)
    )
    active_a = _unique_active(
        (reference_points_a + incident_b) * 0.5,
        active_a,
        epsilon,
    )

    incident_a = _support_face_vertices(
        center_a,
        half_size_a,
        quaternion_a,
        point.normal,
    )
    reference_points_b = _closest_face_points(
        center_b,
        half_size_b,
        quaternion_b,
        -point.normal,
        incident_a,
    )
    distance_b = dot(
        reference_points_b - incident_a,
        point.normal.unsqueeze(-2),
    )
    active_b = (
        (distance_b <= margin)
        & point.active.unsqueeze(-1)
    )
    active_b = _unique_active(
        (incident_a + reference_points_b) * 0.5,
        active_b,
        epsilon,
    )

    use_a = reference_a & active_a.any(axis=-1)
    use_b = reference_b & active_b.any(axis=-1)
    distance = use_a.unsqueeze(-1).where(
        distance_a,
        use_b.unsqueeze(-1).where(
            distance_b,
            fallback.distance,
        ),
    )
    normal = point.normal.unsqueeze(-2).expand(
        distance.shape + (3,)
    )
    point_a = use_a.unsqueeze(-1).unsqueeze(-1).where(
        reference_points_a,
        use_b.unsqueeze(-1).unsqueeze(-1).where(
            incident_a,
            fallback.point_a,
        ),
    )
    point_b = use_a.unsqueeze(-1).unsqueeze(-1).where(
        incident_b,
        use_b.unsqueeze(-1).unsqueeze(-1).where(
            reference_points_b,
            fallback.point_b,
        ),
    )
    active = use_a.unsqueeze(-1).where(
        active_a,
        use_b.unsqueeze(-1).where(
            active_b,
            fallback.active,
        ),
    )
    return ContactManifold(
        distance,
        normal,
        point_a,
        point_b,
        active,
    )
