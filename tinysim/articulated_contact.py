"""Fixed-layout articulated collision, contact Jacobians, and contact forces."""

from dataclasses import dataclass

from tinygrad import Tensor, dtypes

from .collision import (
    ContactGeometry,
    ContactParams,
    box_box,
    box_plane,
    capsule_capsule,
    capsule_plane,
    sphere_capsule,
    sphere_plane,
    sphere_sphere,
)
from .compile import CompiledModel
from .contact import smooth_contact
from .kinematics import Kinematics
from .math import cross
from .spatial import quat_mul, rotate


@dataclass(frozen=True)
class ContactBatch:
    geometry: ContactGeometry
    jacobian_a: Tensor
    jacobian_b: Tensor

    @property
    def constraint_jacobian(self) -> Tensor:
        relative = self.jacobian_b - self.jacobian_a
        return (
            relative * self.geometry.normal.unsqueeze(-1)
        ).sum(axis=-2)


def _batched(value: Tensor, batch: int) -> Tensor:
    return value.unsqueeze(0).expand((batch,) + value.shape)


def geometry_transforms(
    model: CompiledModel, kinematics: Kinematics
) -> tuple[Tensor, Tensor]:
    """Returns world positions and local-to-world quaternions for all geoms."""

    batch = kinematics.body_pos.shape[0]
    positions: list[Tensor] = []
    quaternions: list[Tensor] = []
    for index, geom in enumerate(model.geoms):
        local_pos = _batched(model.geom_pos[index], batch)
        local_quat = _batched(model.geom_quat[index], batch)
        if geom.body == -1:
            positions.append(local_pos)
            quaternions.append(local_quat)
        else:
            body_quat = kinematics.body_quat[:, geom.body]
            positions.append(
                kinematics.body_pos[:, geom.body] + rotate(body_quat, local_pos)
            )
            quaternions.append(quat_mul(body_quat, local_quat))
    stacked_positions = Tensor.stack(*positions, dim=1)
    stacked_quaternions = Tensor.stack(*quaternions, dim=1)
    Tensor.realize(stacked_positions, stacked_quaternions)
    return stacked_positions, stacked_quaternions


def point_jacobian(
    model: CompiledModel,
    kinematics: Kinematics,
    body: int,
    point: Tensor,
) -> Tensor:
    """Returns a world-frame Cartesian point Jacobian with shape ``[B,3,nv]``."""

    batch = point.shape[0]
    if body == -1 or model.nv == 0:
        return Tensor.zeros(
            batch, 3, model.nv, dtype=point.dtype, device=point.device
        )
    joint = model.joint_for_body[body]
    if model.body_parent[body] == -1 and model.joints[joint].kind == "free":
        address = model.joint_dof[joint]
        translation = kinematics.dof_axis[
            :, address : address + 3
        ].transpose(-1, -2)
        offset = point - kinematics.joint_pos[:, body]
        rotation = Tensor.stack(
            *(
                cross(
                    kinematics.dof_axis[:, address + component],
                    offset,
                )
                for component in range(3, 6)
            ),
            dim=-1,
        )
        local = Tensor.cat(translation, rotation, dim=-1)
        return local.pad(
            (
                None,
                None,
                (address, model.nv - address - 6),
            )
        )
    columns: list[Tensor] = []
    for dof in range(model.nv):
        ancestor = model.dof_body[dof]
        descendant = body
        while descendant != -1 and descendant != ancestor:
            descendant = model.body_parent[descendant]
        if descendant == -1:
            columns.append(point * 0.0)
            continue
        axis = kinematics.dof_axis[:, dof]
        columns.append(
            cross(
                axis,
                point - kinematics.joint_pos[:, ancestor],
            )
            if model.dof_angular[dof]
            else axis
        )
    return Tensor.stack(*columns, dim=-1)


def _capsule_endpoints(
    model: CompiledModel,
    index: int,
    position: Tensor,
    quaternion: Tensor,
) -> tuple[Tensor, Tensor]:
    axis = rotate(
        quaternion,
        Tensor([0.0, 0.0, 1.0], dtype=position.dtype, device=position.device)
        .unsqueeze(0)
        .expand(position.shape[0], 3),
    )
    offset = axis * model.geom_size[index, 1]
    return position - offset, position + offset


def _flip(geometry: ContactGeometry) -> ContactGeometry:
    return ContactGeometry(
        geometry.distance,
        -geometry.normal,
        geometry.point_b,
        geometry.point_a,
        geometry.active,
    )


def _collide(
    model: CompiledModel,
    a: int,
    b: int,
    positions: Tensor,
    quaternions: Tensor,
) -> ContactGeometry:
    kind_a, kind_b = model.geoms[a].kind, model.geoms[b].kind
    if kind_a == "plane":
        return _flip(_collide(model, b, a, positions, quaternions))
    margin = model.contact.margin
    if kind_b == "plane":
        normal = rotate(
            quaternions[:, b],
            Tensor(
                [0.0, 0.0, 1.0],
                dtype=positions.dtype,
                device=positions.device,
            ).unsqueeze(0).expand(positions.shape[0], 3),
        )
        if kind_a == "sphere":
            return sphere_plane(
                positions[:, a],
                model.geom_size[a, 0],
                positions[:, b],
                normal,
                margin=margin,
            )
        if kind_a == "capsule":
            start, end = _capsule_endpoints(
                model, a, positions[:, a], quaternions[:, a]
            )
            return capsule_plane(
                start,
                end,
                model.geom_size[a, 0],
                positions[:, b],
                normal,
                margin=margin,
            )
        if kind_a == "box":
            return box_plane(
                positions[:, a],
                _batched(model.geom_size[a, :3], positions.shape[0]),
                quaternions[:, a],
                positions[:, b],
                normal,
                margin=margin,
            )
    if kind_a == kind_b == "sphere":
        return sphere_sphere(
            positions[:, a],
            model.geom_size[a, 0],
            positions[:, b],
            model.geom_size[b, 0],
            margin=margin,
        )
    if kind_a == "capsule" and kind_b == "sphere":
        return _flip(_collide(model, b, a, positions, quaternions))
    if kind_a == "sphere" and kind_b == "capsule":
        start, end = _capsule_endpoints(
            model, b, positions[:, b], quaternions[:, b]
        )
        return sphere_capsule(
            positions[:, a],
            model.geom_size[a, 0],
            start,
            end,
            model.geom_size[b, 0],
            margin=margin,
        )
    if kind_a == kind_b == "capsule":
        start_a, end_a = _capsule_endpoints(
            model, a, positions[:, a], quaternions[:, a]
        )
        start_b, end_b = _capsule_endpoints(
            model, b, positions[:, b], quaternions[:, b]
        )
        return capsule_capsule(
            start_a,
            end_a,
            model.geom_size[a, 0],
            start_b,
            end_b,
            model.geom_size[b, 0],
            margin=margin,
        )
    if kind_a == kind_b == "box":
        return box_box(
            positions[:, a],
            _batched(model.geom_size[a, :3], positions.shape[0]),
            quaternions[:, a],
            positions[:, b],
            _batched(model.geom_size[b, :3], positions.shape[0]),
            quaternions[:, b],
            margin=margin,
        )
    raise AssertionError(f"compiler admitted unsupported pair {(kind_a, kind_b)}")


def contacts(model: CompiledModel, kinematics: Kinematics) -> ContactBatch:
    """Evaluates every compiler-selected pair into one fixed contact slot."""

    batch = kinematics.body_pos.shape[0]
    positions, quaternions = geometry_transforms(model, kinematics)
    geometries: list[ContactGeometry] = []
    jacobian_a: list[Tensor] = []
    jacobian_b: list[Tensor] = []
    for a, b in model.collision_pairs:
        geometry = _collide(model, a, b, positions, quaternions)
        geometries.append(geometry)
        jacobian_a.append(
            point_jacobian(
                model, kinematics, model.geoms[a].body, geometry.point_a
            )
        )
        jacobian_b.append(
            point_jacobian(
                model, kinematics, model.geoms[b].body, geometry.point_b
            )
        )
    if not geometries:
        empty_scalar = Tensor.zeros(
            batch, 0, dtype=kinematics.body_pos.dtype, device=kinematics.body_pos.device
        )
        empty_vector = Tensor.zeros(
            batch, 0, 3, dtype=kinematics.body_pos.dtype, device=kinematics.body_pos.device
        )
        empty_jacobian = Tensor.zeros(
            batch,
            0,
            3,
            model.nv,
            dtype=kinematics.body_pos.dtype,
            device=kinematics.body_pos.device,
        )
        return ContactBatch(
            ContactGeometry(
                empty_scalar,
                empty_vector,
                empty_vector,
                empty_vector,
                empty_scalar.cast(dtypes.bool),
            ),
            empty_jacobian,
            empty_jacobian,
        )
    return ContactBatch(
        ContactGeometry(
            Tensor.stack(*(value.distance for value in geometries), dim=1),
            Tensor.stack(*(value.normal for value in geometries), dim=1),
            Tensor.stack(*(value.point_a for value in geometries), dim=1),
            Tensor.stack(*(value.point_b for value in geometries), dim=1),
            Tensor.stack(*(value.active for value in geometries), dim=1),
        ),
        Tensor.stack(*jacobian_a, dim=1),
        Tensor.stack(*jacobian_b, dim=1),
    )


def smooth_generalized_force(
    model: CompiledModel,
    contact_batch: ContactBatch,
    qvel: Tensor,
    geom_friction: Tensor | None = None,
) -> Tensor:
    """Maps differentiable Cartesian contact forces to generalized forces."""

    if not model.collision_pairs:
        return Tensor.zeros(
            qvel.shape[0], model.nv, dtype=qvel.dtype, device=qvel.device
        )
    generalized_velocity = qvel.unsqueeze(1).unsqueeze(-1)
    velocity_a = (contact_batch.jacobian_a @ generalized_velocity).squeeze(-1)
    velocity_b = (contact_batch.jacobian_b @ generalized_velocity).squeeze(-1)
    spec = model.contact
    if geom_friction is not None:
        if geom_friction.shape != (qvel.shape[0], len(model.geoms)):
            raise ValueError(
                f"geom_friction must have shape [B, {len(model.geoms)}]"
            )
        if geom_friction.dtype != model.dtype or geom_friction.device != model.device:
            raise TypeError("geom_friction must match model dtype/device")
        friction = Tensor.stack(
            *(
                (
                    geom_friction[:, a] * geom_friction[:, b]
                ).sqrt()
                for a, b in model.collision_pairs
            ),
            dim=1,
        ) * spec.friction
    else:
        friction = Tensor.stack(
            *(
                (
                    model.geom_friction[a] * model.geom_friction[b]
                ).sqrt()
                for a, b in model.collision_pairs
            ),
            dim=0,
        ).unsqueeze(0).expand(qvel.shape[0], len(model.collision_pairs)) * spec.friction
    forces = smooth_contact(
        contact_batch.geometry,
        velocity_a,
        velocity_b,
        ContactParams(
            stiffness=spec.stiffness,
            damping=spec.damping,
            friction=friction,
            penetration_smoothing=spec.penetration_smoothing,
            force_smoothing=spec.force_smoothing,
            velocity_smoothing=spec.velocity_smoothing,
        ),
    )
    return (
        contact_batch.jacobian_a * forces.force_a.unsqueeze(-1)
        + contact_batch.jacobian_b * forces.force_b.unsqueeze(-1)
    ).sum(axis=-2).sum(axis=1)


def smooth_free_body_generalized_force(
    model: CompiledModel,
    kinematics: Kinematics,
    qvel: Tensor,
    geom_friction: Tensor | None = None,
) -> Tensor:
    """Sparse smooth contact force path for independent root free bodies."""

    if any(
        parent != -1
        or model.joints[model.joint_for_body[body]].kind != "free"
        for body, parent in enumerate(model.body_parent)
    ):
        raise ValueError("sparse free-body contact requires root free bodies")
    if geom_friction is not None:
        if geom_friction.shape != (qvel.shape[0], len(model.geoms)):
            raise ValueError(
                f"geom_friction must have shape [B, {len(model.geoms)}]"
            )
        if geom_friction.dtype != model.dtype or geom_friction.device != model.device:
            raise TypeError("geom_friction must match model dtype/device")

    positions, quaternions = geometry_transforms(model, kinematics)
    zero6 = Tensor.zeros(
        qvel.shape[0],
        6,
        dtype=qvel.dtype,
        device=qvel.device,
    )
    body_forces = [zero6 for _ in range(model.nbody)]

    def point_velocity(body: int, point: Tensor) -> Tensor:
        if body == -1:
            return point * 0.0
        address = model.joint_dof[model.joint_for_body[body]]
        linear = (
            kinematics.dof_axis[:, address : address + 3]
            * qvel[:, address : address + 3].unsqueeze(-1)
        ).sum(axis=1)
        angular = (
            kinematics.dof_axis[:, address + 3 : address + 6]
            * qvel[:, address + 3 : address + 6].unsqueeze(-1)
        ).sum(axis=1)
        return linear + cross(
            angular,
            point - kinematics.joint_pos[:, body],
        )

    def generalized(body: int, point: Tensor, force: Tensor) -> Tensor:
        address = model.joint_dof[model.joint_for_body[body]]
        translation = (
            kinematics.dof_axis[:, address : address + 3]
            * force.unsqueeze(1)
        ).sum(axis=-1)
        torque = cross(
            point - kinematics.joint_pos[:, body],
            force,
        )
        rotation = (
            kinematics.dof_axis[:, address + 3 : address + 6]
            * torque.unsqueeze(1)
        ).sum(axis=-1)
        return Tensor.cat(translation, rotation, dim=-1)

    spec = model.contact
    for a, b in model.collision_pairs:
        geometry = _collide(
            model,
            a,
            b,
            positions,
            quaternions,
        )
        friction = (
            (
                geom_friction[:, a] * geom_friction[:, b]
            ).sqrt()
            if geom_friction is not None
            else (
                model.geom_friction[a] * model.geom_friction[b]
            ).sqrt()
        ) * spec.friction
        forces = smooth_contact(
            geometry,
            point_velocity(model.geoms[a].body, geometry.point_a),
            point_velocity(model.geoms[b].body, geometry.point_b),
            ContactParams(
                stiffness=spec.stiffness,
                damping=spec.damping,
                friction=friction,
                penetration_smoothing=spec.penetration_smoothing,
                force_smoothing=spec.force_smoothing,
                velocity_smoothing=spec.velocity_smoothing,
            ),
        )
        body_a = model.geoms[a].body
        body_b = model.geoms[b].body
        if body_a != -1:
            body_forces[body_a] = body_forces[body_a] + generalized(
                body_a,
                geometry.point_a,
                forces.force_a,
            )
        if body_b != -1:
            body_forces[body_b] = body_forces[body_b] + generalized(
                body_b,
                geometry.point_b,
                forces.force_b,
            )
    return Tensor.cat(*body_forces, dim=-1)
