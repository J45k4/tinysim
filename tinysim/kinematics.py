"""Depth-ordered forward kinematics for fixed and generalized joints."""

from dataclasses import dataclass

from tinygrad import Tensor

from .compile import CompiledModel
from .spatial import axis_angle, quat_mul, quat_normalize, quat_to_matrix, rotate


@dataclass(frozen=True)
class Kinematics:
    """World transforms and motion axes, all with a leading world dimension."""

    body_pos: Tensor
    body_quat: Tensor
    body_rot: Tensor
    joint_pos: Tensor
    joint_axis: Tensor
    dof_axis: Tensor


def _batched(value: Tensor, batch: int) -> Tensor:
    return value.unsqueeze(0).expand((batch,) + value.shape)


def forward_kinematics(model: CompiledModel, qpos: Tensor) -> Kinematics:
    """Computes local-body to world transforms.

    A joint's ``pos`` and ``quat`` define its zero-coordinate frame relative to
    the parent body. Ball/free angular velocities are expressed in that joint
    frame; free translation coordinates and velocities use the same frame.
    """
    if qpos.ndim != 2 or qpos.shape[1] != model.nq:
        raise ValueError(f"qpos must have shape [B, {model.nq}]")
    if qpos.dtype != model.dtype or qpos.device != model.device:
        raise TypeError("qpos dtype and device must match the compiled model")
    batch = qpos.shape[0]
    zeros = Tensor.zeros(batch, 3, dtype=qpos.dtype, device=qpos.device)
    identity = Tensor([[1.0, 0.0, 0.0, 0.0]], dtype=qpos.dtype, device=qpos.device).expand(batch, 4)
    body_pos: list[Tensor | None] = [None] * model.nbody
    body_quat: list[Tensor | None] = [None] * model.nbody
    joint_pos: list[Tensor | None] = [None] * model.nbody
    joint_axis: list[Tensor | None] = [None] * model.nbody
    dof_axis: list[Tensor | None] = [None] * model.nv
    basis = Tensor.eye(3, dtype=qpos.dtype).to(qpos.device)

    # Bodies are compiler-validated to be parent-before-child. The static loop
    # constructs a Tensor graph; it is not a runtime tree traversal.
    for bodies_at_depth in model.depth_bodies:
        for body_index in bodies_at_depth:
            parent = model.body_parent[body_index]
            parent_pos = zeros if parent == -1 else body_pos[parent]
            parent_quat = identity if parent == -1 else body_quat[parent]
            assert parent_pos is not None and parent_quat is not None
            joint_index = model.joint_for_body[body_index]
            local_pos = _batched(model.joint_pos[joint_index], batch)
            local_quat = _batched(model.joint_quat[joint_index], batch)
            origin = parent_pos + rotate(parent_quat, local_pos)
            base_quat = quat_normalize(quat_mul(parent_quat, local_quat))
            axis = rotate(base_quat, _batched(model.joint_axis[joint_index], batch))
            qadr = model.joint_qpos[joint_index]
            dadr = model.joint_dof[joint_index]
            kind = model.joints[joint_index].kind
            if kind == "hinge":
                quaternion = quat_normalize(
                    quat_mul(
                        base_quat,
                        axis_angle(
                            _batched(model.joint_axis[joint_index], batch),
                            qpos[:, qadr],
                        ),
                    )
                )
                position = origin
            elif kind == "slide":
                quaternion = base_quat
                position = origin + axis * qpos[:, qadr].unsqueeze(-1)
            elif kind == "ball":
                relative = quat_normalize(qpos[:, qadr : qadr + 4])
                quaternion = quat_normalize(quat_mul(base_quat, relative))
                position = origin
            elif kind == "free":
                relative = quat_normalize(qpos[:, qadr + 3 : qadr + 7])
                quaternion = quat_normalize(quat_mul(base_quat, relative))
                position = origin + rotate(base_quat, qpos[:, qadr : qadr + 3])
            else:
                quaternion, position = base_quat, origin
            if dadr != -1:
                if kind in ("hinge", "slide"):
                    dof_axis[dadr] = axis
                else:
                    base_axes = [
                        rotate(base_quat, _batched(basis[component], batch))
                        for component in range(3)
                    ]
                    for component in range(model.joint_nv[joint_index]):
                        dof_axis[dadr + component] = base_axes[
                            model.dof_axis_index[dadr + component]
                        ]
            joint_pos[body_index] = position if kind == "free" else origin
            joint_axis[body_index] = axis
            body_pos[body_index] = position
            body_quat[body_index] = quaternion

    stacked_dof_axis = (
        Tensor.stack(*(value for value in dof_axis if value is not None), dim=1)
        if model.nv
        else Tensor.zeros(batch, 0, 3, dtype=qpos.dtype, device=qpos.device)
    )
    return Kinematics(
        Tensor.stack(*(value for value in body_pos if value is not None), dim=1),
        Tensor.stack(*(value for value in body_quat if value is not None), dim=1),
        Tensor.stack(
            *(quat_to_matrix(value) for value in body_quat if value is not None), dim=1
        ),
        Tensor.stack(*(value for value in joint_pos if value is not None), dim=1),
        Tensor.stack(*(value for value in joint_axis if value is not None), dim=1),
        stacked_dof_axis,
    )
