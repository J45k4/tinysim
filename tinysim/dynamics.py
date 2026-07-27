"""Dense contact-free articulated dynamics.

The initial implementation deliberately targets small robots. It builds the
reduced mass matrix from body Jacobians, evaluates bias with recursive
Newton--Euler, and solves the resulting batched SPD system with scalar
Cholesky expressions. All runtime arithmetic remains in tinygrad Tensors.
"""

from tinygrad import Tensor

from .compile import CompiledModel
from .kinematics import Kinematics, forward_kinematics
from .math import cross, dot, skew
from .spatial import rotate


def _body_com_positions(model: CompiledModel, kinematics: Kinematics) -> list[Tensor]:
    batch = kinematics.body_pos.shape[0]
    return [
        kinematics.body_pos[:, body]
        + rotate(
            kinematics.body_quat[:, body],
            model.body_com[body].unsqueeze(0).expand(batch, 3),
        )
        for body in range(model.nbody)
    ]


def mass_matrix_jacobian(
    model: CompiledModel,
    qpos: Tensor,
    *,
    kinematics: Kinematics | None = None,
    body_mass: Tensor | None = None,
) -> Tensor:
    """Returns the symmetric reduced mass matrix with shape ``[B, nv, nv]``."""
    if qpos.ndim != 2 or qpos.shape[1] != model.nq:
        raise ValueError(f"qpos must have shape [B, {model.nq}]")
    if qpos.dtype != model.dtype or qpos.device != model.device:
        raise TypeError("qpos dtype and device must match the compiled model")
    kin = kinematics or forward_kinematics(model, qpos)
    batch = qpos.shape[0]
    if body_mass is not None:
        if body_mass.shape != (batch, model.nbody):
            raise ValueError(f"body_mass must have shape [B, {model.nbody}]")
        if body_mass.dtype != model.dtype or body_mass.device != model.device:
            raise TypeError("body_mass must match model dtype/device")
    if model.nv == 0:
        return Tensor.zeros(batch, 0, 0, dtype=qpos.dtype, device=qpos.device)
    com_positions = _body_com_positions(model, kin)
    zero = kin.body_pos[:, 0, 0] * 0.0
    entries = [[zero for _ in range(model.nv)] for _ in range(model.nv)]

    for body in range(model.nbody):
        linear: list[Tensor] = []
        angular: list[Tensor] = []
        for dof in range(model.nv):
            dof_body = model.dof_body[dof]
            descendant = body
            is_ancestor = False
            while descendant != -1:
                if descendant == dof_body:
                    is_ancestor = True
                    break
                descendant = model.body_parent[descendant]
            if not is_ancestor:
                z = kin.body_pos[:, body] * 0.0
                linear.append(z)
                angular.append(z)
                continue
            axis = kin.dof_axis[:, dof]
            if model.dof_angular[dof]:
                linear.append(cross(axis, com_positions[body] - kin.joint_pos[:, dof_body]))
                angular.append(axis)
            else:
                linear.append(axis)
                angular.append(axis * 0.0)

        mass = (
            model.body_mass[body]
            if body_mass is None
            else body_mass[:, body]
        )
        rotation = kin.body_rot[:, body]
        inertia = model.body_inertia[body]
        for row in range(model.nv):
            for column in range(row, model.nv):
                value = mass * dot(linear[row], linear[column])
                # Rigid-body rotational energy in principal body axes:
                # (R.T*w_i).T diag(I) (R.T*w_j).
                local_row = (rotation.transpose(-1, -2) @ angular[row].unsqueeze(-1)).squeeze(-1)
                local_column = (rotation.transpose(-1, -2) @ angular[column].unsqueeze(-1)).squeeze(-1)
                value = value + (local_row * inertia * local_column).sum(axis=-1)
                entries[row][column] = entries[row][column] + value
                if row != column:
                    entries[column][row] = entries[column][row] + value

    return Tensor.stack(
        *(Tensor.stack(*row, dim=-1) for row in entries),
        dim=-2,
    )


def mass_matrix(
    model: CompiledModel,
    qpos: Tensor,
    *,
    kinematics: Kinematics | None = None,
    body_mass: Tensor | None = None,
) -> Tensor:
    """Composite-rigid-body mass matrix in world spatial coordinates."""

    if qpos.ndim != 2 or qpos.shape[1] != model.nq:
        raise ValueError(f"qpos must have shape [B, {model.nq}]")
    if qpos.dtype != model.dtype or qpos.device != model.device:
        raise TypeError("qpos dtype and device must match the compiled model")
    kin = kinematics or forward_kinematics(model, qpos)
    batch = qpos.shape[0]
    if body_mass is not None:
        if body_mass.shape != (batch, model.nbody):
            raise ValueError(f"body_mass must have shape [B, {model.nbody}]")
        if body_mass.dtype != model.dtype or body_mass.device != model.device:
            raise TypeError("body_mass must match model dtype/device")
    if model.nv == 0:
        return Tensor.zeros(batch, 0, 0, dtype=qpos.dtype, device=qpos.device)

    spatial_inertia: list[Tensor] = []
    zero33 = kin.body_rot[:, 0] * 0.0
    for body in range(model.nbody):
        rotation = kin.body_rot[:, body]
        identity = rotation @ rotation.transpose(-1, -2)
        com = rotate(
            kin.body_quat[:, body],
            model.body_com[body].unsqueeze(0).expand(batch, 3),
        )
        mass = (
            model.body_mass[body]
            if body_mass is None
            else body_mass[:, body]
        )
        mass_matrix_scale = (
            mass.reshape(batch, 1, 1) if mass.ndim else mass
        )
        inertia_com = (
            rotation * model.body_inertia[body]
        ) @ rotation.transpose(-1, -2)
        inertia_origin = inertia_com + mass_matrix_scale * (
            identity * dot(com, com).reshape(batch, 1, 1)
            - com.unsqueeze(-1) * com.unsqueeze(-2)
        )
        coupling = mass_matrix_scale * skew(com)
        spatial_inertia.append(
            Tensor.cat(
                Tensor.cat(inertia_origin, coupling, dim=-1),
                Tensor.cat(-coupling, mass_matrix_scale * identity, dim=-1),
                dim=-2,
            )
        )

    for bodies_at_depth in model.reverse_depth_bodies:
        for body in bodies_at_depth:
            parent = model.body_parent[body]
            if parent == -1:
                continue
            offset = kin.body_pos[:, body] - kin.body_pos[:, parent]
            identity = kin.body_rot[:, body] @ kin.body_rot[:, body].transpose(-1, -2)
            transform = Tensor.cat(
                Tensor.cat(identity, zero33, dim=-1),
                Tensor.cat(-skew(offset), identity, dim=-1),
                dim=-2,
            )
            spatial_inertia[parent] = (
                spatial_inertia[parent]
                + transform.transpose(-1, -2)
                @ spatial_inertia[body]
                @ transform
            )

    motion_subspace: list[Tensor] = []
    zero3 = kin.body_pos[:, 0] * 0.0
    for dof in range(model.nv):
        body = model.dof_body[dof]
        axis = kin.dof_axis[:, dof]
        if model.dof_angular[dof]:
            motion_subspace.append(
                Tensor.cat(
                    axis,
                    cross(
                        axis,
                        kin.body_pos[:, body] - kin.joint_pos[:, body],
                    ),
                    dim=-1,
                )
            )
        else:
            motion_subspace.append(Tensor.cat(zero3, axis, dim=-1))

    zero = kin.body_pos[:, 0, 0] * 0.0
    entries: list[list[Tensor | None]] = [
        [None for _ in range(model.nv)] for _ in range(model.nv)
    ]
    for column in range(model.nv):
        body = model.dof_body[column]
        force = (
            spatial_inertia[body]
            @ motion_subspace[column].unsqueeze(-1)
        ).squeeze(-1)
        current = body
        while current != -1:
            joint = model.joint_for_body[current]
            dadr = model.joint_dof[joint]
            if dadr != -1:
                for component in range(model.joint_nv[joint]):
                    row = dadr + component
                    value = dot(motion_subspace[row], force)
                    entries[row][column] = value
                    entries[column][row] = value
            parent = model.body_parent[current]
            if parent != -1:
                offset = kin.body_pos[:, current] - kin.body_pos[:, parent]
                force = Tensor.cat(
                    force[:, :3] + cross(offset, force[:, 3:]),
                    force[:, 3:],
                    dim=-1,
                )
            current = parent

    return Tensor.stack(
        *(
            Tensor.stack(
                *(value if value is not None else zero for value in row),
                dim=-1,
            )
            for row in entries
        ),
        dim=-2,
    )


def bias_forces(
    model: CompiledModel,
    qpos: Tensor,
    qvel: Tensor,
    *,
    kinematics: Kinematics | None = None,
    body_mass: Tensor | None = None,
) -> Tensor:
    """Returns Coriolis, centrifugal, gravity, and joint-damping forces."""
    if qpos.ndim != 2 or qpos.shape[1] != model.nq:
        raise ValueError(f"qpos must have shape [B, {model.nq}]")
    if qvel.shape != (qpos.shape[0], model.nv):
        raise ValueError(f"qvel must have shape [B, {model.nv}]")
    if any(
        tensor.dtype != model.dtype or tensor.device != model.device
        for tensor in (qpos, qvel)
    ):
        raise TypeError("qpos/qvel dtype and device must match the compiled model")
    kin = kinematics or forward_kinematics(model, qpos)
    batch = qpos.shape[0]
    if body_mass is not None:
        if body_mass.shape != (batch, model.nbody):
            raise ValueError(f"body_mass must have shape [B, {model.nbody}]")
        if body_mass.dtype != model.dtype or body_mass.device != model.device:
            raise TypeError("body_mass must match model dtype/device")
    if model.nv == 0:
        return Tensor.zeros(batch, 0, dtype=qpos.dtype, device=qpos.device)
    zero3 = kin.body_pos[:, 0] * 0.0
    gravity = model.gravity.unsqueeze(0).expand(batch, 3)
    angular_velocity: list[Tensor] = []
    linear_velocity: list[Tensor] = []
    angular_acceleration: list[Tensor] = []
    linear_acceleration: list[Tensor] = []
    body_force: list[Tensor] = []
    body_torque: list[Tensor] = []

    for body, parent in enumerate(model.body_parent):
        parent_w = zero3 if parent == -1 else angular_velocity[parent]
        parent_v = zero3 if parent == -1 else linear_velocity[parent]
        parent_alpha = zero3 if parent == -1 else angular_acceleration[parent]
        parent_acceleration = zero3 if parent == -1 else linear_acceleration[parent]
        parent_pos = zero3 if parent == -1 else kin.body_pos[:, parent]
        offset = kin.body_pos[:, body] - parent_pos
        joint = model.joint_for_body[body]
        dadr = model.joint_dof[joint]
        joint_angular = zero3
        joint_linear = zero3
        if dadr != -1:
            for component in range(model.joint_nv[joint]):
                dof = dadr + component
                velocity = kin.dof_axis[:, dof] * qvel[:, dof].unsqueeze(-1)
                if model.dof_angular[dof]:
                    joint_angular = joint_angular + velocity
                else:
                    joint_linear = joint_linear + velocity
        w = parent_w + joint_angular
        v = parent_v + cross(parent_w, offset) + joint_linear
        alpha = parent_alpha + cross(parent_w, joint_angular)
        acceleration = (
            parent_acceleration
            + cross(parent_alpha, offset)
            + cross(parent_w, cross(parent_w, offset))
            + 2.0 * cross(parent_w, joint_linear)
        )

        com_offset = rotate(
            kin.body_quat[:, body],
            model.body_com[body].unsqueeze(0).expand(batch, 3),
        )
        com_acceleration = acceleration + cross(alpha, com_offset) + cross(w, cross(w, com_offset))
        mass = (
            model.body_mass[body]
            if body_mass is None
            else body_mass[:, body]
        )
        mass_scale = mass.unsqueeze(-1) if mass.ndim == 1 else mass
        force = mass_scale * (com_acceleration - gravity)
        rotation = kin.body_rot[:, body]
        local_w = (rotation.transpose(-1, -2) @ w.unsqueeze(-1)).squeeze(-1)
        world_angular_momentum = (
            rotation @ (model.body_inertia[body] * local_w).unsqueeze(-1)
        ).squeeze(-1)
        local_alpha = (rotation.transpose(-1, -2) @ alpha.unsqueeze(-1)).squeeze(-1)
        inertia_alpha = (
            rotation @ (model.body_inertia[body] * local_alpha).unsqueeze(-1)
        ).squeeze(-1)
        torque = inertia_alpha + cross(w, world_angular_momentum) + cross(com_offset, force)
        angular_velocity.append(w)
        linear_velocity.append(v)
        angular_acceleration.append(alpha)
        linear_acceleration.append(acceleration)
        body_force.append(force)
        body_torque.append(torque)

    generalized: list[Tensor | None] = [None] * model.nv
    for body in reversed(range(model.nbody)):
        joint = model.joint_for_body[body]
        dadr = model.joint_dof[joint]
        if dadr != -1:
            for component in range(model.joint_nv[joint]):
                dof = dadr + component
                axis = kin.dof_axis[:, dof]
                projected = (
                    dot(axis, body_torque[body])
                    if model.dof_angular[dof]
                    else dot(axis, body_force[body])
                )
                generalized[dof] = projected + model.dof_damping[dof] * qvel[:, dof]
        parent = model.body_parent[body]
        if parent != -1:
            body_torque[parent] = (
                body_torque[parent]
                + body_torque[body]
                + cross(kin.body_pos[:, body] - kin.body_pos[:, parent], body_force[body])
            )
            body_force[parent] = body_force[parent] + body_force[body]
    return Tensor.stack(*(value for value in generalized if value is not None), dim=-1)


def batched_solve(matrix: Tensor, rhs: Tensor) -> Tensor:
    """Differentiable Cholesky solve for small batched SPD matrices."""
    if matrix.ndim != 3 or rhs.ndim != 2:
        raise ValueError("expected matrix [B,N,N] and rhs [B,N]")
    if matrix.shape[0] != rhs.shape[0] or matrix.shape[1] != matrix.shape[2] or matrix.shape[2] != rhs.shape[1]:
        raise ValueError("matrix and rhs dimensions do not form a batched square system")
    size = rhs.shape[1]
    if size == 0:
        return rhs
    lower: list[list[Tensor | None]] = [[None] * size for _ in range(size)]
    for row in range(size):
        for column in range(row + 1):
            value = matrix[:, row, column]
            for k in range(column):
                value = value - lower[row][k] * lower[column][k]  # type: ignore[operator]
            lower[row][column] = (
                value.sqrt()
                if row == column
                else value / lower[column][column]  # type: ignore[operator]
            )
    intermediate: list[Tensor] = []
    for row in range(size):
        value = rhs[:, row]
        for column in range(row):
            value = value - lower[row][column] * intermediate[column]  # type: ignore[operator]
        intermediate.append(value / lower[row][row])  # type: ignore[operator]
    solution: list[Tensor | None] = [None] * size
    for row in reversed(range(size)):
        value = intermediate[row]
        for column in range(row + 1, size):
            value = value - lower[column][row] * solution[column]  # type: ignore[operator]
        solution[row] = value / lower[row][row]  # type: ignore[operator]
    return Tensor.stack(*(value for value in solution if value is not None), dim=-1)


def inverse_mass_matrix(matrix: Tensor) -> Tensor:
    """Returns a small batched SPD inverse through the reference solve."""

    if matrix.ndim != 3 or matrix.shape[1] != matrix.shape[2]:
        raise ValueError("mass matrix must have shape [B,N,N]")
    batch, size, _ = matrix.shape
    if size == 0:
        return matrix
    identity = Tensor.eye(size, dtype=matrix.dtype).to(matrix.device)
    columns = [
        batched_solve(
            matrix,
            identity[:, column].unsqueeze(0).expand(batch, size),
        )
        for column in range(size)
    ]
    return Tensor.stack(*columns, dim=-1)


def forward_dynamics(
    model: CompiledModel,
    qpos: Tensor,
    qvel: Tensor,
    generalized_force: Tensor,
    *,
    body_mass: Tensor | None = None,
) -> Tensor:
    """Solves ``M(q) qacc + bias(q, qvel) = generalized_force``."""
    if generalized_force.shape != (qpos.shape[0], model.nv):
        raise ValueError(f"generalized_force must have shape [B, {model.nv}]")
    if any(
        tensor.dtype != model.dtype or tensor.device != model.device
        for tensor in (qpos, qvel, generalized_force)
    ):
        raise TypeError("dynamics tensors must match the compiled model dtype/device")
    kin = forward_kinematics(model, qpos)
    independent_free = all(
        body.parent == -1
        and model.joints[model.joint_for_body[index]].kind == "free"
        and body.com == (0.0, 0.0, 0.0)
        for index, body in enumerate(model.bodies)
    )
    if independent_free:
        bias = bias_forces(
            model,
            qpos,
            qvel,
            kinematics=kin,
            body_mass=body_mass,
        )
        net_force = generalized_force - bias
        accelerations: list[Tensor] = []
        for body in range(model.nbody):
            joint = model.joint_for_body[body]
            address = model.joint_dof[joint]
            mass = (
                model.body_mass[body]
                if body_mass is None
                else body_mass[:, body].unsqueeze(-1)
            )
            linear = net_force[:, address : address + 3] / mass
            world_rotation = kin.body_rot[:, body]
            world_inertia = (
                world_rotation * model.body_inertia[body]
            ) @ world_rotation.transpose(-1, -2)
            axes = kin.dof_axis[
                :, address + 3 : address + 6
            ].transpose(-1, -2)
            generalized_inertia = (
                axes.transpose(-1, -2) @ world_inertia @ axes
            )
            angular = batched_solve(
                generalized_inertia,
                net_force[:, address + 3 : address + 6],
            )
            accelerations.append(Tensor.cat(linear, angular, dim=-1))
        return Tensor.cat(*accelerations, dim=-1)
    matrix = mass_matrix(
        model, qpos, kinematics=kin, body_mass=body_mass
    )
    bias = bias_forces(
        model, qpos, qvel, kinematics=kin, body_mass=body_mass
    )
    return batched_solve(matrix, generalized_force - bias)
