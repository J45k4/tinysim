"""Actuator control and generalized-force mapping."""

from tinygrad import Tensor

from .compile import CompiledModel


def actuator_forces(
    model: CompiledModel,
    qpos: Tensor,
    qvel: Tensor,
    control: Tensor,
    actuator_gain: Tensor | None = None,
) -> Tensor:
    """Maps direct motors and simple servos to reduced-coordinate forces."""
    batch = qpos.shape[0]
    if qpos.shape != (batch, model.nq) or qvel.shape != (batch, model.nv):
        raise ValueError("qpos and qvel dimensions do not match the model")
    if control.shape != (batch, model.nu):
        raise ValueError(f"control must have shape [B, {model.nu}]")
    if any(
        tensor.dtype != model.dtype or tensor.device != model.device
        for tensor in (qpos, qvel, control)
    ):
        raise TypeError("state and control dtype/device must match the compiled model")
    if model.nv == 0:
        return Tensor.zeros(batch, 0, dtype=qpos.dtype, device=qpos.device)
    if actuator_gain is not None:
        if actuator_gain.shape != (batch, model.nu):
            raise ValueError(f"actuator_gain must have shape [B, {model.nu}]")
        if actuator_gain.dtype != model.dtype or actuator_gain.device != model.device:
            raise TypeError("actuator_gain must match model dtype/device")
    zero = qvel[:, 0] * 0.0
    forces = [zero for _ in range(model.nv)]
    for index, actuator in enumerate(model.actuators):
        dof = model.actuator_dof[index]
        lower, upper = model.actuator_control_range[index, 0], model.actuator_control_range[index, 1]
        command = control[:, index].maximum(lower).minimum(upper)
        gain = (
            model.actuator_gain[index]
            if actuator_gain is None
            else actuator_gain[:, index]
        )
        if actuator.kind == "motor":
            force = gain * command
        elif actuator.kind == "position":
            qadr = model.joint_qpos[model.dof_joint[dof]]
            force = (
                gain * (command - qpos[:, qadr])
                - model.actuator_damping[index] * qvel[:, dof]
            )
        else:
            force = gain * (command - qvel[:, dof])
        force_lower, force_upper = model.actuator_force_range[index, 0], model.actuator_force_range[index, 1]
        forces[dof] = forces[dof] + force.maximum(force_lower).minimum(force_upper)
    return Tensor.stack(*forces, dim=-1)
