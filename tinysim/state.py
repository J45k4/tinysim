"""Dynamic simulation state."""

from dataclasses import dataclass

from tinygrad import Tensor

from .compile import CompiledModel


@dataclass(frozen=True)
class RuntimeParameters:
    """Numeric per-world parameters that do not alter compiled topology."""

    body_mass: Tensor
    geom_friction: Tensor
    actuator_gain: Tensor


@dataclass(frozen=True)
class State:
    qpos: Tensor
    qvel: Tensor
    ctrl: Tensor
    time: Tensor
    constraint_impulse: Tensor | None = None
    parameters: RuntimeParameters | None = None


def make_runtime_parameters(
    model: CompiledModel, batch_size: int
) -> RuntimeParameters:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    return RuntimeParameters(
        model.body_mass.unsqueeze(0).expand(batch_size, model.nbody),
        model.geom_friction.unsqueeze(0).expand(batch_size, len(model.geoms)),
        model.actuator_gain.unsqueeze(0).expand(batch_size, model.nu),
    )


def make_state(model: CompiledModel, batch_size: int = 1) -> State:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    zeros = lambda *shape: Tensor.zeros(*shape, dtype=model.dtype, device=model.device)
    qpos = [0.0] * model.nq
    for joint_index, joint in enumerate(model.joints):
        address = model.joint_qpos[joint_index]
        if joint.kind == "ball":
            qpos[address] = 1.0
        elif joint.kind == "free":
            qpos[address + 3] = 1.0
    return State(
        qpos=Tensor([qpos], dtype=model.dtype, device=model.device).expand(batch_size, model.nq),
        qvel=zeros(batch_size, model.nv),
        ctrl=zeros(batch_size, model.nu),
        time=zeros(batch_size),
        constraint_impulse=zeros(batch_size, model.nconstraint),
        parameters=make_runtime_parameters(model, batch_size),
    )


def validate_state(model: CompiledModel, state: State) -> None:
    if state.qpos.ndim != 2 or state.qpos.shape[1] != model.nq:
        raise ValueError(f"qpos must have shape [B, {model.nq}]")
    batch = state.qpos.shape[0]
    if state.qvel.shape != (batch, model.nv):
        raise ValueError(f"qvel must have shape [B, {model.nv}]")
    if state.ctrl.shape != (batch, model.nu):
        raise ValueError(f"ctrl must have shape [B, {model.nu}]")
    if state.time.shape != (batch,):
        raise ValueError("time must have shape [B]")
    tensors = (state.qpos, state.qvel, state.ctrl, state.time)
    if any(tensor.dtype != model.dtype for tensor in tensors):
        raise TypeError(f"state tensors must use model dtype {model.dtype}")
    if any(tensor.device != model.device for tensor in tensors):
        raise ValueError(f"state tensors must be on model device {model.device}")
    if state.constraint_impulse is not None:
        if state.constraint_impulse.shape != (batch, model.nconstraint):
            raise ValueError(
                "constraint_impulse must have shape "
                f"[B, {model.nconstraint}]"
            )
        if state.constraint_impulse.dtype != model.dtype:
            raise TypeError("constraint_impulse dtype must match the model")
        if state.constraint_impulse.device != model.device:
            raise ValueError("constraint_impulse device must match the model")
    if state.parameters is not None:
        expected = (
            (state.parameters.body_mass, (batch, model.nbody), "body_mass"),
            (
                state.parameters.geom_friction,
                (batch, len(model.geoms)),
                "geom_friction",
            ),
            (
                state.parameters.actuator_gain,
                (batch, model.nu),
                "actuator_gain",
            ),
        )
        for tensor, shape, name in expected:
            if tensor.shape != shape:
                raise ValueError(f"runtime {name} must have shape {shape}")
            if tensor.dtype != model.dtype or tensor.device != model.device:
                raise TypeError(
                    f"runtime {name} must match model dtype/device"
                )
