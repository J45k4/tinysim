"""Contact-free semi-implicit Euler integration."""

import math

from tinygrad import Tensor

from .compile import CompiledModel
from .spatial import quat_integrate
from .state import State, validate_state


def integrate(
    position: Tensor,
    velocity: Tensor,
    acceleration: Tensor,
    timestep: float,
) -> tuple[Tensor, Tensor]:
    """Updates velocity first, then position."""
    if timestep <= 0 or not math.isfinite(timestep):
        raise ValueError("timestep must be finite and positive")
    if position.shape != velocity.shape or velocity.shape != acceleration.shape:
        raise ValueError("position, velocity, and acceleration shapes must match")
    next_velocity = velocity + acceleration * timestep
    return position + next_velocity * timestep, next_velocity


def semi_implicit_euler(
    model: CompiledModel,
    state: State,
    qacc: Tensor,
    *,
    control: Tensor | None = None,
    constraint_impulse: Tensor | None = None,
    timestep: float | None = None,
) -> State:
    """Updates generalized velocity, then positions and unit quaternions."""
    validate_state(model, state)
    if qacc.shape != state.qvel.shape:
        raise ValueError(f"qacc must have shape {state.qvel.shape}")
    if qacc.dtype != model.dtype or qacc.device != model.device:
        raise TypeError("qacc dtype and device must match the compiled model")
    dt = model.timestep if timestep is None else timestep
    if dt <= 0 or not math.isfinite(dt):
        raise ValueError("timestep must be finite and positive")
    next_velocity = state.qvel + qacc * dt
    segments: list[Tensor] = []
    for joint, spec in enumerate(model.joints):
        qadr = model.joint_qpos[joint]
        dadr = model.joint_dof[joint]
        if spec.kind in ("hinge", "slide"):
            segments.append(
                state.qpos[:, qadr : qadr + 1]
                + next_velocity[:, dadr : dadr + 1] * dt
            )
        elif spec.kind == "ball":
            segments.append(
                quat_integrate(
                    state.qpos[:, qadr : qadr + 4],
                    next_velocity[:, dadr : dadr + 3],
                    dt,
                )
            )
        elif spec.kind == "free":
            segments.append(
                Tensor.cat(
                    state.qpos[:, qadr : qadr + 3]
                    + next_velocity[:, dadr : dadr + 3] * dt,
                    quat_integrate(
                        state.qpos[:, qadr + 3 : qadr + 7],
                        next_velocity[:, dadr + 3 : dadr + 6],
                        dt,
                    ),
                    dim=-1,
                )
            )
    next_position = (
        Tensor.cat(*segments, dim=-1)
        if segments
        else Tensor.zeros(
            state.qpos.shape[0], 0, dtype=state.qpos.dtype, device=state.qpos.device
        )
    )
    next_control = state.ctrl if control is None else control
    if next_control.shape != state.ctrl.shape:
        raise ValueError(f"control must have shape {state.ctrl.shape}")
    if next_control.dtype != model.dtype or next_control.device != model.device:
        raise TypeError("control dtype and device must match the compiled model")
    return State(
        qpos=next_position,
        qvel=next_velocity,
        ctrl=next_control,
        time=state.time + dt,
        constraint_impulse=(
            state.constraint_impulse
            if constraint_impulse is None
            else constraint_impulse
        ),
        parameters=state.parameters,
    )
