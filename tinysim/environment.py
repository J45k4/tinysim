"""Minimal batched robotics-environment facade."""

from collections.abc import Callable
from dataclasses import dataclass
import math

from tinygrad import Tensor, dtypes

from .simulation import Simulator
from .state import RuntimeParameters, State, make_runtime_parameters


@dataclass(frozen=True)
class Transition:
    state: State
    observation: Tensor
    reward: Tensor
    terminated: Tensor


@dataclass(frozen=True)
class DomainRandomization:
    """Per-world numeric ranges; topology and Tensor shapes remain fixed."""

    mass_scale: tuple[float, float] = (1.0, 1.0)
    friction_scale: tuple[float, float] = (1.0, 1.0)
    actuator_gain_scale: tuple[float, float] = (1.0, 1.0)

    def __post_init__(self) -> None:
        for name in (
            "mass_scale",
            "friction_scale",
            "actuator_gain_scale",
        ):
            limits = getattr(self, name)
            if (
                len(limits) != 2
                or limits[0] <= 0
                or limits[0] > limits[1]
                or not all(math.isfinite(value) for value in limits)
            ):
                raise ValueError(
                    f"{name} must be a positive finite ordered range"
                )


def _uniform_scale(
    shape: tuple[int, int],
    limits: tuple[float, float],
    template: Tensor,
) -> Tensor:
    lower, upper = limits
    return lower + (upper - lower) * Tensor.rand(
        *shape, dtype=template.dtype, device=template.device
    )


def randomize_parameters(
    simulator: Simulator,
    randomization: DomainRandomization,
    *,
    seed: int,
) -> RuntimeParameters:
    """Creates deterministic per-world parameters without recompiling topology."""

    Tensor.manual_seed(seed)
    base = make_runtime_parameters(simulator.model, simulator.worlds)
    return RuntimeParameters(
        base.body_mass
        * _uniform_scale(
            base.body_mass.shape,
            randomization.mass_scale,
            base.body_mass,
        ),
        base.geom_friction
        * _uniform_scale(
            base.geom_friction.shape,
            randomization.friction_scale,
            base.geom_friction,
        ),
        base.actuator_gain
        * _uniform_scale(
            base.actuator_gain.shape,
            randomization.actuator_gain_scale,
            base.actuator_gain,
        ),
    )


class Environment:
    """Keeps batched reset, stepping, observations, and rewards tensor-only."""

    def __init__(
        self,
        simulator: Simulator,
        *,
        reward: Callable[[State, Tensor], Tensor] | None = None,
        termination: Callable[[State], Tensor] | None = None,
        randomization: DomainRandomization | None = None,
    ):
        self.simulator = simulator
        self.reward_function = reward
        self.termination_function = termination
        self.randomization = randomization

    def reset(
        self,
        state: State | None = None,
        mask: Tensor | None = None,
        *,
        seed: int = 0,
    ) -> tuple[State, Tensor]:
        """Resets all or selected worlds without reading a per-world mask on host."""

        fresh = self.simulator.make_state()
        if self.randomization is not None:
            fresh = State(
                fresh.qpos,
                fresh.qvel,
                fresh.ctrl,
                fresh.time,
                fresh.constraint_impulse,
                randomize_parameters(
                    self.simulator, self.randomization, seed=seed
                ),
            )
        if state is None:
            if mask is not None:
                raise ValueError("a reset mask requires an existing state")
            return fresh, self.simulator.observe(fresh)
        if mask is None:
            mask = Tensor.ones(
                self.simulator.worlds,
                dtype=dtypes.bool,
                device=self.simulator.model.device,
            )
        if mask.shape != (self.simulator.worlds,) or mask.dtype != dtypes.bool:
            raise ValueError("reset mask must be boolean [worlds]")
        if mask.device != self.simulator.model.device:
            raise ValueError("reset mask device must match the simulator")
        select = lambda new, old: mask.reshape(
            (self.simulator.worlds,) + (1,) * (new.ndim - 1)
        ).where(new, old)
        old_parameters = state.parameters or self.simulator.parameters
        new_parameters = fresh.parameters or self.simulator.parameters
        constraint_impulse = (
            None
            if state.constraint_impulse is None
            else select(fresh.constraint_impulse, state.constraint_impulse)
        )
        reset_state = State(
            select(fresh.qpos, state.qpos),
            select(fresh.qvel, state.qvel),
            select(fresh.ctrl, state.ctrl),
            mask.where(fresh.time, state.time),
            constraint_impulse,
            RuntimeParameters(
                select(new_parameters.body_mass, old_parameters.body_mass),
                select(
                    new_parameters.geom_friction,
                    old_parameters.geom_friction,
                ),
                select(
                    new_parameters.actuator_gain,
                    old_parameters.actuator_gain,
                ),
            ),
        )
        return reset_state, self.simulator.observe(reset_state)

    def step(self, state: State, control: Tensor) -> Transition:
        next_state = self.simulator.step(state, control)
        observation = self.simulator.observe(next_state)
        reward = (
            self.reward_function(next_state, control)
            if self.reward_function is not None
            else -next_state.qpos.square().sum(axis=-1)
        )
        terminated = (
            self.termination_function(next_state)
            if self.termination_function is not None
            else (reward != reward)
        )
        if reward.shape != (self.simulator.worlds,):
            raise ValueError("reward function must return [worlds]")
        if terminated.shape != reward.shape:
            raise ValueError("termination function must return [worlds]")
        if reward.dtype != self.simulator.model.dtype:
            raise TypeError("reward dtype must match the compiled model")
        if reward.device != self.simulator.model.device:
            raise ValueError("reward device must match the compiled model")
        if terminated.dtype != dtypes.bool:
            raise TypeError("termination function must return a boolean mask")
        if terminated.device != self.simulator.model.device:
            raise ValueError("termination device must match the compiled model")
        return Transition(next_state, observation, reward, terminated)

    def rollout(
        self,
        state: State,
        policy: Callable[[Tensor], Tensor],
        *,
        steps: int,
        inference: bool = True,
    ) -> State:
        """Runs a policy and simulator on their shared device with no host reads."""

        if steps < 1:
            raise ValueError("steps must be positive")
        current = state
        for _ in range(steps):
            control = policy(self.simulator.observe(current))
            current = (
                self.simulator.inference_step(current, control)
                if inference
                else self.simulator.step(current, control)
            )
        return current
