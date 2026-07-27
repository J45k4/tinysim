"""End-to-end TinySim performance matrix.

This reports wall-clock timings around explicit device synchronization.  It
does not infer unavailable utilization or allocator peak metrics.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, fields, is_dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import resource
import statistics
import sys
import time
from typing import Any

from tinygrad import Device, GlobalCounters, Tensor, TinyJit, dtypes

from tinysim.actuator import actuator_forces
from tinysim.analytical import (
    cartpole_acceleration,
    cartpole_step,
    pendulum_acceleration,
    pendulum_step,
)
from tinysim.articulated_contact import contacts, smooth_generalized_force
from tinysim.compile import compile_model
from tinysim.dynamics import bias_forces, mass_matrix
from tinysim.environment import Environment
from tinysim.integrator import integrate
from tinysim.kinematics import forward_kinematics
from tinysim.simulation import Simulator
from tinysim.robots import quadruped_spec
from tinysim.provenance import project_source_sha256, source_tree_sha256
from tinysim.state import State
from tinysim.systems import sphere_plane_step

from .models import humanoid_scale_spec, quadruped_like_spec


DEFAULT_BATCHES = (1, 16, 64, 256, 1024, 4096, 16384)
QUICK_BATCHES = (1,)
WORKLOAD_NAMES = (
    "pendulum",
    "cartpole",
    "quadruped",
    "humanoid",
    "bouncing_contact",
    "locomotion_policy",
)


def _device() -> str:
    return Device.DEFAULT


def _sync() -> None:
    Device[_device()].synchronize()


def _tensor_values(value: Any) -> list[Tensor]:
    if isinstance(value, Tensor):
        return [value]
    if is_dataclass(value):
        return [
            tensor
            for field in fields(value)
            for tensor in _tensor_values(getattr(value, field.name))
        ]
    if isinstance(value, dict):
        return [tensor for item in value.values() for tensor in _tensor_values(item)]
    if isinstance(value, (tuple, list)):
        return [tensor for item in value for tensor in _tensor_values(item)]
    return []


def _realize(value: Any) -> Any:
    tensors = _tensor_values(value)
    if tensors:
        Tensor.realize(*tensors)
    return value


def _with_grad(tensor: Tensor, enabled: bool) -> Tensor:
    tensor.requires_grad = enabled
    return tensor


def _rss_bytes() -> int:
    # Linux reports KiB; macOS reports bytes.
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def _resident_bytes() -> int:
    return int(GlobalCounters.mem_used_per_device[_device()])


@dataclass(frozen=True)
class CallMetrics:
    wall_s: float
    device_kernel_s: float
    kernels: int
    operations: int
    memory_access_bytes: int
    resident_before_bytes: int
    resident_after_bytes: int
    process_peak_rss_before_bytes: int
    process_peak_rss_after_bytes: int


@dataclass(frozen=True)
class SampleSummary:
    samples: int
    mean_s: float
    median_s: float
    min_s: float
    max_s: float
    p95_s: float
    stdev_s: float
    mean_device_kernel_s: float
    mean_kernels: float
    resident_after_bytes: int
    process_peak_rss_after_bytes: int


def measure_call(call: Callable[[Any], Any], state: Any) -> tuple[Any, CallMetrics]:
    """Measure one synchronized call and return its realized output."""

    _sync()
    resident_before, rss_before = _resident_bytes(), _rss_bytes()
    GlobalCounters.reset()
    started = time.perf_counter()
    output = _realize(call(state))
    _sync()
    elapsed = time.perf_counter() - started
    metrics = CallMetrics(
        wall_s=elapsed,
        device_kernel_s=float(GlobalCounters.time_sum_s),
        kernels=int(GlobalCounters.kernel_count),
        operations=int(GlobalCounters.global_ops),
        memory_access_bytes=int(GlobalCounters.global_mem),
        resident_before_bytes=resident_before,
        resident_after_bytes=_resident_bytes(),
        process_peak_rss_before_bytes=rss_before,
        process_peak_rss_after_bytes=_rss_bytes(),
    )
    return output, metrics


def summarize(samples: Sequence[CallMetrics]) -> SampleSummary:
    if not samples:
        raise ValueError("at least one sample is required")
    wall = sorted(sample.wall_s for sample in samples)
    p95_index = math.ceil(0.95 * len(wall)) - 1
    return SampleSummary(
        samples=len(samples),
        mean_s=statistics.fmean(wall),
        median_s=statistics.median(wall),
        min_s=wall[0],
        max_s=wall[-1],
        p95_s=wall[p95_index],
        stdev_s=statistics.stdev(wall) if len(wall) > 1 else 0.0,
        mean_device_kernel_s=statistics.fmean(
            sample.device_kernel_s for sample in samples
        ),
        mean_kernels=statistics.fmean(sample.kernels for sample in samples),
        resident_after_bytes=samples[-1].resident_after_bytes,
        process_peak_rss_after_bytes=samples[-1].process_peak_rss_after_bytes,
    )


class Workload:
    name: str
    family: str

    def initial_state(self, worlds: int, *, gradients: bool = False) -> Any:
        raise NotImplementedError

    def eager_step(self, state: Any) -> Any:
        raise NotImplementedError

    def make_jitted_step(self, worlds: int) -> Callable[[Any], Any]:
        raise NotImplementedError

    def backward_step(self, worlds: int) -> None:
        raise NotImplementedError

    def settings(self) -> dict[str, Any]:
        raise NotImplementedError

    def subsystem_calls(self, worlds: int) -> dict[str, Callable[[], Any]]:
        return {}

    def python_baseline(self, worlds: int, steps: int) -> float | None:
        return None

    def mujoco_baseline(self, worlds: int, steps: int) -> float | None:
        return None

    def mjx_baseline(
        self, worlds: int, steps: int
    ) -> tuple[float, dict[str, str]] | None:
        return None


@dataclass(frozen=True)
class PendulumState:
    theta: Tensor
    omega: Tensor
    torque: Tensor


class PendulumWorkload(Workload):
    name, family = "pendulum", "analytical"

    def initial_state(self, worlds: int, *, gradients: bool = False) -> PendulumState:
        return PendulumState(
            _with_grad(Tensor.full((worlds, 1), 0.2), gradients).realize(),
            Tensor.zeros(worlds, 1).realize(),
            _with_grad(Tensor.full((worlds, 1), 0.1), gradients).realize(),
        )

    def eager_step(self, state: PendulumState) -> PendulumState:
        theta, omega = pendulum_step(state.theta, state.omega, state.torque)
        return PendulumState(theta, omega, state.torque)

    def make_jitted_step(self, worlds: int) -> Callable[[PendulumState], PendulumState]:
        def run(theta: Tensor, omega: Tensor, torque: Tensor) -> tuple[Tensor, Tensor]:
            output = pendulum_step(theta, omega, torque)
            Tensor.realize(*output)
            return output

        jit = TinyJit(run)

        def step(state: PendulumState) -> PendulumState:
            theta, omega = jit(state.theta, state.omega, state.torque)
            return PendulumState(theta, omega, state.torque)

        step.tinyjit = jit  # type: ignore[attr-defined]
        return step

    def backward_step(self, worlds: int) -> None:
        state = self.initial_state(worlds, gradients=True)
        output = self.eager_step(state)
        loss = output.theta.square().mean() + output.omega.square().mean()
        loss.backward()
        assert state.torque.grad is not None
        state.torque.grad.realize()

    def settings(self) -> dict[str, Any]:
        return {
            "mass": 1.0,
            "length": 1.0,
            "damping": 0.05,
            "gravity": 9.81,
            "timestep": 0.01,
            "solver_iterations": 0,
            "contact_capacity": 0,
        }

    def subsystem_calls(self, worlds: int) -> dict[str, Callable[[], Any]]:
        state = self.initial_state(worlds)
        acceleration = pendulum_acceleration(
            state.theta, state.omega, state.torque
        )
        return {
            "dynamics": lambda: pendulum_acceleration(
                state.theta, state.omega, state.torque
            ),
            "integration": lambda: integrate(
                state.theta, state.omega, acceleration, 0.01
            ),
        }

    def python_baseline(self, worlds: int, steps: int) -> float:
        theta, omega = [0.2] * worlds, [0.0] * worlds
        started = time.perf_counter()
        for _ in range(steps):
            for world in range(worlds):
                acceleration = (
                    0.1 - 0.05 * omega[world] - 9.81 * math.sin(theta[world])
                )
                omega[world] += 0.01 * acceleration
                theta[world] += 0.01 * omega[world]
        return time.perf_counter() - started


@dataclass(frozen=True)
class CartpoleState:
    position: Tensor
    velocity: Tensor
    theta: Tensor
    omega: Tensor
    force: Tensor


class CartpoleWorkload(Workload):
    name, family = "cartpole", "analytical"

    def initial_state(self, worlds: int, *, gradients: bool = False) -> CartpoleState:
        return CartpoleState(
            Tensor.zeros(worlds, 1).realize(),
            Tensor.zeros(worlds, 1).realize(),
            _with_grad(Tensor.full((worlds, 1), 0.1), gradients).realize(),
            Tensor.zeros(worlds, 1).realize(),
            _with_grad(Tensor.full((worlds, 1), 0.1), gradients).realize(),
        )

    def eager_step(self, state: CartpoleState) -> CartpoleState:
        values = cartpole_step(
            state.position,
            state.velocity,
            state.theta,
            state.omega,
            state.force,
        )
        return CartpoleState(*values, state.force)

    def make_jitted_step(self, worlds: int) -> Callable[[CartpoleState], CartpoleState]:
        def run(
            position: Tensor,
            velocity: Tensor,
            theta: Tensor,
            omega: Tensor,
            force: Tensor,
        ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
            output = cartpole_step(position, velocity, theta, omega, force)
            Tensor.realize(*output)
            return output

        jit = TinyJit(run)

        def step(state: CartpoleState) -> CartpoleState:
            output = jit(
                state.position,
                state.velocity,
                state.theta,
                state.omega,
                state.force,
            )
            return CartpoleState(*output, state.force)

        step.tinyjit = jit  # type: ignore[attr-defined]
        return step

    def backward_step(self, worlds: int) -> None:
        state = self.initial_state(worlds, gradients=True)
        output = self.eager_step(state)
        loss = output.position.square().mean() + output.theta.square().mean()
        loss.backward()
        assert state.force.grad is not None
        state.force.grad.realize()

    def settings(self) -> dict[str, Any]:
        return {
            "cart_mass": 1.0,
            "pole_mass": 0.1,
            "length": 0.5,
            "gravity": 9.81,
            "timestep": 0.01,
            "solver_iterations": 0,
            "contact_capacity": 0,
        }

    def subsystem_calls(self, worlds: int) -> dict[str, Callable[[], Any]]:
        state = self.initial_state(worlds)
        acceleration = cartpole_acceleration(
            state.velocity, state.theta, state.omega, state.force
        )
        return {
            "dynamics": lambda: cartpole_acceleration(
                state.velocity, state.theta, state.omega, state.force
            ),
            "integration": lambda: (
                integrate(state.position, state.velocity, acceleration[0], 0.01),
                integrate(state.theta, state.omega, acceleration[1], 0.01),
            ),
        }

    def python_baseline(self, worlds: int, steps: int) -> float:
        position = [0.0] * worlds
        velocity = [0.0] * worlds
        theta = [0.1] * worlds
        omega = [0.0] * worlds
        started = time.perf_counter()
        for _ in range(steps):
            for world in range(worlds):
                sine, cosine = math.sin(theta[world]), math.cos(theta[world])
                coupling = 0.05 * cosine
                rhs_cart = 0.1 + 0.05 * sine * omega[world] ** 2
                rhs_pole = 0.4905 * sine
                determinant = 1.1 * 0.025 - coupling * coupling
                cart_acc = (0.025 * rhs_cart - coupling * rhs_pole) / determinant
                pole_acc = (1.1 * rhs_pole - coupling * rhs_cart) / determinant
                velocity[world] += 0.01 * cart_acc
                position[world] += 0.01 * velocity[world]
                omega[world] += 0.01 * pole_acc
                theta[world] += 0.01 * omega[world]
        return time.perf_counter() - started


@dataclass(frozen=True)
class BouncingState:
    position: Tensor
    velocity: Tensor


class BouncingContactWorkload(Workload):
    name, family = "bouncing_contact", "smooth_contact"

    def initial_state(self, worlds: int, *, gradients: bool = False) -> BouncingState:
        position = Tensor(
            [[0.0, 0.0, 0.2]], dtype=dtypes.float32
        ).expand(worlds, 3).contiguous()
        position = _with_grad(position, gradients).realize()
        velocity = Tensor(
            [[0.1, 0.0, -1.0]], dtype=dtypes.float32
        ).expand(worlds, 3).contiguous().realize()
        return BouncingState(position, velocity)

    def eager_step(self, state: BouncingState) -> BouncingState:
        return BouncingState(*sphere_plane_step(state.position, state.velocity))

    def make_jitted_step(
        self, worlds: int
    ) -> Callable[[BouncingState], BouncingState]:
        def run(position: Tensor, velocity: Tensor) -> tuple[Tensor, Tensor]:
            output = sphere_plane_step(position, velocity)
            Tensor.realize(*output)
            return output

        jit = TinyJit(run)

        def step(state: BouncingState) -> BouncingState:
            return BouncingState(*jit(state.position, state.velocity))

        step.tinyjit = jit  # type: ignore[attr-defined]
        return step

    def backward_step(self, worlds: int) -> None:
        state = self.initial_state(worlds, gradients=True)
        output = self.eager_step(state)
        output.position.square().mean().backward()
        assert state.position.grad is not None
        state.position.grad.realize()

    def settings(self) -> dict[str, Any]:
        return {
            "mass": 1.0,
            "radius": 0.25,
            "gravity": [0.0, 0.0, -9.81],
            "timestep": 0.002,
            "contact": "smooth_sphere_plane",
            "stiffness": 5000.0,
            "damping": 100.0,
            "solver_iterations": 0,
            "contact_capacity": 1,
        }

    def subsystem_calls(self, worlds: int) -> dict[str, Callable[[], Any]]:
        state = self.initial_state(worlds)

        def collision_contact() -> Tensor:
            # Full step is intentionally retained as the stable public boundary;
            # the isolated contact timing excludes the final integration kernels.
            result = sphere_plane_step(state.position, state.velocity)
            return (result[1] - state.velocity) / 0.002

        acceleration = collision_contact()
        return {
            "collision_contact_force": collision_contact,
            "integration": lambda: integrate(
                state.position, state.velocity, acceleration, 0.002
            ),
        }

    def python_baseline(self, worlds: int, steps: int) -> float:
        position = [[0.0, 0.0, 0.2] for _ in range(worlds)]
        velocity = [[0.1, 0.0, -1.0] for _ in range(worlds)]

        def smooth_positive(value: float, epsilon: float) -> float:
            return 0.5 * (value + math.sqrt(value * value + epsilon * epsilon))

        started = time.perf_counter()
        for _ in range(steps):
            for world in range(worlds):
                penetration = smooth_positive(
                    0.25 - position[world][2], 1e-4
                )
                normal_force = smooth_positive(
                    5000.0 * penetration - 100.0 * velocity[world][2],
                    1e-6,
                )
                tangent_speed = math.sqrt(
                    velocity[world][0] ** 2
                    + velocity[world][1] ** 2
                    + 1e-6
                )
                velocity[world][0] += (
                    -0.002
                    * 0.5
                    * normal_force
                    * velocity[world][0]
                    / tangent_speed
                )
                velocity[world][1] += (
                    -0.002
                    * 0.5
                    * normal_force
                    * velocity[world][1]
                    / tangent_speed
                )
                velocity[world][2] += 0.002 * (-9.81 + normal_force)
                for axis in range(3):
                    position[world][axis] += 0.002 * velocity[world][axis]
        return time.perf_counter() - started


class ArticulatedWorkload(Workload):
    family = "articulated"

    def __init__(self, name: str):
        self.name = name
        spec = quadruped_like_spec() if name == "quadruped" else humanoid_scale_spec()
        # Static model tensors are constants for the captured step.  Realizing
        # them before capture prevents their construction graphs from becoming
        # accidental JIT inputs or compile-time work.
        self.model = _realize(compile_model(spec))

    def initial_state(self, worlds: int, *, gradients: bool = False) -> State:
        simulator = Simulator(self.model, worlds)
        state = simulator.make_state()
        if gradients:
            return State(
                _with_grad(state.qpos.contiguous(), True).realize(),
                state.qvel,
                _with_grad(state.ctrl.contiguous(), True).realize(),
                state.time,
                state.constraint_impulse,
                state.parameters,
            )
        return state

    def eager_step(self, state: State) -> State:
        return Simulator(self.model, state.qpos.shape[0]).step(state)

    def make_jitted_step(self, worlds: int) -> Callable[[State], State]:
        simulator = Simulator(self.model, worlds)

        def step(state: State) -> State:
            return simulator.inference_step(state)

        step.simulator = simulator  # type: ignore[attr-defined]
        return step

    def backward_step(self, worlds: int) -> None:
        state = self.initial_state(worlds, gradients=True)
        output = self.eager_step(state)
        (output.qpos.square().mean() + output.qvel.square().mean()).backward()
        assert state.ctrl.grad is not None
        state.ctrl.grad.realize()

    def settings(self) -> dict[str, Any]:
        return {
            "model": self.model.name,
            "bodies": self.model.nbody,
            "qpos": self.model.nq,
            "dofs": self.model.nv,
            "actuators": self.model.nu,
            "timestep": self.model.timestep,
            "contact_mode": self.model.contact.mode,
            "solver_iterations": (
                self.model.contact.solver_iterations
                if self.model.contact.mode == "constraint"
                else 0
            ),
            "contact_capacity": len(self.model.collision_pairs),
        }

    def subsystem_calls(self, worlds: int) -> dict[str, Callable[[], Any]]:
        state = self.initial_state(worlds)
        kinematics = forward_kinematics(self.model, state.qpos)
        calls: dict[str, Callable[[], Any]] = {
            "kinematics": lambda: forward_kinematics(self.model, state.qpos),
            "mass_matrix": lambda: mass_matrix(self.model, state.qpos),
            "bias_forces": lambda: bias_forces(
                self.model, state.qpos, state.qvel
            ),
            "actuation": lambda: actuator_forces(
                self.model, state.qpos, state.qvel, state.ctrl
            ),
        }
        if self.model.collision_pairs:
            contact_batch = contacts(self.model, kinematics)
            calls["contact"] = lambda: smooth_generalized_force(
                self.model, contact_batch, state.qvel
            )
        return calls

    def mujoco_baseline(self, worlds: int, steps: int) -> float | None:
        try:
            import mujoco
            from tinysim.reference.mujoco import model_to_mjcf
        except (ImportError, OSError):
            return None
        model = mujoco.MjModel.from_xml_string(model_to_mjcf(self.model))
        data = [mujoco.MjData(model) for _ in range(worlds)]
        started = time.perf_counter()
        for _ in range(steps):
            for world in data:
                mujoco.mj_step(model, world)
        return time.perf_counter() - started

    def mjx_baseline(
        self, worlds: int, steps: int
    ) -> tuple[float, dict[str, str]] | None:
        try:
            import jax
            import mujoco
            from mujoco import mjx
            from tinysim.reference.mujoco import model_to_mjcf
        except (ImportError, OSError):
            return None
        model = mujoco.MjModel.from_xml_string(model_to_mjcf(self.model))
        mjx_model = mjx.put_model(model)
        make_batched_data = jax.vmap(
            mjx.make_data, in_axes=None, out_axes=0, axis_size=worlds
        )
        data = make_batched_data(mjx_model)
        step = jax.jit(jax.vmap(mjx.step, in_axes=(None, 0)))
        data = step(mjx_model, data)
        data.qpos.block_until_ready()
        started = time.perf_counter()
        for _ in range(steps):
            data = step(mjx_model, data)
        data.qpos.block_until_ready()
        device = jax.devices()[0]
        return time.perf_counter() - started, {
            "device": str(device),
            "platform": device.platform,
            "dtype": str(data.qpos.dtype),
            "jax_version": jax.__version__,
            "mujoco_version": mujoco.__version__,
        }


class LocomotionPolicyWorkload(Workload):
    """Representative Environment + policy + free-base contact step."""

    name, family = "locomotion_policy", "robotics_environment"

    def __init__(self) -> None:
        self.model = _realize(compile_model(quadruped_spec()))

    def initial_state(self, worlds: int, *, gradients: bool = False) -> State:
        state = Simulator(self.model, worlds).make_state()
        qpos = state.qpos.contiguous()
        qpos[:, 2].assign(0.55)
        if gradients:
            qpos = _with_grad(qpos, True)
        return State(
            qpos.realize(),
            state.qvel,
            state.ctrl,
            state.time,
            state.constraint_impulse,
            state.parameters,
        )

    @staticmethod
    def _policy(observation: Tensor, worlds: int) -> Tensor:
        pattern = Tensor(
            [[1.0, -1.0, -1.0, 1.0]],
            dtype=observation.dtype,
            device=observation.device,
        ).expand(worlds, 4)
        return observation[:, :1] * 0.0 + pattern * 0.25

    def eager_step(self, state: State) -> State:
        simulator = Simulator(self.model, state.qpos.shape[0])
        environment = Environment(simulator)
        observation = simulator.observe(state)
        return environment.step(
            state,
            self._policy(observation, state.qpos.shape[0]),
        ).state

    def make_jitted_step(self, worlds: int) -> Callable[[State], State]:
        simulator = Simulator(self.model, worlds)
        environment = Environment(simulator)

        def step(state: State) -> State:
            return environment.rollout(
                state,
                lambda observation: self._policy(observation, worlds),
                steps=1,
                inference=True,
            )

        step.simulator = simulator  # type: ignore[attr-defined]
        return step

    def backward_step(self, worlds: int) -> None:
        state = self.initial_state(worlds, gradients=True)
        output = self.eager_step(state)
        output.qpos.square().mean().backward()
        assert state.qpos.grad is not None
        state.qpos.grad.realize()

    def settings(self) -> dict[str, Any]:
        return {
            "model": self.model.name,
            "bodies": self.model.nbody,
            "qpos": self.model.nq,
            "dofs": self.model.nv,
            "actuators": self.model.nu,
            "timestep": self.model.timestep,
            "contact_mode": self.model.contact.mode,
            "solver_iterations": 0,
            "contact_capacity": len(self.model.collision_pairs),
            "policy": "constant_diagonal_gait",
            "environment_api": "Environment.step/rollout",
        }

    def subsystem_calls(self, worlds: int) -> dict[str, Callable[[], Any]]:
        state = self.initial_state(worlds)
        simulator = Simulator(self.model, worlds)
        return {
            "observation": lambda: simulator.observe(state),
            "policy": lambda: self._policy(
                simulator.observe(state), worlds
            ),
        }


def make_workload(name: str) -> Workload:
    if name == "pendulum":
        return PendulumWorkload()
    if name == "cartpole":
        return CartpoleWorkload()
    if name == "bouncing_contact":
        return BouncingContactWorkload()
    if name in ("quadruped", "humanoid"):
        return ArticulatedWorkload(name)
    if name == "locomotion_policy":
        return LocomotionPolicyWorkload()
    raise ValueError(f"unknown workload {name!r}")


def _captured_calls(step: Callable[[Any], Any]) -> int | None:
    jit = getattr(step, "tinyjit", None)
    if jit is None:
        simulator = getattr(step, "simulator", None)
        jit = getattr(simulator, "_inference_step", None)
    captured = getattr(jit, "captured", None)
    if captured is None:
        return None
    return len(captured.linear.src)


def _backward_measurement(
    workload: Workload, worlds: int, samples: int
) -> dict[str, Any]:
    measured: list[CallMetrics] = []
    for _ in range(samples):
        _, sample = measure_call(lambda _: workload.backward_step(worlds), None)
        measured.append(sample)
    summary = asdict(summarize(measured))
    summary.update(
        {
            "world_steps_per_s": worlds / summary["mean_s"],
            "per_world_ns": summary["mean_s"] * 1e9 / worlds,
            "scope": "graph_build_forward_backward_and_gradient_realize",
        }
    )
    return summary


def _subsystem_measurements(
    workload: Workload, worlds: int, samples: int
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, call in workload.subsystem_calls(worlds).items():
        measured: list[CallMetrics] = []
        for _ in range(samples):
            _, sample = measure_call(lambda _: call(), None)
            measured.append(sample)
        result[name] = asdict(summarize(measured))
    return result


def _baseline_measurement(
    workload: Workload, worlds: int, steps: int
) -> dict[str, Any]:
    elapsed = workload.python_baseline(worlds, steps)
    if elapsed is None:
        return {
            "status": "unavailable",
            "reason": "no honest like-for-like pure-Python articulated baseline",
        }
    return {
        "status": "available",
        "implementation": "standard_library_scalar_loop",
        "steps": steps,
        "wall_s": elapsed,
        "world_steps_per_s": worlds * steps / elapsed,
    }


def _mujoco_baseline_measurement(
    workload: Workload, worlds: int, steps: int
) -> dict[str, Any]:
    elapsed = workload.mujoco_baseline(worlds, steps)
    if elapsed is None:
        return {
            "status": "unavailable",
            "reason": (
                "canonical MuJoCo bindings unavailable or workload has no "
                "like-for-like adapter"
            ),
        }
    return {
        "status": "available",
        "implementation": "canonical_mujoco_mj_step_scalar_world_loop",
        "steps": steps,
        "wall_s": elapsed,
        "world_steps_per_s": worlds * steps / elapsed,
    }


def _mjx_baseline_measurement(
    workload: Workload, worlds: int, steps: int
) -> dict[str, Any]:
    measurement = workload.mjx_baseline(worlds, steps)
    if measurement is None:
        return {
            "status": "unavailable",
            "reason": (
                "JAX/MJX unavailable or workload has no like-for-like adapter"
            ),
        }
    elapsed, environment = measurement
    return {
        "status": "available",
        "implementation": "canonical_mjx_jit_vmap",
        "steps": steps,
        "wall_s": elapsed,
        "world_steps_per_s": worlds * steps / elapsed,
        "environment": environment,
    }


def benchmark_case(
    workload: Workload,
    worlds: int,
    *,
    warm_steps: int,
    backward_steps: int,
    subsystem_steps: int,
    baseline_steps: int,
) -> dict[str, Any]:
    if min(worlds, warm_steps) <= 0:
        raise ValueError("worlds and warm_steps must be positive")
    state = _realize(workload.initial_state(worlds))
    jitted = workload.make_jitted_step(worlds)
    state, first = measure_call(jitted, state)
    state, capture = measure_call(jitted, state)
    warm_samples: list[CallMetrics] = []
    for _ in range(warm_steps):
        state, sample = measure_call(jitted, state)
        warm_samples.append(sample)
    warm = asdict(summarize(warm_samples))
    warm.update(
        {
            "world_steps_per_s": worlds / warm["mean_s"],
            "per_world_ns": warm["mean_s"] * 1e9 / worlds,
        }
    )
    result: dict[str, Any] = {
        "workload": workload.name,
        "family": workload.family,
        "worlds": worlds,
        "settings": workload.settings(),
        "first": asdict(first),
        "capture": asdict(capture),
        "warm": warm,
        "captured_calls": _captured_calls(jitted),
        "backward": (
            _backward_measurement(workload, worlds, backward_steps)
            if backward_steps
            else {"status": "not_requested"}
        ),
        "subsystems": (
            _subsystem_measurements(workload, worlds, subsystem_steps)
            if subsystem_steps
            else {"status": "not_requested"}
        ),
        "baselines": {
            "python": (
                _baseline_measurement(workload, worlds, baseline_steps)
                if baseline_steps
                else {"status": "not_requested"}
            ),
            "mujoco": (
                _mujoco_baseline_measurement(
                    workload, worlds, baseline_steps
                )
                if baseline_steps
                else {"status": "not_requested"}
            ),
            "mjx_jax": (
                _mjx_baseline_measurement(
                    workload, worlds, baseline_steps
                )
                if baseline_steps
                else {"status": "not_requested"}
            ),
            "specialized_uop": {
                "status": "unavailable",
                "reason": "TinySim has no specialized UOp step kernel",
            },
        },
    }
    return result


def _environment() -> dict[str, Any]:
    project = Path(__file__).resolve().parents[1]
    source_revision = source_tree_sha256(project / "tinysim")
    benchmark_revision = source_tree_sha256(project / "benchmarks")
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "device": _device(),
        "dtype": "float32",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or "unreported",
        "tinygrad_revision": _git_revision(
            Path(__file__).resolve().parents[2] / "tinygrad"
        ),
        "tinysim_revision": f"source-sha256:{source_revision}",
        "tinysim_source_sha256": source_revision,
        "benchmark_source_sha256": benchmark_revision,
        "project_source_sha256": project_source_sha256(project),
        "synchronization": "Device[Device.DEFAULT].synchronize before and after each sample",
        "timer": "time.perf_counter",
        "memory_method": {
            "resident": "tinygrad GlobalCounters.mem_used_per_device snapshot",
            "process_peak": "resource.getrusage process high-water RSS",
            "per_call_allocator_peak": "unavailable: tinygrad counter exposes current, not peak, allocation",
        },
        "device_utilization": {
            "status": "unavailable",
            "reason": "no portable utilization counter in tinygrad benchmark API",
        },
    }


def _git_revision(path: Path) -> str:
    head = path.resolve() / ".git" / "HEAD"
    try:
        value = head.read_text(encoding="utf-8").strip()
        if value.startswith("ref: "):
            value = (
                path.resolve() / ".git" / value.removeprefix("ref: ")
            ).read_text(encoding="utf-8").strip()
        return value
    except OSError:
        return "unavailable"


def add_scaling_efficiency(cases: list[dict[str, Any]]) -> None:
    by_workload: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        by_workload.setdefault(case["workload"], []).append(case)
    for workload_cases in by_workload.values():
        baseline = next(
            (case for case in workload_cases if case["worlds"] == 1), None
        )
        for case in workload_cases:
            case["warm"]["scaling_efficiency_vs_batch1"] = (
                case["warm"]["world_steps_per_s"]
                / (
                    case["worlds"]
                    * baseline["warm"]["world_steps_per_s"]
                )
                if baseline is not None
                else None
            )


def run_matrix(
    workloads: Sequence[str],
    worlds: Sequence[int],
    *,
    warm_steps: int,
    backward_steps: int = 0,
    subsystem_steps: int = 0,
    baseline_steps: int = 0,
) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for workload_name in workloads:
        workload = make_workload(workload_name)
        for batch in worlds:
            try:
                cases.append(
                    benchmark_case(
                        workload,
                        batch,
                        warm_steps=warm_steps,
                        backward_steps=backward_steps,
                        subsystem_steps=subsystem_steps,
                        baseline_steps=baseline_steps,
                    )
                )
            except (MemoryError, RuntimeError) as error:
                failures.append(
                    {
                        "workload": workload_name,
                        "worlds": batch,
                        "error": type(error).__name__,
                        "message": str(error),
                    }
                )
                break
    add_scaling_efficiency(cases)
    return {
        "schema_version": 1,
        "environment": _environment(),
        "protocol": {
            "batches_requested": list(worlds),
            "workloads_requested": list(workloads),
            "warm_samples": warm_steps,
            "backward_samples": backward_steps,
            "subsystem_samples": subsystem_steps,
            "baseline_steps": baseline_steps,
            "jit_stages": [
                "first uncaptured call",
                "capture/compile call",
                "warm replay samples",
            ],
            "failure_policy": "record allocation/runtime failure and stop larger batches for that workload",
        },
        "cases": cases,
        "failures": failures,
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# TinySim benchmark report",
        "",
        f"Device: `{report['environment']['device']}`; dtype: `float32`.",
        "",
        "| Workload | Worlds | First (ms) | Capture (ms) | Warm (ms) | World steps/s | Kernels/step | Backward world steps/s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case in report["cases"]:
        backward = case["backward"]
        backward_rate = (
            f"{backward['world_steps_per_s']:.3f}"
            if "world_steps_per_s" in backward
            else "n/a"
        )
        lines.append(
            "| {workload} | {worlds} | {first:.3f} | {capture:.3f} | "
            "{warm:.3f} | {rate:.3f} | {kernels:.1f} | {backward} |".format(
                workload=case["workload"],
                worlds=case["worlds"],
                first=case["first"]["wall_s"] * 1e3,
                capture=case["capture"]["wall_s"] * 1e3,
                warm=case["warm"]["mean_s"] * 1e3,
                rate=case["warm"]["world_steps_per_s"],
                kernels=case["warm"]["mean_kernels"],
                backward=backward_rate,
            )
        )
    lines.extend(
        (
            "",
            "Memory columns in JSON are allocator-resident snapshots and process "
            "high-water RSS, not per-call allocator peaks. Device utilization is "
            "reported unavailable when the backend exposes no portable counter.",
            "",
        )
    )
    return "\n".join(lines)


def write_report(report: dict[str, Any], output: Path) -> tuple[Path, Path]:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    markdown = output.with_suffix(".md")
    markdown.write_text(_markdown(report), encoding="utf-8")
    return output, markdown


def _positive(parser: argparse.ArgumentParser, values: Sequence[int], label: str) -> None:
    if any(value <= 0 for value in values):
        parser.error(f"{label} must contain positive integers")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workloads",
        nargs="+",
        choices=WORKLOAD_NAMES,
        default=list(WORKLOAD_NAMES),
    )
    parser.add_argument("--worlds", nargs="+", type=int)
    parser.add_argument("--warm-steps", type=int)
    parser.add_argument("--backward-steps", type=int)
    parser.add_argument("--subsystem-steps", type=int)
    parser.add_argument("--baseline-steps", type=int)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="CI-sized matrix: one world and one sample per expensive metric",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/benchmarks/latest.json"),
    )
    args = parser.parse_args(argv)
    worlds = tuple(args.worlds or (QUICK_BATCHES if args.quick else DEFAULT_BATCHES))
    warm_steps = (
        args.warm_steps
        if args.warm_steps is not None
        else (1 if args.quick else 20)
    )
    backward_steps = (
        args.backward_steps
        if args.backward_steps is not None
        else (0 if args.quick else 5)
    )
    subsystem_steps = (
        args.subsystem_steps
        if args.subsystem_steps is not None
        else (0 if args.quick else 5)
    )
    baseline_steps = (
        args.baseline_steps
        if args.baseline_steps is not None
        else (3 if args.quick else 100)
    )
    _positive(parser, worlds, "--worlds")
    _positive(parser, (warm_steps,), "--warm-steps")
    if min(backward_steps, subsystem_steps, baseline_steps) < 0:
        parser.error("sample counts must be nonnegative")
    report = run_matrix(
        args.workloads,
        worlds,
        warm_steps=warm_steps,
        backward_steps=backward_steps,
        subsystem_steps=subsystem_steps,
        baseline_steps=baseline_steps,
    )
    json_path, markdown_path = write_report(report, args.output)
    print(f"wrote {json_path}")
    print(f"wrote {markdown_path}")
    print(
        f"completed_cases={len(report['cases'])} failures={len(report['failures'])}"
    )
    return 1 if report["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
