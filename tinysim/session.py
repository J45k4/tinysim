"""Stateful simulation runs with optional host-side trajectory recording."""

from collections.abc import Callable, Sequence
from math import isqrt
from pathlib import Path

from tinygrad import Tensor, dtypes

from .model import ModelSpec
from .render import Camera2D, encode_mp4, render_model, render_model_grid
from .simulation import Simulator
from .state import State, validate_state
from .trajectory import (
    TrajectoryReader,
    TrajectoryWriter,
    iter_playback_indices,
)


FrameRenderer = Callable[[Sequence[float]], bytes]
GridFrameRenderer = Callable[[Sequence[Sequence[float]]], bytes]


def _batched_value(
    value: Tensor | Sequence[float] | Sequence[Sequence[float]],
    *,
    worlds: int,
    width: int,
    template: Tensor,
    name: str,
) -> Tensor:
    tensor = (
        value
        if isinstance(value, Tensor)
        else Tensor(value, dtype=template.dtype, device=template.device)
    )
    if tensor.dtype != template.dtype or tensor.device != template.device:
        raise TypeError(f"{name} must match the model dtype and device")
    if tensor.shape == (width,):
        tensor = tensor.reshape(1, width).expand(worlds, width)
    if tensor.shape != (worlds, width):
        raise ValueError(
            f"{name} must have shape [{width}] or [{worlds}, {width}]"
        )
    return tensor


class Simulation:
    """Owns one simulator state and optionally records individual rollouts."""

    def __init__(
        self,
        model: ModelSpec,
        *,
        worlds: int = 1,
        device: str | None = None,
        dtype: object | str = dtypes.float32,
        contact: str | None = None,
        solver_iterations: int | None = None,
    ):
        self.simulator = Simulator.compile(
            model,
            worlds=worlds,
            device=device,
            dtype=dtype,
            contact=contact,
            solver_iterations=solver_iterations,
        )
        self.state = self.simulator.make_state()

    @classmethod
    def from_compiled(
        cls, simulator: Simulator, *, state: State | None = None
    ) -> "Simulation":
        """Reuses a compiled simulator and an optional checkpoint state."""

        current = simulator.make_state() if state is None else state
        validate_state(simulator.model, current)
        if current.qpos.shape[0] != simulator.worlds:
            raise ValueError(
                f"state world dimension must be {simulator.worlds}"
            )
        simulation = cls.__new__(cls)
        simulation.simulator = simulator
        simulation.state = current
        return simulation

    @property
    def worlds(self) -> int:
        return self.simulator.worlds

    def reset(
        self,
        *,
        qpos: Tensor | Sequence[float] | Sequence[Sequence[float]] | None = None,
        qvel: Tensor | Sequence[float] | Sequence[Sequence[float]] | None = None,
        control: Tensor | Sequence[float] | Sequence[Sequence[float]] | None = None,
    ) -> State:
        """Resets every world, broadcasting one supplied state when requested."""

        fresh = self.simulator.make_state()
        self.state = State(
            qpos=(
                fresh.qpos
                if qpos is None
                else _batched_value(
                    qpos,
                    worlds=self.worlds,
                    width=self.simulator.model.nq,
                    template=fresh.qpos,
                    name="qpos",
                )
            ),
            qvel=(
                fresh.qvel
                if qvel is None
                else _batched_value(
                    qvel,
                    worlds=self.worlds,
                    width=self.simulator.model.nv,
                    template=fresh.qvel,
                    name="qvel",
                )
            ),
            ctrl=(
                fresh.ctrl
                if control is None
                else _batched_value(
                    control,
                    worlds=self.worlds,
                    width=self.simulator.model.nu,
                    template=fresh.ctrl,
                    name="control",
                )
            ),
            time=fresh.time,
            constraint_impulse=fresh.constraint_impulse,
            parameters=fresh.parameters,
        )
        validate_state(self.simulator.model, self.state)
        return self.state

    def run(
        self,
        *,
        steps: int,
        control: Tensor | None = None,
        inference: bool = True,
        record: str | Path | None = None,
        world: int = 0,
        worlds: Sequence[int] | None = None,
        columns: int | None = None,
        record_every: int = 1,
        fps: int = 60,
        width: int = 640,
        height: int = 480,
        camera: Camera2D = Camera2D(),
        renderer: FrameRenderer | None = None,
        grid_renderer: GridFrameRenderer | None = None,
    ) -> State:
        """Advance state and optionally record one world or a grid of worlds."""

        if steps < 1:
            raise ValueError("steps must be positive")
        recorded_worlds = (world,) if worlds is None else tuple(worlds)
        if record is not None:
            if not recorded_worlds:
                raise ValueError("recording worlds must not be empty")
            if len(set(recorded_worlds)) != len(recorded_worlds):
                raise ValueError("recording worlds must be unique")
            if any(
                not isinstance(index, int)
                or not 0 <= index < self.worlds
                for index in recorded_worlds
            ):
                raise ValueError(
                    "recording worlds must index existing worlds"
                )
            if columns is not None and not 1 <= columns <= len(recorded_worlds):
                raise ValueError(
                    "grid columns must be between one and the recorded world count"
                )
            if len(recorded_worlds) > 1 and renderer is not None:
                raise ValueError(
                    "use grid_renderer for multi-world recording"
                )
            if fps <= 0:
                raise ValueError("recording fps must be positive")
            if not 1 <= record_every <= steps:
                raise ValueError(
                    "record_every must be between one and the step count"
                )
            if (
                min(width, height) < 16
                or width % 2
                or height % 2
            ):
                raise ValueError(
                    "recording dimensions must be even and at least 16"
                )
            if len(recorded_worlds) > 1 and grid_renderer is None:
                grid_columns = columns or isqrt(len(recorded_worlds) - 1) + 1
                grid_rows = (
                    len(recorded_worlds) + grid_columns - 1
                ) // grid_columns
                if (
                    width // grid_columns < 16
                    or height // grid_rows < 16
                ):
                    raise ValueError(
                        "recording grid tiles must be at least 16 pixels "
                        "in each dimension"
                    )
        if record is None:
            for _ in range(steps):
                self.state = (
                    self.simulator.inference_step(self.state, control)
                    if inference
                    else self.simulator.step(self.state, control)
                )
            return self.state

        video_path = Path(record)
        trajectory_path = video_path.with_suffix(".trajectory.tstraj")
        model = self.simulator.model
        dtype = (
            "float32"
            if model.dtype == dtypes.float32
            else "float64"
            if model.dtype == dtypes.float64
            else None
        )
        if dtype is None:
            raise TypeError("recording supports float32 and float64")
        selected_count = len(recorded_worlds)

        def capture(writer: TrajectoryWriter) -> None:
            fields = [self.state.qpos, self.state.qvel]
            if model.nu:
                fields.append(self.state.ctrl)
            selected_fields = [
                Tensor.stack(
                    *(tensor[index] for index in recorded_worlds),
                    dim=0,
                ).flatten()
                for tensor in fields
            ]
            packed = Tensor.cat(*selected_fields).contiguous().realize()
            writer.append(packed.data())

        with TrajectoryWriter(
            trajectory_path,
            scenario=model.name,
            timestep=model.timestep * record_every,
            qpos_width=selected_count * model.nq,
            qvel_width=selected_count * model.nv,
            control_width=selected_count * model.nu,
            dtype=dtype,
            metadata={
                "recorded_world": recorded_worlds[0],
                "recorded_worlds": list(recorded_worlds),
                "worlds": self.worlds,
                "grid_columns": columns,
                "record_every": record_every,
                "simulation_steps": steps,
                "simulation_timestep": model.timestep,
            },
        ) as writer:
            capture(writer)
            for step_index in range(steps):
                self.state = (
                    self.simulator.inference_step(self.state, control)
                    if inference
                    else self.simulator.step(self.state, control)
                )
                if (step_index + 1) % record_every == 0:
                    capture(writer)

        nq = self.simulator.model.nq
        if len(recorded_worlds) == 1:
            render_frame = renderer or (
                lambda position: render_model(
                    self.simulator.model,
                    position,
                    width=width,
                    height=height,
                    camera=camera,
                )
            )
        else:
            render_frame = grid_renderer or (
                lambda position: render_model_grid(
                    self.simulator.model,
                    [
                        position[offset : offset + nq]
                        for offset in range(0, len(position), nq)
                    ],
                    width=width,
                    height=height,
                    columns=columns,
                    camera=camera,
                )
            )
        with TrajectoryReader(trajectory_path) as trajectory:
            indices = iter_playback_indices(
                trajectory.frame_count,
                timestep=trajectory.timestep,
                fps=fps,
            )
            encode_mp4(
                (
                    render_frame(trajectory.read_qpos(index))
                    for index in indices
                ),
                video_path,
                width=width,
                height=height,
                fps=fps,
            )
        return self.state

    def record(
        self,
        path: str | Path,
        *,
        steps: int,
        world: int = 0,
        worlds: Sequence[int] | None = None,
        columns: int | None = None,
        record_every: int = 1,
        fps: int = 60,
        width: int = 640,
        height: int = 480,
        camera: Camera2D = Camera2D(),
        control: Tensor | None = None,
        inference: bool = True,
        renderer: FrameRenderer | None = None,
        grid_renderer: GridFrameRenderer | None = None,
    ) -> State:
        """Convenience alias for a recorded :meth:`run`."""

        return self.run(
            steps=steps,
            control=control,
            inference=inference,
            record=path,
            world=world,
            worlds=worlds,
            columns=columns,
            record_every=record_every,
            fps=fps,
            width=width,
            height=height,
            camera=camera,
            renderer=renderer,
            grid_renderer=grid_renderer,
        )
