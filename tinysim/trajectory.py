"""Portable host-side trajectory and verification artifact format."""

from dataclasses import asdict, dataclass, field
import json
from math import isfinite
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Trajectory:
    scenario: str
    timestep: float
    qpos: list[list[float]]
    qvel: list[list[float]]
    controls: list[list[float]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.scenario:
            raise ValueError("scenario must not be empty")
        if not isfinite(self.timestep) or self.timestep <= 0:
            raise ValueError("timestep must be finite and positive")
        if not self.qpos or len(self.qpos) != len(self.qvel):
            raise ValueError("qpos and qvel must contain the same nonzero frame count")
        if self.controls and len(self.controls) not in (len(self.qpos), len(self.qpos) - 1):
            raise ValueError("controls must be empty, per-frame, or per-transition")
        widths = {len(row) for row in self.qpos}
        velocity_widths = {len(row) for row in self.qvel}
        control_widths = {len(row) for row in self.controls}
        if len(widths) != 1 or len(velocity_widths) != 1 or len(control_widths) > 1:
            raise ValueError("trajectory rows must have stable dimensions")
        if not widths or next(iter(widths)) == 0:
            raise ValueError("trajectory coordinates must not be empty")
        if not velocity_widths or next(iter(velocity_widths)) == 0:
            raise ValueError("trajectory velocities must not be empty")
        values = (
            value
            for collection in (self.qpos, self.qvel, self.controls)
            for row in collection
            for value in row
        )
        if not all(isfinite(value) for value in values):
            raise ValueError("trajectory contains a non-finite value")


def save_trajectory(trajectory: Trajectory, path: str | Path) -> None:
    trajectory.validate()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(asdict(trajectory), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_trajectory(path: str | Path) -> Trajectory:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    trajectory = Trajectory(**payload)
    trajectory.validate()
    return trajectory


def playback_indices(frame_count: int, *, timestep: float, fps: int) -> list[int]:
    """Map fixed-step states to video frames without changing playback speed.

    A trajectory with ``N`` states spans ``(N - 1) * timestep`` seconds.
    Nearest-state sampling keeps video duration within half a frame of that
    simulated duration and deliberately performs no tensor work.
    """
    if frame_count < 2:
        raise ValueError("frame_count must include at least two states")
    if not isfinite(timestep) or timestep <= 0:
        raise ValueError("timestep must be finite and positive")
    if fps <= 0:
        raise ValueError("fps must be positive")
    transitions = frame_count - 1
    video_frames = max(1, round(transitions * timestep * fps))
    return [
        min(round(frame / (fps * timestep)), transitions)
        for frame in range(video_frames)
    ]
