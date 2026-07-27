"""Measure TinySim recording memory in isolated subprocesses."""

from __future__ import annotations

from argparse import ArgumentParser
from array import array
from datetime import datetime, timezone
import json
from pathlib import Path
import resource
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any

from tinygrad import Device, GlobalCounters

from examples.jenga import initial_state as jenga_initial_state
from examples.jenga import jenga_model
from tinysim import (
    BodySpec,
    ContactSpec,
    GeomSpec,
    JointSpec,
    ModelSpec,
    Simulation,
)
from tinysim.trajectory import TrajectoryWriter


QUICK_CASES = (
    "baseline",
    "ball-unrecorded",
    "ball-one",
    "ball-grid",
    "jenga-grid",
    "stream-101",
    "stream-1001",
    "stream-10001",
)
FULL_CASES = QUICK_CASES


def _current_rss_bytes() -> int | None:
    status = Path("/proc/self/status")
    if not status.is_file():
        return None
    for line in status.read_text(encoding="utf-8").splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    return None


def _peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def _snapshot(label: str) -> dict[str, Any]:
    return {
        "label": label,
        "current_rss_bytes": _current_rss_bytes(),
        "peak_rss_bytes": _peak_rss_bytes(),
        "tinygrad_resident_bytes": int(
            GlobalCounters.mem_used_per_device[Device.DEFAULT]
        ),
    }


def _ball_model() -> ModelSpec:
    return ModelSpec(
        name="memory_ball",
        bodies=[
            BodySpec(
                "ball",
                mass=1.0,
                inertia=(0.025, 0.025, 0.025),
            )
        ],
        joints=[JointSpec("free", body=0, kind="free")],
        geoms=[
            GeomSpec("ball", body=0, kind="sphere", size=(0.25,)),
            GeomSpec(
                "floor",
                body=-1,
                kind="plane",
                size=(0.0, 0.0, 0.1),
            ),
        ],
        contact=ContactSpec(
            mode="smooth",
            stiffness=5_000.0,
            damping=100.0,
            friction=0.0,
        ),
        timestep=0.002,
    )


def _run_ball(
    *,
    recorded_worlds: int | None,
    full: bool,
    directory: Path,
    snapshots: list[dict[str, Any]],
) -> dict[str, Any]:
    worlds = 256
    steps = 2_000 if full else 50
    simulation = Simulation(_ball_model(), worlds=worlds)
    simulation.reset(
        qpos=[0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    )
    simulation.run(steps=3)
    snapshots.append(_snapshot("warmup"))
    started = time.perf_counter()
    video = directory / "ball.mp4"
    record_every = 10 if full else 5
    if recorded_worlds is None:
        simulation.run(steps=steps)
    else:
        simulation.run(
            steps=steps,
            record=video,
            worlds=range(recorded_worlds),
            columns=(
                (8 if full else 2)
                if recorded_worlds > 1
                else None
            ),
            record_every=record_every,
            width=640 if full else 128,
            height=480 if full else 96,
        )
    elapsed = time.perf_counter() - started
    snapshots.append(_snapshot("complete"))
    trajectory = video.with_suffix(".trajectory.tstraj")
    raw_payload_bytes = (
        (steps // record_every + 1) * recorded_worlds * (7 + 6) * 4
        if recorded_worlds is not None
        else 0
    )
    return {
        "worlds": worlds,
        "recorded_worlds": recorded_worlds or 0,
        "steps": steps,
        "elapsed_seconds": elapsed,
        "trajectory_bytes": (
            trajectory.stat().st_size if trajectory.is_file() else 0
        ),
        "raw_payload_bytes": raw_payload_bytes,
        "video_bytes": video.stat().st_size if video.is_file() else 0,
    }


def _run_jenga(
    *,
    full: bool,
    directory: Path,
    snapshots: list[dict[str, Any]],
) -> dict[str, Any]:
    levels, worlds, recorded_worlds, steps = (
        (10, 256, 64, 1_000) if full else (2, 16, 4, 20)
    )
    simulation = Simulation(jenga_model(levels), worlds=worlds)
    qpos, qvel = jenga_initial_state(levels, worlds, push=0.8)
    simulation.reset(qpos=qpos, qvel=qvel)
    simulation.run(steps=3)
    snapshots.append(_snapshot("warmup"))
    video = directory / "jenga.mp4"
    record_every = 10 if full else 2
    started = time.perf_counter()
    simulation.run(
        steps=steps,
        record=video,
        worlds=range(recorded_worlds),
        columns=8 if full else 2,
        record_every=record_every,
        width=1280 if full else 128,
        height=960 if full else 96,
    )
    elapsed = time.perf_counter() - started
    snapshots.append(_snapshot("complete"))
    trajectory = video.with_suffix(".trajectory.tstraj")
    blocks = levels * 3
    raw_payload_bytes = (
        (steps // record_every + 1)
        * recorded_worlds
        * (7 * blocks + 6 * blocks)
        * 4
    )
    return {
        "levels": levels,
        "worlds": worlds,
        "recorded_worlds": recorded_worlds,
        "steps": steps,
        "elapsed_seconds": elapsed,
        "raw_payload_bytes": raw_payload_bytes,
        "trajectory_bytes": trajectory.stat().st_size,
        "video_bytes": video.stat().st_size,
    }


def _run_stream(
    *,
    frames: int,
    directory: Path,
    snapshots: list[dict[str, Any]],
) -> dict[str, Any]:
    qpos_width, qvel_width = 256, 256
    payload = array("f", [0.0] * (qpos_width + qvel_width))
    path = directory / "synthetic.tstraj"
    snapshots.append(_snapshot("before-write"))
    started = time.perf_counter()
    with TrajectoryWriter(
        path,
        scenario="synthetic",
        timestep=0.01,
        qpos_width=qpos_width,
        qvel_width=qvel_width,
        control_width=0,
    ) as writer:
        for _ in range(frames):
            writer.append(payload)
    elapsed = time.perf_counter() - started
    snapshots.append(_snapshot("after-write"))
    return {
        "frames": frames,
        "raw_payload_bytes": frames * len(payload) * payload.itemsize,
        "trajectory_bytes": path.stat().st_size,
        "elapsed_seconds": elapsed,
    }


def _run_child(case: str, *, full: bool) -> dict[str, Any]:
    snapshots = [_snapshot("imported")]
    with TemporaryDirectory() as temporary:
        directory = Path(temporary)
        if case == "baseline":
            details: dict[str, Any] = {}
        elif case == "ball-unrecorded":
            details = _run_ball(
                recorded_worlds=None,
                full=full,
                directory=directory,
                snapshots=snapshots,
            )
        elif case == "ball-one":
            details = _run_ball(
                recorded_worlds=1,
                full=full,
                directory=directory,
                snapshots=snapshots,
            )
        elif case == "ball-grid":
            details = _run_ball(
                recorded_worlds=64 if full else 4,
                full=full,
                directory=directory,
                snapshots=snapshots,
            )
        elif case == "jenga-grid":
            details = _run_jenga(
                full=full,
                directory=directory,
                snapshots=snapshots,
            )
        elif case.startswith("stream-"):
            details = _run_stream(
                frames=int(case.removeprefix("stream-")),
                directory=directory,
                snapshots=snapshots,
            )
        else:
            raise ValueError(f"unknown benchmark case {case!r}")
    result = {
        "case": case,
        "device": Device.DEFAULT,
        "full": full,
        "snapshots": snapshots,
        **details,
    }
    if len(snapshots) > 1:
        before, after = snapshots[-2:]
        result["current_rss_delta_bytes"] = (
            None
            if before["current_rss_bytes"] is None
            or after["current_rss_bytes"] is None
            else after["current_rss_bytes"] - before["current_rss_bytes"]
        )
        result["peak_rss_delta_bytes"] = (
            after["peak_rss_bytes"] - before["peak_rss_bytes"]
        )
    if "raw_payload_bytes" in result:
        result["trajectory_overhead_bytes"] = (
            result["trajectory_bytes"] - result["raw_payload_bytes"]
        )
        result["size_gate_passed"] = result["trajectory_overhead_bytes"] <= max(
            result["raw_payload_bytes"] // 100,
            64 * 1024,
        )
    if case == "jenga-grid" and full:
        result["memory_gate_passed"] = (
            result["peak_rss_delta_bytes"] <= 32 * 1024 * 1024
        )
    if case == "stream-10001":
        result["memory_gate_passed"] = (
            result["peak_rss_delta_bytes"] <= 8 * 1024 * 1024
        )
    return result


def _run_parent(*, full: bool) -> dict[str, Any]:
    cases = FULL_CASES if full else QUICK_CASES
    results = []
    for case in cases:
        command = [sys.executable, __file__, "--child", case]
        if full:
            command.append("--full")
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
        )
        if completed.returncode:
            raise RuntimeError(
                f"recording-memory case {case!r} failed:\n"
                f"{completed.stderr.strip()}"
            )
        results.append(json.loads(completed.stdout))
    failures = [
        f"{result['case']}:{gate}"
        for result in results
        for gate in ("memory_gate_passed", "size_gate_passed")
        if result.get(gate) is False
    ]
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "full" if full else "quick",
        "failures": failures,
        "results": results,
    }


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--child", choices=QUICK_CASES)
    parser.add_argument(
        "--full",
        action="store_true",
        help="run the 10-level, 256-world Jenga acceptance workload",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = (
        _run_child(args.child, full=args.full)
        if args.child
        else _run_parent(full=args.full)
    )
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
