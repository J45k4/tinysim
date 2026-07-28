"""Measure TinyJit compile/capture memory in a fresh process."""

from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import replace
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
from typing import Any

from tinygrad import Device, GlobalCounters
from tinygrad.uop.ops import UOpMetaClass

from examples.jenga import initial_state, jenga_model
from tinysim import Simulation


ACCEPTANCE_RSS_BYTES = 6_000_000_000


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


def _snapshot(label: str, simulation: Simulation) -> dict[str, Any]:
    jit = simulation.simulator._inference_step
    captured = None if jit is None else jit.captured
    linear = None if captured is None else captured._linear
    return {
        "label": label,
        "current_rss_bytes": _current_rss_bytes(),
        "peak_rss_bytes": _peak_rss_bytes(),
        "tinygrad_resident_bytes": int(
            GlobalCounters.mem_used_per_device[Device.DEFAULT]
        ),
        "live_uops": len(UOpMetaClass.ucache),
        "captured_calls": None if linear is None else len(linear.src),
    }


def _run_child(
    *,
    levels: int,
    worlds: int,
    pair_set: str,
) -> dict[str, Any]:
    spec = jenga_model(levels)
    if pair_set == "none":
        spec = replace(spec, contact=replace(spec.contact, mode="none"))
    elif pair_set == "floor":
        spec = replace(
            spec,
            collision_pairs=spec.collision_pairs[: 3 * levels],
        )
    simulation = Simulation(spec, worlds=worlds)
    qpos, qvel = initial_state(levels, worlds, push=0.8)
    simulation.reset(qpos=qpos, qvel=qvel)
    snapshots = [_snapshot("reset", simulation)]
    for call in range(1, 5):
        started = time.perf_counter()
        simulation.run(steps=1)
        Device[Device.DEFAULT].synchronize()
        snapshot = _snapshot(f"call-{call}", simulation)
        snapshot["elapsed_seconds"] = time.perf_counter() - started
        snapshots.append(snapshot)
    peak_rss = max(snapshot["peak_rss_bytes"] for snapshot in snapshots)
    return {
        "device": Device.DEFAULT,
        "levels": levels,
        "worlds": worlds,
        "pair_set": pair_set,
        "pairs": len(simulation.simulator.model.collision_pairs),
        "dofs": simulation.simulator.model.nv,
        "rss_limit_bytes": ACCEPTANCE_RSS_BYTES,
        "peak_rss_bytes": peak_rss,
        "memory_gate_passed": peak_rss < ACCEPTANCE_RSS_BYTES,
        "snapshots": snapshots,
    }


def _run_parent(
    *,
    levels: int,
    worlds: int,
    pair_set: str,
) -> dict[str, Any]:
    command = [
        sys.executable,
        __file__,
        "--child",
        "--levels",
        str(levels),
        "--worlds",
        str(worlds),
        "--pair-set",
        pair_set,
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        env=os.environ.copy(),
        text=True,
    )
    if completed.returncode:
        raise RuntimeError(
            "compiler-memory child failed:\n"
            f"{completed.stderr.strip()}"
        )
    return json.loads(completed.stdout)


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--levels", type=int, default=2)
    parser.add_argument("--worlds", type=int, default=1)
    parser.add_argument(
        "--pair-set",
        choices=("none", "floor", "support"),
        default="support",
    )
    parser.add_argument(
        "--acceptance",
        action="store_true",
        help="run the 10-level, 256-world, less-than-6-GB gate",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--child", action="store_true", help="internal")
    args = parser.parse_args()
    levels, worlds = (
        (10, 256)
        if args.acceptance
        else (args.levels, args.worlds)
    )
    if min(levels, worlds) < 1:
        parser.error("--levels and --worlds must be positive")
    payload = (
        _run_child(
            levels=levels,
            worlds=worlds,
            pair_set=args.pair_set,
        )
        if args.child
        else _run_parent(
            levels=levels,
            worlds=worlds,
            pair_set=args.pair_set,
        )
    )
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    if not args.child and not payload["memory_gate_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
