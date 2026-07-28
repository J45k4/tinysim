"""Stress TinySim with a pushed tower of free rigid boxes.

This is intentionally a collapse test, not a tuned stable-stacking benchmark.
Box pairs use fixed four-slot SAT manifolds with inactive edge/vertex slots.
The scenario stresses box-box collision, rotation, friction, independent
free-body dynamics, TinyJit, and optional batched grid recording.
"""

from argparse import ArgumentParser
from math import sqrt
from pathlib import Path

from tinysim import (
    BodySpec,
    ContactSpec,
    GeomSpec,
    JointSpec,
    ModelSpec,
    Simulation,
)


HALF_SIZE = (0.30, 0.10, 0.06)
MASS = 0.20
TIMESTEP = 0.001


def box_inertia() -> tuple[float, float, float]:
    x, y, z = HALF_SIZE
    return (
        MASS * (y * y + z * z) / 3.0,
        MASS * (x * x + z * z) / 3.0,
        MASS * (x * x + y * y) / 3.0,
    )


def collision_pairs(
    levels: int,
    mode: str,
) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for block in range(3 * levels):
        pairs.append((f"block_geom_{block}", "floor"))
    if mode == "support":
        pairs.extend(
            (
                f"block_geom_{3 * level + slot}",
                f"block_geom_{3 * (level + 1) + slot}",
            )
            for level in range(levels - 1)
            for slot in range(3)
        )
        return pairs
    for level in range(levels):
        blocks = tuple(3 * level + slot for slot in range(3))
        pairs.extend(
            (
                f"block_geom_{blocks[left]}",
                f"block_geom_{blocks[right]}",
            )
            for left in range(3)
            for right in range(left + 1, 3)
        )
    for level in range(levels - 1):
        lower = tuple(3 * level + slot for slot in range(3))
        upper = tuple(3 * (level + 1) + slot for slot in range(3))
        pairs.extend(
            (f"block_geom_{a}", f"block_geom_{b}")
            for a in lower
            for b in upper
        )
    return pairs


def jenga_model(levels: int, *, pair_mode: str = "support") -> ModelSpec:
    if levels < 1:
        raise ValueError("levels must be positive")
    if pair_mode not in ("support", "local", "all"):
        raise ValueError("pair_mode must be support, local, or all")
    blocks = 3 * levels
    return ModelSpec(
        name=f"jenga_{levels}_levels",
        bodies=[
            BodySpec(
                f"block_{index}",
                mass=MASS,
                inertia=box_inertia(),
            )
            for index in range(blocks)
        ],
        joints=[
            JointSpec(f"free_{index}", body=index, kind="free")
            for index in range(blocks)
        ],
        geoms=[
            GeomSpec(
                f"block_geom_{index}",
                body=index,
                kind="box",
                size=HALF_SIZE,
                friction=0.8,
            )
            for index in range(blocks)
        ]
        + [
            GeomSpec(
                "floor",
                body=-1,
                kind="plane",
                size=(0.0, 0.0, 0.1),
                friction=0.9,
            )
        ],
        collision_pairs=(
            ()
            if pair_mode == "all"
            else collision_pairs(levels, pair_mode)
        ),
        contact=ContactSpec(
            mode="smooth",
            stiffness=2_000.0,
            damping=20.0,
            friction=1.0,
            penetration_smoothing=1e-5,
            velocity_smoothing=1e-3,
        ),
        timestep=TIMESTEP,
    )


def initial_state(
    levels: int,
    worlds: int,
    *,
    push: float,
) -> tuple[list[list[float]], list[list[float]]]:
    quarter_turn = sqrt(0.5)
    positions: list[list[float]] = []
    velocities: list[list[float]] = []
    for world in range(worlds):
        qpos: list[float] = []
        qvel = [0.0] * (levels * 3 * 6)
        for level in range(levels):
            rotated = level % 2 == 1
            for slot in range(3):
                lateral = (slot - 1) * 2.05 * HALF_SIZE[1]
                x, y = (lateral, 0.0) if rotated else (0.0, lateral)
                z = HALF_SIZE[2] + level * 2.0 * HALF_SIZE[2]
                quaternion = (
                    (quarter_turn, 0.0, 0.0, quarter_turn)
                    if rotated
                    else (1.0, 0.0, 0.0, 0.0)
                )
                qpos.extend((x, y, z, *quaternion))
        top_middle = 3 * (levels - 1) + 1
        qvel[6 * top_middle] = push * (1.0 + 0.08 * (world % 16))
        positions.append(qpos)
        velocities.append(qvel)
    return positions, velocities


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--levels", type=int, default=2)
    parser.add_argument("--worlds", type=int, default=16)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--push", type=float, default=0.8)
    parser.add_argument(
        "--pair-mode",
        choices=("support", "local", "all"),
        default="support",
        help="collision graph density (default: support)",
    )
    parser.add_argument("--record", type=Path)
    parser.add_argument(
        "--record-grid",
        type=int,
        metavar="COUNT",
        help="record the first COUNT worlds in one video",
    )
    parser.add_argument("--grid-columns", type=int)
    parser.add_argument(
        "--record-every",
        type=int,
        default=10,
        help="store one trajectory state per N simulation steps",
    )
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=960)
    args = parser.parse_args()
    if min(args.levels, args.worlds, args.steps) < 1:
        parser.error("--levels, --worlds, and --steps must be positive")
    if args.push < 0.0:
        parser.error("--push must be nonnegative")
    if not 1 <= args.record_every <= args.steps:
        parser.error("--record-every must be between 1 and --steps")
    if args.record_grid is not None:
        if args.record is None:
            parser.error("--record-grid requires --record")
        if not 1 <= args.record_grid <= args.worlds:
            parser.error("--record-grid must be between 1 and --worlds")
    if args.grid_columns is not None:
        if args.record_grid is None:
            parser.error("--grid-columns requires --record-grid")
        if not 1 <= args.grid_columns <= args.record_grid:
            parser.error(
                "--grid-columns must be between 1 and --record-grid"
            )

    model = jenga_model(args.levels, pair_mode=args.pair_mode)
    blocks = 3 * args.levels
    pairs = (
        blocks * (blocks - 1) // 2 + blocks
        if args.pair_mode == "all"
        else len(collision_pairs(args.levels, args.pair_mode))
    )
    print(
        f"compiling {blocks} free blocks, {6 * blocks} DoFs, "
        f"{pairs} fixed collision pairs ({args.pair_mode}), "
        f"{args.worlds} worlds"
    )
    simulation = Simulation(model, worlds=args.worlds)
    qpos, qvel = initial_state(
        args.levels,
        args.worlds,
        push=args.push,
    )
    simulation.reset(qpos=qpos, qvel=qvel)
    simulation.run(
        steps=args.steps,
        record=args.record,
        worlds=(
            range(args.record_grid)
            if args.record_grid is not None
            else None
        ),
        columns=args.grid_columns,
        record_every=args.record_every,
        fps=args.fps,
        width=args.width,
        height=args.height,
    )

    final = simulation.state.qpos[0].clone().realize().tolist()
    heights = [final[7 * block + 2] for block in range(blocks)]
    radial_spread = max(
        sqrt(
            final[7 * block] ** 2
            + final[7 * block + 1] ** 2
        )
        for block in range(blocks)
    )
    maximum_speed = float(
        simulation.state.qvel.abs().max().item()
    )
    maximum_position = float(
        simulation.state.qpos.abs().max().item()
    )
    finite = bool(
        simulation.state.qpos.isfinite().all().item()
        and simulation.state.qvel.isfinite().all().item()
    )
    bounded = maximum_position < 10.0 and maximum_speed < 100.0
    print(
        f"world 0: finite={finite}, bounded={bounded}, "
        f"height range=[{min(heights):.4f}, {max(heights):.4f}] m, "
        f"radial spread={radial_spread:.4f} m, "
        f"max speed={maximum_speed:.4f}"
    )
    if not finite or not bounded:
        raise RuntimeError("Jenga stress rollout became non-finite or unbounded")
    if args.record:
        print(
            f"video={args.record}; "
            f"trajectory={args.record.with_suffix('.trajectory.tstraj')}"
        )


if __name__ == "__main__":
    main()
