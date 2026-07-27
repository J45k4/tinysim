"""Run 4,096 smooth-contact spheres above a plane.

Parameters: unit mass, radius=0.25 m, gravity=-9.81 m/s², contact stiffness
5,000 N/m, damping=20 N·s/m, dt=0.002 s, float32 on the selected tinygrad
device. Integration is semi-implicit Euler and there is no constraint solver.
The expected behavior is a bounded rebound followed by settling near z=0.25 m.

Pass ``--record output.mp4`` to save and reload one world's trajectory before
rendering it with the optional host-side encoder.
"""

from argparse import ArgumentParser
from pathlib import Path

from tinysim import (
    BodySpec,
    ContactSpec,
    GeomSpec,
    JointSpec,
    ModelSpec,
    Simulation,
)


TIMESTEP = 0.002


def bouncing_ball_model() -> ModelSpec:
    return ModelSpec(
        name="bouncing_ball",
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
            damping=20.0,
            friction=0.0,
        ),
        timestep=TIMESTEP,
    )


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--worlds", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--record", type=Path, help="optional MP4 output path")
    parser.add_argument(
        "--record-world",
        type=int,
        default=0,
        help="single batched world to render (default: 0)",
    )
    parser.add_argument(
        "--record-grid",
        type=int,
        metavar="COUNT",
        help="record the first COUNT worlds as one tiled video",
    )
    parser.add_argument(
        "--grid-columns",
        type=int,
        help="number of video grid columns (default: automatic)",
    )
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()
    if args.worlds <= 0 or args.steps <= 0:
        parser.error("--worlds and --steps must be positive")
    if not 0 <= args.record_world < args.worlds:
        parser.error("--record-world must index an existing world")
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
    if args.fps <= 0:
        parser.error("--fps must be positive")
    if min(args.width, args.height) < 16:
        parser.error("--width and --height must be at least 16")

    simulation = Simulation(bouncing_ball_model(), worlds=args.worlds)
    if args.record_grid is None:
        initial_qpos = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    else:
        initial_qpos = [
            [
                0.0,
                0.0,
                1.0 + 0.05 * (index % 16),
                1.0,
                0.0,
                0.0,
                0.0,
            ]
            for index in range(args.worlds)
        ]
    simulation.reset(qpos=initial_qpos)
    simulation.run(
        steps=args.steps,
        record=args.record,
        world=args.record_world,
        worlds=(
            range(args.record_grid)
            if args.record_grid is not None
            else None
        ),
        columns=args.grid_columns,
        fps=args.fps,
        width=args.width,
        height=args.height,
    )
    state = simulation.state
    world_position = state.qpos[0].clone().realize().tolist()
    world_velocity = state.qvel[0].clone().realize().tolist()
    print(
        f"world 0 after {args.steps} steps: "
        f"height={world_position[2]:.6f}, "
        f"vertical_velocity={world_velocity[2]:.6f}"
    )
    if args.record:
        trajectory_path = args.record.with_suffix(".trajectory.json")
        recorded = (
            f"worlds 0..{args.record_grid - 1}"
            if args.record_grid is not None
            else f"world {args.record_world}"
        )
        print(
            f"recorded {recorded}: {args.record}; "
            f"trajectory={trajectory_path}"
        )


if __name__ == "__main__":
    main()
