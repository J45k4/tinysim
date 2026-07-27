import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from tinygrad import Tensor

from tinysim import (
    BodySpec,
    ContactSpec,
    GeomSpec,
    JointSpec,
    ModelSpec,
    Simulation,
    Simulator,
)
from tinysim.render import render_model, render_model_grid


def ball_spec() -> ModelSpec:
    return ModelSpec(
        name="recorded_ball",
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


class TestSimulationSession(unittest.TestCase):
    def test_reset_broadcasts_one_state_to_every_world(self):
        simulation = Simulation(ball_spec(), worlds=3)
        state = simulation.reset(
            qpos=[0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]
        )
        self.assertEqual(state.qpos.shape, (3, 7))
        values = state.qpos.tolist()
        self.assertEqual(values, [values[0]] * 3)

    def test_unrecorded_run_performs_no_host_list_read(self):
        simulation = Simulation(ball_spec(), worlds=2)
        simulation.reset(
            qpos=[0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]
        )
        with patch.object(
            Tensor,
            "tolist",
            side_effect=AssertionError("unrecorded run read host state"),
        ):
            state = simulation.run(steps=3)
        self.assertAlmostEqual(float(state.time[0].item()), 0.006, places=6)

    def test_record_alias_saves_reloads_and_renders_trajectory(self):
        encoded: dict[str, object] = {}

        def fake_encode(frames, path, *, width, height, fps):
            encoded.update(
                frames=list(frames), width=width, height=height, fps=fps
            )
            Path(path).write_bytes(b"mock mp4")

        simulation = Simulation(ball_spec(), worlds=2)
        simulation.reset(
            qpos=[0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]
        )
        with TemporaryDirectory() as directory:
            video = Path(directory) / "ball.mp4"
            with patch("tinysim.session.encode_mp4", side_effect=fake_encode):
                state = simulation.record(
                    video,
                    steps=10,
                    world=1,
                    fps=60,
                    width=64,
                    height=48,
                )
            trajectory_path = video.with_suffix(".trajectory.json")
            payload = json.loads(trajectory_path.read_text())
            self.assertTrue(video.is_file())
            self.assertEqual(len(payload["qpos"]), 11)
            self.assertEqual(len(payload["qvel"]), 11)
            self.assertEqual(payload["metadata"]["recorded_world"], 1)
            self.assertEqual(payload["metadata"]["worlds"], 2)
            self.assertEqual(encoded["fps"], 60)
            self.assertEqual(len(encoded["frames"]), 1)
            self.assertEqual(len(encoded["frames"][0]), 64 * 48 * 3)
            self.assertAlmostEqual(
                float(state.time[0].item()), 0.02, places=6
            )

    def test_from_compiled_reuses_simulator(self):
        simulator = Simulator.compile(ball_spec(), worlds=2)
        simulation = Simulation.from_compiled(simulator)
        self.assertIs(simulation.simulator, simulator)
        self.assertEqual(simulation.state.qpos.shape, (2, 7))

    def test_record_grid_saves_selected_worlds_and_renders_one_frame(self):
        encoded: dict[str, object] = {}

        def fake_encode(frames, path, *, width, height, fps):
            encoded["frames"] = list(frames)

        simulation = Simulation(ball_spec(), worlds=4)
        simulation.reset(
            qpos=[
                [0.0, 0.0, height, 1.0, 0.0, 0.0, 0.0]
                for height in (0.5, 0.75, 1.0, 1.25)
            ]
        )
        with TemporaryDirectory() as directory:
            video = Path(directory) / "grid.mp4"
            with patch("tinysim.session.encode_mp4", side_effect=fake_encode):
                simulation.record(
                    video,
                    steps=10,
                    worlds=[0, 2, 3],
                    columns=2,
                    width=128,
                    height=96,
                )
            payload = json.loads(
                video.with_suffix(".trajectory.json").read_text()
            )
        self.assertEqual(payload["metadata"]["recorded_worlds"], [0, 2, 3])
        self.assertEqual(payload["metadata"]["grid_columns"], 2)
        self.assertEqual(len(payload["qpos"][0]), 3 * 7)
        self.assertEqual(len(encoded["frames"]), 1)
        self.assertEqual(len(encoded["frames"][0]), 128 * 96 * 3)

    def test_record_grid_rejects_duplicate_worlds(self):
        simulation = Simulation(ball_spec(), worlds=2)
        with self.assertRaisesRegex(ValueError, "unique"):
            simulation.record(
                "ignored.mp4",
                steps=1,
                worlds=[0, 0],
            )

    def test_record_every_decimates_the_saved_trajectory(self):
        simulation = Simulation(ball_spec())
        simulation.reset(
            qpos=[0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]
        )
        with TemporaryDirectory() as directory:
            video = Path(directory) / "decimated.mp4"
            with patch("tinysim.session.encode_mp4"):
                simulation.record(
                    video,
                    steps=10,
                    record_every=5,
                    width=64,
                    height=48,
                )
            payload = json.loads(
                video.with_suffix(".trajectory.json").read_text()
            )
        self.assertEqual(len(payload["qpos"]), 3)
        self.assertAlmostEqual(payload["timestep"], 0.01, places=8)
        self.assertEqual(payload["metadata"]["record_every"], 5)
        self.assertEqual(payload["metadata"]["simulation_steps"], 10)

    def test_automatic_model_renderer_is_state_sensitive(self):
        model = Simulator.compile(ball_spec()).model
        high = render_model(
            model,
            [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            width=64,
            height=48,
        )
        low = render_model(
            model,
            [0.0, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0],
            width=64,
            height=48,
        )
        self.assertEqual(len(high), 64 * 48 * 3)
        self.assertNotEqual(high, low)

    def test_automatic_model_grid_renderer_tiles_worlds(self):
        model = Simulator.compile(ball_spec()).model
        frame = render_model_grid(
            model,
            [
                [0.0, 0.0, height, 1.0, 0.0, 0.0, 0.0]
                for height in (0.5, 0.75, 1.0)
            ],
            width=128,
            height=96,
            columns=2,
        )
        self.assertEqual(len(frame), 128 * 96 * 3)


if __name__ == "__main__":
    unittest.main()
