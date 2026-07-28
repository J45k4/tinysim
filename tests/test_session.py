import math
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
from tinysim.render import (
    Camera3D,
    OrbitCamera,
    render_model,
    render_model_grid,
)
from tinysim.trajectory import TrajectoryReader


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
            trajectory_path = video.with_suffix(".trajectory.tstraj")
            with TrajectoryReader(trajectory_path) as trajectory:
                self.assertEqual(trajectory.frame_count, 11)
                self.assertEqual(trajectory.qpos_width, 7)
                self.assertEqual(trajectory.qvel_width, 6)
                self.assertEqual(trajectory.metadata["recorded_world"], 1)
                self.assertEqual(trajectory.metadata["worlds"], 2)
            self.assertTrue(video.is_file())
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
            with TrajectoryReader(
                video.with_suffix(".trajectory.tstraj")
            ) as trajectory:
                self.assertEqual(
                    trajectory.metadata["recorded_worlds"], [0, 2, 3]
                )
                self.assertEqual(trajectory.metadata["grid_columns"], 2)
                self.assertEqual(trajectory.qpos_width, 3 * 7)
                qpos = trajectory.read_qpos(0)
                self.assertEqual(
                    [qpos[offset + 2] for offset in (0, 7, 14)],
                    [0.5, 1.0, 1.25],
                )
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
            with TrajectoryReader(
                video.with_suffix(".trajectory.tstraj")
            ) as trajectory:
                self.assertEqual(trajectory.frame_count, 3)
                self.assertAlmostEqual(trajectory.timestep, 0.01, places=8)
                self.assertEqual(trajectory.metadata["record_every"], 5)
                self.assertEqual(trajectory.metadata["simulation_steps"], 10)

    def test_recording_does_not_materialize_python_state_lists(self):
        simulation = Simulation(ball_spec(), worlds=2)
        simulation.reset(
            qpos=[0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]
        )
        with TemporaryDirectory() as directory:
            video = Path(directory) / "ball.mp4"
            with (
                patch.object(
                    Tensor,
                    "tolist",
                    side_effect=AssertionError("recording read a host list"),
                ),
                patch("tinysim.session.encode_mp4"),
            ):
                simulation.record(video, steps=2, width=64, height=48)

    def test_recording_performs_one_host_transfer_per_capture(self):
        simulation = Simulation(ball_spec(), worlds=2)
        simulation.reset(
            qpos=[0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]
        )
        simulation.run(steps=3)
        original = Tensor.data
        transfers = 0

        def counted(tensor):
            nonlocal transfers
            transfers += 1
            return original(tensor)

        with TemporaryDirectory() as directory:
            video = Path(directory) / "ball.mp4"
            with (
                patch.object(Tensor, "data", new=counted),
                patch("tinysim.session.encode_mp4"),
            ):
                simulation.record(
                    video,
                    steps=4,
                    record_every=2,
                    width=64,
                    height=48,
                )
        self.assertEqual(transfers, 3)

    def test_recording_does_not_change_final_state(self):
        plain = Simulation(ball_spec(), worlds=2)
        recorded = Simulation(ball_spec(), worlds=2)
        qpos = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]
        plain.reset(qpos=qpos)
        recorded.reset(qpos=qpos)
        expected = plain.run(steps=4)
        with TemporaryDirectory() as directory:
            with patch("tinysim.session.encode_mp4"):
                actual = recorded.record(
                    Path(directory) / "ball.mp4",
                    steps=4,
                    width=64,
                    height=48,
                )
        self.assertEqual(expected.qpos.tolist(), actual.qpos.tolist())
        self.assertEqual(expected.qvel.tolist(), actual.qvel.tolist())
        self.assertEqual(expected.time.tolist(), actual.time.tolist())

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

    def test_model_renderer_projects_world_y_with_3d_camera(self):
        model = Simulator.compile(ball_spec()).model
        qpos = [0.2, 0.4, 1.0, 1.0, 0.0, 0.0, 0.0]
        first = render_model(
            model,
            qpos,
            width=96,
            height=64,
            camera=Camera3D(azimuth=0.0, elevation=0.25),
        )
        quarter_turn = render_model(
            model,
            qpos,
            width=96,
            height=64,
            camera=Camera3D(azimuth=math.pi / 2.0, elevation=0.25),
        )
        self.assertNotEqual(first, quarter_turn)

    def test_recording_accepts_an_orbit_camera_path(self):
        encoded: list[bytes] = []

        def fake_encode(frames, path, *, width, height, fps):
            encoded.extend(frames)

        simulation = Simulation(ball_spec())
        simulation.reset(
            qpos=[0.2, 0.4, 1.0, 1.0, 0.0, 0.0, 0.0]
        )
        with TemporaryDirectory() as directory:
            with patch(
                "tinysim.session.encode_mp4",
                side_effect=fake_encode,
            ):
                simulation.record(
                    Path(directory) / "orbit.mp4",
                    steps=100,
                    record_every=10,
                    fps=50,
                    width=96,
                    height=64,
                    camera=OrbitCamera(elevation=0.25),
                )
        self.assertGreater(len(encoded), 2)
        self.assertNotEqual(encoded[0], encoded[len(encoded) // 2])

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
