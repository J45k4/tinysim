from array import array
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
import unittest

from tinysim.trajectory import (
    Trajectory,
    TrajectoryReader,
    TrajectoryWriter,
    export_trajectory_json,
    iter_playback_indices,
    load_trajectory,
    playback_indices,
    save_trajectory,
)


class TestTrajectory(unittest.TestCase):
    def test_round_trip(self):
        trajectory = Trajectory(
            scenario="pendulum",
            timestep=0.01,
            qpos=[[0.2], [0.199]],
            qvel=[[0.0], [-0.1]],
            controls=[[0.0]],
            metadata={"seed": 0},
        )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "trajectory.json"
            save_trajectory(trajectory, path)
            self.assertEqual(load_trajectory(path), trajectory)

    def test_rejects_nonfinite(self):
        trajectory = Trajectory(
            scenario="bad",
            timestep=0.01,
            qpos=[[float("nan")]],
            qvel=[[0.0]],
        )
        with self.assertRaisesRegex(ValueError, "non-finite"):
            trajectory.validate()

    def test_free_joint_round_trip_allows_distinct_qpos_qvel_widths(self):
        trajectory = Trajectory(
            scenario="free_body",
            timestep=0.01,
            qpos=[
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0, 0.999, 0.0, 0.045, 0.0],
            ],
            qvel=[
                [1.0, 0.0, 0.0, 0.0, 0.5, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.5, 0.0],
            ],
        )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "free.json"
            save_trajectory(trajectory, path)
            self.assertEqual(load_trajectory(path), trajectory)

    def test_rejects_unstable_stream_widths(self):
        with self.assertRaisesRegex(ValueError, "stable dimensions"):
            Trajectory(
                scenario="bad",
                timestep=0.01,
                qpos=[[1.0], [1.0, 2.0]],
                qvel=[[1.0], [2.0]],
            ).validate()

    def test_playback_indices_preserve_simulated_duration(self):
        indices = playback_indices(101, timestep=0.01, fps=60)
        self.assertEqual(len(indices), 60)
        self.assertEqual(indices[0], 0)
        self.assertTrue(all(left <= right for left, right in zip(indices, indices[1:])))
        self.assertTrue(all(0 <= index < 101 for index in indices))
        self.assertAlmostEqual(len(indices) / 60, (101 - 1) * 0.01)

    def test_playback_indices_validate_inputs(self):
        with self.assertRaisesRegex(ValueError, "at least two"):
            playback_indices(1, timestep=0.01, fps=60)
        with self.assertRaisesRegex(ValueError, "timestep"):
            playback_indices(2, timestep=0.0, fps=60)
        with self.assertRaisesRegex(ValueError, "fps"):
            playback_indices(2, timestep=0.01, fps=0)

    def test_stream_round_trip_is_fixed_width_and_deterministic(self):
        with TemporaryDirectory() as directory:
            first = Path(directory) / "first.tstraj"
            second = Path(directory) / "second.tstraj"
            for path in (first, second):
                with TrajectoryWriter(
                    path,
                    scenario="free_body",
                    timestep=0.01,
                    qpos_width=7,
                    qvel_width=6,
                    control_width=1,
                    metadata={"recorded_worlds": [2]},
                ) as writer:
                    writer.append(
                        array(
                            "f",
                            [
                                0.0,
                                0.0,
                                1.0,
                                1.0,
                                0.0,
                                0.0,
                                0.0,
                                1.0,
                                2.0,
                                3.0,
                                4.0,
                                5.0,
                                6.0,
                                0.5,
                            ],
                        )
                    )
            self.assertEqual(first.read_bytes(), second.read_bytes())
            with TrajectoryReader(first) as trajectory:
                self.assertEqual(trajectory.frame_count, 1)
                self.assertEqual(trajectory.dtype, "float32")
                self.assertEqual(trajectory.scenario, "free_body")
                self.assertEqual(
                    trajectory.metadata["recorded_worlds"], [2]
                )
                frame = trajectory.read_frame(0)
                self.assertEqual(list(frame.qpos), [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0])
                self.assertEqual(list(frame.qvel), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
                self.assertEqual(list(frame.controls), [0.5])

    def test_stream_supports_float64_and_empty_control(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "double.tstraj"
            with TrajectoryWriter(
                path,
                scenario="pendulum",
                timestep=0.001,
                qpos_width=1,
                qvel_width=1,
                control_width=0,
                dtype="float64",
            ) as writer:
                writer.append(array("d", [0.2, -0.3]))
            with TrajectoryReader(path) as trajectory:
                frame = trajectory.read_frame(0)
                self.assertEqual(list(frame.qpos), [0.2])
                self.assertEqual(list(frame.qvel), [-0.3])
                self.assertEqual(list(frame.controls), [])

    def test_stream_rejects_bad_frame_and_nonfinite_value(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "bad.tstraj"
            writer = TrajectoryWriter(
                path,
                scenario="pendulum",
                timestep=0.01,
                qpos_width=1,
                qvel_width=1,
                control_width=0,
            )
            with self.assertRaisesRegex(ValueError, "expected"):
                writer.append(array("f", [0.2]))
            with self.assertRaisesRegex(ValueError, "non-finite"):
                writer.append(array("f", [float("nan"), 0.0]))
            writer.abort()
            self.assertFalse(path.exists())

    def test_stream_aborts_partial_file_after_exception(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "partial.tstraj"
            with self.assertRaisesRegex(RuntimeError, "stop"):
                with TrajectoryWriter(
                    path,
                    scenario="pendulum",
                    timestep=0.01,
                    qpos_width=1,
                    qvel_width=1,
                    control_width=0,
                ) as writer:
                    writer.append(array("f", [0.2, 0.0]))
                    raise RuntimeError("stop")
            self.assertFalse(path.exists())
            self.assertEqual(list(Path(directory).glob("*.partial")), [])

    def test_stream_rejects_wrong_length_and_nonfinite_file(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "corrupt.tstraj"
            with TrajectoryWriter(
                path,
                scenario="pendulum",
                timestep=0.01,
                qpos_width=1,
                qvel_width=1,
                control_width=0,
            ) as writer:
                writer.append(array("f", [0.2, 0.0]))
            path.write_bytes(path.read_bytes() + b"x")
            with self.assertRaisesRegex(ValueError, "expected"):
                TrajectoryReader(path)

            finite = Path(directory) / "nonfinite.tstraj"
            with TrajectoryWriter(
                finite,
                scenario="pendulum",
                timestep=0.01,
                qpos_width=1,
                qvel_width=1,
                control_width=0,
            ) as writer:
                writer.append(array("f", [0.2, 0.0]))
            with finite.open("r+b") as stream:
                stream.seek(-4, 2)
                stream.write(struct.pack("<f", float("inf")))
            with TrajectoryReader(finite) as trajectory:
                with self.assertRaisesRegex(ValueError, "non-finite"):
                    trajectory.read_frame(0)

    def test_playback_index_iterator_is_lazy_and_matches_wrapper(self):
        iterator = iter_playback_indices(101, timestep=0.01, fps=60)
        self.assertFalse(isinstance(iterator, list))
        self.assertEqual(
            list(iterator),
            playback_indices(101, timestep=0.01, fps=60),
        )

    def test_streaming_json_export_uses_portable_schema(self):
        with TemporaryDirectory() as directory:
            stream = Path(directory) / "pendulum.tstraj"
            exported = Path(directory) / "pendulum.json"
            with TrajectoryWriter(
                stream,
                scenario="pendulum",
                timestep=0.01,
                qpos_width=1,
                qvel_width=1,
                control_width=1,
                metadata={"seed": 3},
            ) as writer:
                writer.append(array("f", [0.2, 0.0, 0.1]))
                writer.append(array("f", [0.19, -0.1, 0.1]))
            export_trajectory_json(stream, exported)
            trajectory = load_trajectory(exported)
        self.assertEqual(trajectory.scenario, "pendulum")
        self.assertAlmostEqual(trajectory.timestep, 0.01)
        self.assertEqual(trajectory.metadata, {"seed": 3})
        self.assertTrue(
            all(
                abs(row[0] - 0.1) < 1e-6
                for row in trajectory.controls
            )
        )
        self.assertAlmostEqual(trajectory.qpos[1][0], 0.19, places=6)


if __name__ == "__main__":
    unittest.main()
