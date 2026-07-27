from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from tinysim.trajectory import (
    Trajectory,
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


if __name__ == "__main__":
    unittest.main()
