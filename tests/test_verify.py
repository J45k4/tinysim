from pathlib import Path
from tempfile import TemporaryDirectory
from dataclasses import replace
from hashlib import sha256
import json
import unittest
from unittest.mock import patch

from tinysim.verify import (
    load_verification_report,
    main as verify_main,
    verify_batched_worlds,
    verify_bouncing_ball,
    verify_cartpole,
    verify_free_body,
    verify_imported_articulated,
    verify_pendulum,
    refresh_release_report,
    write_release_report,
)
from tinysim.provenance import project_source_sha256, source_tree_sha256


class TestVerification(unittest.TestCase):
    def assert_release_evidence(self, report, *, steps: int, worlds: int):
        self.assertTrue(report.passed, report)
        self.assertEqual(set(report.checks.values()), {True})
        for artifact in report.artifacts.values():
            self.assertTrue(Path(artifact).is_file(), artifact)
        manifest = json.loads(Path(report.artifacts["manifest"]).read_text())
        self.assertTrue(manifest["passed"])
        self.assertEqual(manifest["environment"]["backend"], "CPU")
        self.assertTrue(manifest["environment"]["dtype"])
        self.assertEqual(len(manifest["environment"]["tinygrad_commit"]), 40)
        self.assertEqual(len(manifest["environment"]["tinysim_source_sha256"]), 64)
        self.assertEqual(
            len(
                manifest["environment"][
                    "tinysim_project_source_sha256"
                ]
            ),
            64,
        )
        self.assertEqual(manifest["configuration"]["steps"], steps)
        self.assertEqual(manifest["configuration"]["worlds"], worlds)
        self.assertAlmostEqual(
            report.metrics["video_playback_seconds"],
            report.metrics["simulated_seconds"],
            delta=0.5 / manifest["configuration"]["fps"] + 1e-12,
        )
        self.assertGreater(
            report.metrics["trajectory_recording_wall_seconds"], 0.0
        )
        self.assertGreater(
            report.metrics["trajectory_recording_realtime_factor"], 0.0
        )
        self.assertLessEqual(report.metrics["tinyjit_calls"], 128)
        self.assertGreater(report.metrics["tinyjit_calls"], 0)
        self.assertEqual(len(report.trajectory_sha256), 64)

    def test_pendulum_release_evidence(self):
        with TemporaryDirectory() as directory:
            report = verify_pendulum(directory, steps=40, width=64, height=48)
            self.assert_release_evidence(report, steps=40, worlds=1)

    def test_cartpole_release_evidence(self):
        with TemporaryDirectory() as directory:
            report = verify_cartpole(directory, steps=40, width=64, height=48)
            self.assert_release_evidence(report, steps=40, worlds=1)
            self.assertTrue(report.checks["upright_equilibrium_is_stationary"])

    def test_bouncing_ball_release_evidence(self):
        with TemporaryDirectory() as directory:
            report = verify_bouncing_ball(
                directory, steps=400, width=64, height=48
            )
            self.assert_release_evidence(report, steps=400, worlds=1)
            self.assertTrue(report.checks["rebound_detected"])
            self.assertTrue(report.checks["settles_near_surface"])
            self.assertTrue(report.checks["effective_restitution_is_bounded"])

    def test_imported_articulated_release_evidence(self):
        with TemporaryDirectory() as directory:
            report = verify_imported_articulated(
                directory, steps=40, width=64, height=48
            )
            self.assert_release_evidence(report, steps=40, worlds=1)
            self.assertTrue(report.checks["joint_limits_respected"])
            manifest = json.loads(Path(report.artifacts["manifest"]).read_text())
            self.assertEqual(manifest["configuration"]["contact"], "none")
            self.assertEqual(len(manifest["configuration"]["mjcf_sha256"]), 64)

    def test_free_body_release_evidence(self):
        with TemporaryDirectory() as directory:
            report = verify_free_body(
                directory, steps=20, width=64, height=48
            )
            self.assert_release_evidence(report, steps=20, worlds=1)
            self.assertTrue(report.checks["quaternion_is_normalized"])
            self.assertTrue(report.checks["angular_momentum_is_conserved"])
            trajectory = json.loads(
                Path(report.artifacts["trajectory"]).read_text()
            )
            self.assertEqual(len(trajectory["qpos"][0]), 7)
            self.assertEqual(len(trajectory["qvel"][0]), 6)

    def test_selected_batched_world_release_evidence(self):
        with TemporaryDirectory() as directory:
            report = verify_batched_worlds(
                directory, steps=40, width=96, height=64
            )
            self.assert_release_evidence(report, steps=40, worlds=4)
            self.assertTrue(report.checks["duplicate_worlds_remain_identical"])
            self.assertTrue(report.checks["mirrored_worlds_remain_mirrored"])

    def test_release_report_links_scenario_evidence(self):
        with TemporaryDirectory() as directory:
            report = verify_pendulum(
                Path(directory) / "pendulum",
                steps=12,
                width=64,
                height=48,
            )
            path = write_release_report(directory, [report])
            release = json.loads(path.read_text(encoding="utf-8"))
            self.assertFalse(release["passed"])
            self.assertFalse(release["release_ready"])
            self.assertEqual(release["scenario_count"], 1)
            self.assertIn("performance", release["evidence"]["links"])
            self.assertIn(
                "mujoco_reference", release["evidence"]["links"]
            )
            self.assertIn(
                "critic_decision", release["evidence"]["links"]
            )
            self.assertFalse(release["release_gates"]["critic_accept"])
            self.assertEqual(
                release["scenarios"][0]["trajectory_sha256"],
                report.trajectory_sha256,
            )
            with self.assertRaises(TypeError):
                write_release_report(
                    directory,
                    [report],
                    supplemental_evidence={
                        "critic_decision": {"status": "available"}
                    },
                )

    def test_release_gates_reject_selective_benchmark_and_gpu_pendulum(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            report = verify_pendulum(
                root / "pendulum",
                steps=12,
                width=64,
                height=48,
            )
            project = Path(__file__).parents[1]
            benchmark = {
                "environment": {
                    "device": "CPU",
                    "tinysim_source_sha256": report.environment[
                        "tinysim_source_sha256"
                    ],
                    "project_source_sha256": project_source_sha256(project),
                    "benchmark_source_sha256": source_tree_sha256(
                        project / "benchmarks"
                    ),
                },
                "protocol": {
                    "workloads_requested": ["pendulum"],
                    "batches_requested": [1],
                },
                "cases": [{"workload": "pendulum", "worlds": 1}],
                "failures": [],
            }
            benchmark_path = root / "benchmarks" / "cpu-quick.json"
            benchmark_path.parent.mkdir()
            benchmark_path.write_text(json.dumps(benchmark), encoding="utf-8")
            video = root / "fake.mp4"
            video.write_bytes(b"not a valid accelerator recording")
            fake_gpu_pendulum = replace(
                report,
                artifacts={**report.artifacts, "video": str(video)},
                environment={**report.environment, "backend": "NV"},
            )
            release_path = write_release_report(
                root / "verify", [fake_gpu_pendulum]
            )
            release = json.loads(release_path.read_text())
            self.assertFalse(release["release_gates"]["performance"])
            self.assertFalse(
                release["release_gates"]["accelerator_recording"]
            )
            self.assertFalse(release["release_gates"]["scenario_checks"])

    def test_manager_review_is_bound_to_accelerator_video(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            report = verify_batched_worlds(
                root / "batched",
                steps=12,
                width=64,
                height=48,
            )
            video = root / "accelerator.mp4"
            video.write_bytes(b"accelerator video")
            golden = root / "golden" / "pendulum.mp4"
            golden.parent.mkdir()
            golden.write_bytes(b"golden video")
            accelerator_report = replace(
                report,
                artifacts={**report.artifacts, "video": str(video)},
                environment={**report.environment, "backend": "CUDA"},
            )
            project = Path(__file__).parents[1]
            review_path = root / "review" / "manager-video-review.json"
            review_path.parent.mkdir()
            review = {
                "decision": "ACCEPT",
                "signed_by": "manager",
                "tinysim_source_sha256": report.environment[
                    "tinysim_source_sha256"
                ],
                "project_source_sha256": project_source_sha256(project),
                "golden_video_sha256": sha256(golden.read_bytes()).hexdigest(),
                "accelerator_video_sha256": "wrong",
            }
            review_path.write_text(json.dumps(review), encoding="utf-8")
            rejected = json.loads(
                write_release_report(
                    root / "verify", [accelerator_report]
                ).read_text()
            )
            self.assertTrue(
                rejected["release_gates"]["accelerator_recording"]
            )
            self.assertFalse(
                rejected["release_gates"]["manager_video_review"]
            )
            review["accelerator_video_sha256"] = sha256(
                video.read_bytes()
            ).hexdigest()
            review_path.write_text(json.dumps(review), encoding="utf-8")
            accepted = json.loads(
                write_release_report(
                    root / "verify", [accelerator_report]
                ).read_text()
            )
            self.assertTrue(
                accepted["release_gates"]["manager_video_review"]
            )
            self.assertEqual(
                accepted["evidence"]["accelerator_video"]["sha256"],
                review["accelerator_video_sha256"],
            )

    def test_release_refresh_does_not_reencode_signed_video(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            report = verify_batched_worlds(
                root / "verify" / "batched-worlds",
                steps=12,
                width=64,
                height=48,
            )
            video = root / "accelerator.mp4"
            video.write_bytes(b"signed accelerator video")
            manifest_path = Path(report.artifacts["manifest"])
            manifest = json.loads(manifest_path.read_text())
            manifest["artifacts"]["video"] = str(video)
            manifest["environment"]["backend"] = "CUDA"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            golden = root / "golden" / "pendulum.mp4"
            golden.parent.mkdir()
            golden.write_bytes(b"signed golden video")
            review = {
                "decision": "ACCEPT",
                "signed_by": "manager",
                "tinysim_source_sha256": report.environment[
                    "tinysim_source_sha256"
                ],
                "project_source_sha256": project_source_sha256(
                    Path(__file__).parents[1]
                ),
                "golden_video_sha256": sha256(golden.read_bytes()).hexdigest(),
                "accelerator_video_sha256": sha256(
                    video.read_bytes()
                ).hexdigest(),
            }
            review_path = root / "review" / "manager-video-review.json"
            review_path.parent.mkdir()
            review_path.write_text(json.dumps(review), encoding="utf-8")
            original_video = video.read_bytes()
            with patch(
                "tinysim.verify.encode_mp4",
                side_effect=AssertionError("refresh must not encode"),
            ):
                release_path = refresh_release_report(root / "verify")
            release = json.loads(release_path.read_text())
            self.assertTrue(
                release["release_gates"]["manager_video_review"]
            )
            self.assertEqual(video.read_bytes(), original_video)

    def test_manifest_loader_rejects_inconsistent_passed_summary(self):
        with TemporaryDirectory() as directory:
            report = verify_pendulum(
                directory,
                steps=12,
                width=64,
                height=48,
            )
            manifest_path = Path(report.artifacts["manifest"])
            manifest = json.loads(manifest_path.read_text())
            manifest["checks"]["finite_state"] = False
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid verification"):
                load_verification_report(manifest_path)

    def test_release_refresh_rejects_scenario_argument(self):
        with patch(
            "sys.argv",
            [
                "tinysim.verify",
                "--refresh-release",
                "--scenario",
                "pendulum",
            ],
        ):
            with self.assertRaises(SystemExit) as raised:
                verify_main()
        self.assertEqual(raised.exception.code, 2)

    def test_manifest_and_trajectory_are_reproducible(self):
        with TemporaryDirectory() as directory:
            first = verify_cartpole(directory, steps=12, width=64, height=48)
            trajectory = Path(first.artifacts["trajectory"]).read_bytes()
            second = verify_cartpole(directory, steps=12, width=64, height=48)
            self.assertEqual(trajectory, Path(second.artifacts["trajectory"]).read_bytes())
            self.assertEqual(first.trajectory_sha256, second.trajectory_sha256)
            self.assertEqual(first.checks, second.checks)
            self.assertEqual(first.configuration, second.configuration)
            self.assertEqual(first.environment, second.environment)

    def test_mp4_uses_ffmpeg_adapter_at_simulated_duration(self):
        encoded: dict[str, object] = {}

        def fake_encode(frames, path, *, width, height, fps):
            encoded.update(
                frames=list(frames), width=width, height=height, fps=fps
            )
            Path(path).write_bytes(b"mock mp4")

        with TemporaryDirectory() as directory:
            video = Path(directory) / "cartpole.mp4"
            with patch("tinysim.verify.encode_mp4", side_effect=fake_encode):
                report = verify_cartpole(
                    directory,
                    steps=10,
                    timestep=0.01,
                    record=video,
                    width=64,
                    height=48,
                    fps=20,
                )
                repeated = verify_cartpole(
                    directory,
                    steps=10,
                    timestep=0.01,
                    record=video,
                    width=64,
                    height=48,
                    fps=20,
                )
            self.assertTrue(report.passed, report)
            self.assertEqual(len(encoded["frames"]), 2)
            self.assertEqual(encoded["fps"], 20)
            self.assertAlmostEqual(report.metrics["video_playback_seconds"], 0.1)
            self.assertEqual(report.trajectory_sha256, repeated.trajectory_sha256)
            self.assertEqual(report.checks, repeated.checks)


if __name__ == "__main__":
    unittest.main()
