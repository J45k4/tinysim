import json
from pathlib import Path
import tempfile
import unittest

from benchmarks.bench_matrix import (
    DEFAULT_BATCHES,
    WORKLOAD_NAMES,
    _mjx_baseline_measurement,
    add_scaling_efficiency,
    make_workload,
    run_matrix,
    write_report,
)
from tinysim.provenance import source_tree_sha256


class TestBenchmarkProtocol(unittest.TestCase):
    def test_mjx_baseline_records_its_own_environment(self):
        class FakeWorkload:
            def mjx_baseline(self, worlds, steps):
                return 0.5, {
                    "device": "CUDA:0",
                    "platform": "gpu",
                    "dtype": "float32",
                    "jax_version": "test",
                    "mujoco_version": "test",
                }

        result = _mjx_baseline_measurement(FakeWorkload(), 16, 2)
        self.assertEqual(result["status"], "available")
        self.assertEqual(result["world_steps_per_s"], 64.0)
        self.assertEqual(result["environment"]["platform"], "gpu")

    def test_plan_workloads_and_batches_are_present(self):
        self.assertEqual(DEFAULT_BATCHES, (1, 16, 64, 256, 1024, 4096, 16384))
        self.assertEqual(
            WORKLOAD_NAMES,
            (
                "pendulum",
                "cartpole",
                "quadruped",
                "humanoid",
                "bouncing_contact",
                "locomotion_policy",
            ),
        )
        for name in WORKLOAD_NAMES:
            workload = make_workload(name)
            self.assertEqual(workload.name, name)
            self.assertIn("solver_iterations", workload.settings())
            self.assertIn("contact_capacity", workload.settings())

    def test_quick_case_reports_required_measurements(self):
        report = run_matrix(
            ["pendulum"],
            [1],
            warm_steps=1,
            backward_steps=1,
            subsystem_steps=1,
            baseline_steps=1,
        )
        self.assertFalse(report["failures"])
        case = report["cases"][0]
        self.assertGreater(case["first"]["wall_s"], 0.0)
        self.assertGreater(case["capture"]["wall_s"], 0.0)
        self.assertGreater(case["warm"]["world_steps_per_s"], 0.0)
        self.assertGreaterEqual(case["warm"]["mean_kernels"], 1.0)
        self.assertIsNotNone(case["captured_calls"])
        self.assertGreater(case["backward"]["world_steps_per_s"], 0.0)
        self.assertIn("dynamics", case["subsystems"])
        self.assertEqual(case["baselines"]["python"]["status"], "available")
        self.assertEqual(case["baselines"]["mjx_jax"]["status"], "unavailable")
        self.assertEqual(
            report["environment"]["device_utilization"]["status"], "unavailable"
        )
        self.assertEqual(
            len(report["environment"]["tinysim_source_sha256"]), 64
        )
        self.assertEqual(
            len(report["environment"]["benchmark_source_sha256"]), 64
        )

    def test_benchmark_revision_covers_harness_source(self):
        from pathlib import Path

        directory = Path(__file__).parents[1] / "benchmarks"
        original = source_tree_sha256(directory)
        with tempfile.TemporaryDirectory() as temporary:
            copy = Path(temporary)
            (copy / "bench.py").write_bytes(
                (directory / "bench_matrix.py").read_bytes()
            )
            before = source_tree_sha256(copy)
            (copy / "bench.py").write_bytes(
                (copy / "bench.py").read_bytes() + b"\n# changed\n"
            )
            self.assertNotEqual(before, source_tree_sha256(copy))
        self.assertEqual(len(original), 64)

    def test_scaling_efficiency_uses_batch_one(self):
        cases = [
            {
                "workload": "x",
                "worlds": 1,
                "warm": {"world_steps_per_s": 10.0},
            },
            {
                "workload": "x",
                "worlds": 16,
                "warm": {"world_steps_per_s": 80.0},
            },
        ]
        add_scaling_efficiency(cases)
        self.assertEqual(cases[0]["warm"]["scaling_efficiency_vs_batch1"], 1.0)
        self.assertEqual(cases[1]["warm"]["scaling_efficiency_vs_batch1"], 0.5)

    def test_report_writer_emits_json_and_markdown(self):
        report = run_matrix(
            ["pendulum"],
            [1],
            warm_steps=1,
            backward_steps=0,
            subsystem_steps=0,
            baseline_steps=0,
        )
        with tempfile.TemporaryDirectory() as directory:
            json_path, markdown_path = write_report(
                report, Path(directory) / "result.json"
            )
            loaded = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(loaded["schema_version"], 1)
            self.assertIn("TinySim benchmark report", markdown_path.read_text())


if __name__ == "__main__":
    unittest.main()
