#!/usr/bin/env python3
"""Run the optional canonical-MuJoCo suite and record source-bound evidence."""

from dataclasses import asdict
import io
import json
from pathlib import Path
import time
import unittest

from tinygrad import Device

from tinysim.provenance import project_source_sha256, source_tree_sha256
from tinysim.reference import mujoco_status


def main() -> None:
    if Device.DEFAULT != "CPU":
        raise RuntimeError(
            f"MuJoCo evidence requires DEV=CPU, got {Device.DEFAULT}"
        )
    project = Path(__file__).resolve().parents[1]
    destination = project / "artifacts" / "reference" / "mujoco.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    status = mujoco_status()
    payload: dict[str, object] = {
        "schema_version": 1,
        "tinysim_source_sha256": source_tree_sha256(project / "tinysim"),
        "project_source_sha256": project_source_sha256(project),
        "validation_source_sha256": source_tree_sha256(project / "tests"),
        "device": Device.DEFAULT,
        "backend": asdict(status),
        "suite": "tests.test_mujoco_reference",
    }
    if not status.available:
        payload.update(
            passed=False,
            status="unavailable",
            reason=status.reason,
        )
    else:
        suite = unittest.defaultTestLoader.loadTestsFromName(
            "tests.test_mujoco_reference"
        )
        stream = io.StringIO()
        started = time.perf_counter()
        result = unittest.TextTestRunner(
            stream=stream, verbosity=2
        ).run(suite)
        payload.update(
            passed=result.wasSuccessful() and result.testsRun > 0,
            status="available",
            tests_run=result.testsRun,
            failures=len(result.failures),
            errors=len(result.errors),
            skipped=len(result.skipped),
            wall_seconds=time.perf_counter() - started,
            coverage=[
                "float64 contact-free intermediates and integrated state",
                "float32 contact-free one-step tolerance",
                "sphere-plane distance, normal, point, and normal Jacobian",
                "ball/free generalized-basis rejection boundary",
            ],
            test_output=stream.getvalue(),
        )
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {destination}")
    if status.available and not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
