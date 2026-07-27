#!/usr/bin/env python3
"""Run the clean CPU suite and record source-bound release evidence."""

import io
import json
from pathlib import Path
import time
import unittest

from tinygrad import Device

from tinysim.provenance import project_source_sha256, source_tree_sha256


def main() -> None:
    if Device.DEFAULT != "CPU":
        raise RuntimeError(
            f"CPU validation requires DEV=CPU, got {Device.DEFAULT}"
        )
    project = Path(__file__).resolve().parents[1]
    suite = unittest.defaultTestLoader.discover(str(project / "tests"))
    stream = io.StringIO()
    started = time.perf_counter()
    result = unittest.TextTestRunner(stream=stream, verbosity=1).run(suite)
    destination = project / "artifacts" / "validation" / "cpu-tests.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "passed": result.wasSuccessful() and result.testsRun > 0,
        "tinysim_source_sha256": source_tree_sha256(project / "tinysim"),
        "project_source_sha256": project_source_sha256(project),
        "validation_source_sha256": source_tree_sha256(project / "tests"),
        "device": Device.DEFAULT,
        "suite": "unittest discover -s tests",
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "wall_seconds": time.perf_counter() - started,
        "test_output": stream.getvalue(),
    }
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(stream.getvalue(), end="")
    print(f"wrote {destination}")
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
