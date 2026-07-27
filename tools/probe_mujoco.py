#!/usr/bin/env python3
"""Print machine-readable status for TinySim's optional MuJoCo oracle."""

from dataclasses import asdict
import json

from tinysim.reference import mujoco_status


def main() -> None:
    print(json.dumps(asdict(mujoco_status()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
