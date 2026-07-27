"""Compatibility entry point for the comprehensive pendulum benchmark."""

import sys

from .bench_matrix import main


if __name__ == "__main__":
    raise SystemExit(main(["--workloads", "pendulum", *sys.argv[1:]]))
