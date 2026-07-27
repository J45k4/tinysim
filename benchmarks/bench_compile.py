"""Convenience entry point for first/capture/warm compile profiling."""

import sys

from .bench_matrix import main


if __name__ == "__main__":
    raise SystemExit(
        main(
            [
                "--worlds",
                "1",
                "--warm-steps",
                "1",
                "--backward-steps",
                "0",
                "--subsystem-steps",
                "0",
                "--baseline-steps",
                "0",
                *sys.argv[1:],
            ]
        )
    )
