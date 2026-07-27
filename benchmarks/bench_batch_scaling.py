"""Convenience entry point for forward batch scaling."""

import sys

from .bench_matrix import main


if __name__ == "__main__":
    raise SystemExit(
        main(
            [
                "--backward-steps",
                "0",
                "--subsystem-steps",
                "0",
                *sys.argv[1:],
            ]
        )
    )
