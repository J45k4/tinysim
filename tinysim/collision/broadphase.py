"""Compile-time helpers for fixed collision candidate layouts."""

from dataclasses import dataclass
from collections.abc import Iterable

from tinygrad import Tensor, dtypes


@dataclass(frozen=True)
class FixedPairs:
    """Immutable ordered shape pairs; no runtime allocation or filtering."""

    indices: tuple[tuple[int, int], ...]

    @property
    def count(self) -> int:
        return len(self.indices)

    def tensor(self, *, device: str | None = None) -> Tensor:
        flat = [index for pair in self.indices for index in pair]
        return Tensor(flat, dtype=dtypes.int, device=device).reshape(self.count, 2)


def fixed_pairs(
    pairs: Iterable[tuple[int, int]], *, geometry_count: int | None = None
) -> FixedPairs:
    """Validates candidate pairs once while compiling a model.

    Pair order is retained because collision normals point from the first shape
    to the second. Reversed duplicates are rejected.
    """

    result: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for a, b in pairs:
        if not isinstance(a, int) or not isinstance(b, int):
            raise ValueError(f"collision pair indices must be integers, got {(a, b)}")
        if a < 0 or b < 0 or a == b:
            raise ValueError(f"invalid collision pair {(a, b)}")
        if geometry_count is not None and (a >= geometry_count or b >= geometry_count):
            raise ValueError(f"collision pair {(a, b)} is outside geometry range")
        key = (min(a, b), max(a, b))
        if key in seen:
            raise ValueError(f"duplicate collision pair {(a, b)}")
        result.append((a, b))
        seen.add(key)
    return FixedPairs(tuple(result))


def all_pairs(geometry_count: int) -> FixedPairs:
    """Builds the static all-pairs layout for small models."""

    if geometry_count < 0:
        raise ValueError("geometry_count must be nonnegative")
    return fixed_pairs(
        ((a, b) for a in range(geometry_count) for b in range(a + 1, geometry_count)),
        geometry_count=geometry_count,
    )
