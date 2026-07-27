"""Primitive collision queries with fixed-shape tensor outputs."""

from .broadphase import FixedPairs, all_pairs, fixed_pairs
from .primitive import (
    box_box,
    box_plane,
    capsule_capsule,
    capsule_plane,
    sphere_capsule,
    sphere_plane,
    sphere_sphere,
)
from .types import ContactForces, ContactGeometry, ContactParams

__all__ = [
    "ContactForces",
    "ContactGeometry",
    "ContactParams",
    "FixedPairs",
    "all_pairs",
    "box_box",
    "box_plane",
    "capsule_capsule",
    "capsule_plane",
    "fixed_pairs",
    "sphere_capsule",
    "sphere_plane",
    "sphere_sphere",
]
