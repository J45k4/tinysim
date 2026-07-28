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
from .manifold import (
    box_box_manifold,
    box_plane_manifold,
    capsule_capsule_manifold,
    capsule_plane_manifold,
    singleton_manifold,
    sphere_capsule_manifold,
    sphere_plane_manifold,
    sphere_sphere_manifold,
)
from .types import (
    ContactForces,
    ContactGeometry,
    ContactManifold,
    ContactParams,
)

__all__ = [
    "ContactForces",
    "ContactGeometry",
    "ContactManifold",
    "ContactParams",
    "FixedPairs",
    "all_pairs",
    "box_box",
    "box_box_manifold",
    "box_plane",
    "box_plane_manifold",
    "capsule_capsule",
    "capsule_capsule_manifold",
    "capsule_plane",
    "capsule_plane_manifold",
    "fixed_pairs",
    "sphere_capsule",
    "sphere_capsule_manifold",
    "sphere_plane",
    "sphere_plane_manifold",
    "sphere_sphere",
    "sphere_sphere_manifold",
    "singleton_manifold",
]
