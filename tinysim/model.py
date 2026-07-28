"""Direct Python authoring structures for the supported TinySim subset."""

from dataclasses import dataclass, field
from os import PathLike
from typing import Literal


JointKind = Literal["fixed", "hinge", "slide", "ball", "free"]
ActuatorKind = Literal["motor", "position", "velocity"]
GeomKind = Literal["plane", "sphere", "capsule", "box"]
ContactMode = Literal["none", "smooth", "constraint"]
COLLISION_MANIFOLD_CAPACITY = {
    ("sphere", "plane"): 1,
    ("sphere", "sphere"): 1,
    ("sphere", "capsule"): 1,
    ("capsule", "plane"): 2,
    ("capsule", "capsule"): 2,
    ("box", "box"): 4,
    ("box", "plane"): 4,
}
SUPPORTED_COLLISION_PAIRS = frozenset(COLLISION_MANIFOLD_CAPACITY)


class UnsupportedModelError(ValueError):
    """Raised when a model is valid in principle but outside TinySim's subset."""


@dataclass(frozen=True)
class BodySpec:
    name: str
    parent: int = -1
    mass: float = 1.0
    inertia: tuple[float, float, float] = (1.0, 1.0, 1.0)
    com: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class JointSpec:
    name: str
    body: int
    kind: JointKind = "hinge"
    axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    damping: float = 0.0
    limit: tuple[float, float] | None = None


@dataclass(frozen=True)
class ActuatorSpec:
    name: str
    joint: int | str
    kind: ActuatorKind = "motor"
    gain: float = 1.0
    damping: float = 0.0
    control_range: tuple[float, float] | None = None
    force_range: tuple[float, float] | None = None


@dataclass(frozen=True)
class GeomSpec:
    name: str | None
    body: int
    kind: GeomKind
    size: tuple[float, ...]
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    friction: float = 0.5


@dataclass(frozen=True)
class ContactSpec:
    """Compile-time contact semantics; iteration count is part of the graph."""

    mode: ContactMode = "none"
    margin: float = 0.0
    stiffness: float = 5_000.0
    damping: float = 100.0
    friction: float = 1.0
    penetration_smoothing: float = 1e-4
    force_smoothing: float = 1e-6
    velocity_smoothing: float = 1e-3
    solver_iterations: int = 8
    stabilization: float = 0.2
    regularization: float = 1e-6


@dataclass(frozen=True)
class ModelSpec:
    bodies: tuple[BodySpec, ...] | list[BodySpec]
    joints: tuple[JointSpec, ...] | list[JointSpec]
    actuators: tuple[ActuatorSpec, ...] | list[ActuatorSpec] = ()
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    timestep: float = 0.002
    name: str = "model"
    geoms: tuple[GeomSpec, ...] | list[GeomSpec] = ()
    collision_pairs: tuple[tuple[int | str, int | str], ...] | list[tuple[int | str, int | str]] = ()
    collision_exclusions: tuple[tuple[int | str, int | str], ...] | list[tuple[int | str, int | str]] = ()
    contact: ContactSpec = field(default_factory=ContactSpec)

    def __post_init__(self) -> None:
        object.__setattr__(self, "bodies", tuple(self.bodies))
        object.__setattr__(self, "joints", tuple(self.joints))
        object.__setattr__(self, "actuators", tuple(self.actuators))
        object.__setattr__(self, "geoms", tuple(self.geoms))
        object.__setattr__(self, "collision_pairs", tuple(self.collision_pairs))
        object.__setattr__(
            self, "collision_exclusions", tuple(self.collision_exclusions)
        )

    @classmethod
    def from_mjcf(cls, source: str | PathLike[str]) -> "ModelSpec":
        """Imports the supported MJCF subset without adding a runtime dependency."""

        from .importers.mjcf import load_mjcf

        return load_mjcf(source).spec
