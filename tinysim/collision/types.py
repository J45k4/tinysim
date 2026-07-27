"""Small tensor containers shared by collision and contact code."""

from dataclasses import dataclass
import math

from tinygrad import Tensor


Scalar = float | int | Tensor


def column(value: Scalar, vector: Tensor) -> Scalar:
    """Adds a vector-component axis to per-item Tensor scalars when needed."""
    return (
        value.unsqueeze(-1)
        if isinstance(value, Tensor) and value.ndim == vector.ndim - 1
        else value
    )


@dataclass(frozen=True)
class ContactGeometry:
    """Fixed-shape narrow-phase output.

    ``distance`` is the signed surface gap: positive means separated, zero
    means touching, and negative means penetrating. ``normal`` always points
    from shape A toward shape B. Points are the corresponding witnesses on A
    and B. The leading dimensions are arbitrary fixed batch dimensions.
    """

    distance: Tensor
    normal: Tensor
    point_a: Tensor
    point_b: Tensor
    active: Tensor

    def __post_init__(self) -> None:
        if self.normal.ndim < 1 or self.normal.shape[-1] != 3:
            raise ValueError("contact normal must end in dimension 3")
        if self.point_a.shape != self.normal.shape or self.point_b.shape != self.normal.shape:
            raise ValueError("contact points and normal must have identical shapes")
        if self.distance.shape != self.normal.shape[:-1] or self.active.shape != self.distance.shape:
            raise ValueError("distance and activity must match contact batch dimensions")

    @property
    def position(self) -> Tensor:
        """Midpoint of the two witness points."""

        return (self.point_a + self.point_b) * 0.5

    @property
    def penetration(self) -> Tensor:
        return (-self.distance).maximum(0.0)


@dataclass(frozen=True)
class ContactParams:
    """Parameters for the differentiable compliant contact law.

    Tensor-valued parameters remain on device and are not read back for
    validation; their caller is responsible for nonnegative physical values.
    """

    stiffness: Scalar = 1_000.0
    damping: Scalar = 10.0
    friction: Scalar = 0.5
    penetration_smoothing: Scalar = 1e-4
    force_smoothing: Scalar = 1e-6
    velocity_smoothing: Scalar = 1e-3

    def __post_init__(self) -> None:
        for name in ("penetration_smoothing", "force_smoothing", "velocity_smoothing"):
            value = getattr(self, name)
            if not isinstance(value, Tensor) and (
                value <= 0 or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be finite and positive")
        for name in ("stiffness", "damping", "friction"):
            value = getattr(self, name)
            if not isinstance(value, Tensor) and (value < 0 or not math.isfinite(value)):
                raise ValueError(f"{name} must be finite and nonnegative")


@dataclass(frozen=True)
class ContactForces:
    """Equal-and-opposite Cartesian forces at contact witness points."""

    force_a: Tensor
    force_b: Tensor
    normal_force: Tensor
    friction_force_a: Tensor
    normal_velocity: Tensor
    tangent_velocity: Tensor
