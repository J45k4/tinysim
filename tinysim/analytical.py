"""Batched analytical systems used to validate TinySim's execution path."""

from collections.abc import Callable
import math

from tinygrad import Tensor, TinyJit

from .integrator import integrate


def _require_world_columns(*tensors: Tensor) -> None:
  shape = tensors[0].shape
  if len(shape) != 2 or shape[1] != 1:
    raise ValueError(f"expected [worlds, 1] tensors, got {shape}")
  if any(tensor.shape != shape for tensor in tensors[1:]):
    raise ValueError(f"state/control shape mismatch: {[tensor.shape for tensor in tensors]}")


def semi_implicit_euler(position: Tensor, velocity: Tensor, acceleration: Tensor, dt: float) -> tuple[Tensor, Tensor]:
  """Integrate velocity first, then position, preserving the input shape."""
  return integrate(position, velocity, acceleration, dt)


def pendulum_acceleration(theta: Tensor, omega: Tensor, torque: Tensor, *, mass: float = 1.0, length: float = 1.0,
                          damping: float = 0.05, gravity: float = 9.81) -> Tensor:
  """Point-mass pendulum acceleration with theta=0 at the downward vertical."""
  _require_world_columns(theta, omega, torque)
  if mass <= 0 or length <= 0 or not all(math.isfinite(value) for value in (mass, length, damping, gravity)):
    raise ValueError("pendulum parameters must be finite and mass/length positive")
  return (torque - damping * omega - mass * gravity * length * theta.sin()) / (mass * length * length)


def pendulum_step(theta: Tensor, omega: Tensor, torque: Tensor, *, mass: float = 1.0, length: float = 1.0,
                  damping: float = 0.05, gravity: float = 9.81, dt: float = 0.01) -> tuple[Tensor, Tensor]:
  """Advance independent point-mass pendulums with semi-implicit Euler."""
  acceleration = pendulum_acceleration(theta, omega, torque, mass=mass, length=length, damping=damping, gravity=gravity)
  return semi_implicit_euler(theta, omega, acceleration, dt)


def spring_acceleration(
    position: Tensor,
    velocity: Tensor,
    force: Tensor,
    *,
    mass: float = 1.0,
    stiffness: float = 10.0,
    damping: float = 0.0,
    rest_position: float = 0.0,
) -> Tensor:
  """Acceleration of independent one-dimensional point-mass springs."""
  _require_world_columns(position, velocity, force)
  parameters = (mass, stiffness, damping, rest_position)
  if mass <= 0 or stiffness < 0 or damping < 0 or not all(
      math.isfinite(value) for value in parameters
  ):
    raise ValueError("spring parameters must be finite with positive mass and nonnegative stiffness/damping")
  return (
      force - stiffness * (position - rest_position) - damping * velocity
  ) / mass


def spring_step(
    position: Tensor,
    velocity: Tensor,
    force: Tensor,
    *,
    mass: float = 1.0,
    stiffness: float = 10.0,
    damping: float = 0.0,
    rest_position: float = 0.0,
    dt: float = 0.01,
) -> tuple[Tensor, Tensor]:
  """Advance independent point-mass springs with semi-implicit Euler."""
  acceleration = spring_acceleration(
      position,
      velocity,
      force,
      mass=mass,
      stiffness=stiffness,
      damping=damping,
      rest_position=rest_position,
  )
  return semi_implicit_euler(position, velocity, acceleration, dt)


def cartpole_acceleration(velocity: Tensor, theta: Tensor, omega: Tensor, force: Tensor, *, cart_mass: float = 1.0,
                          pole_mass: float = 0.1, length: float = 0.5, cart_damping: float = 0.0,
                          pole_damping: float = 0.0, gravity: float = 9.81) -> tuple[Tensor, Tensor]:
  """Cart and point-mass pole accelerations with theta=0 at the unstable upright."""
  _require_world_columns(velocity, theta, omega, force)
  parameters = (cart_mass, pole_mass, length, cart_damping, pole_damping, gravity)
  if cart_mass <= 0 or pole_mass <= 0 or length <= 0 or not all(math.isfinite(value) for value in parameters):
    raise ValueError("cart-pole parameters must be finite and masses/length positive")
  sine, cosine = theta.sin(), theta.cos()
  total_mass = cart_mass + pole_mass
  coupling = pole_mass * length * cosine
  rhs_cart = force - cart_damping * velocity + pole_mass * length * sine * omega.square()
  rhs_pole = pole_mass * gravity * length * sine - pole_damping * omega
  determinant = total_mass * pole_mass * length * length - coupling.square()
  cart_acceleration = (pole_mass * length * length * rhs_cart - coupling * rhs_pole) / determinant
  pole_acceleration = (total_mass * rhs_pole - coupling * rhs_cart) / determinant
  return cart_acceleration, pole_acceleration


def cartpole_step(position: Tensor, velocity: Tensor, theta: Tensor, omega: Tensor, force: Tensor, *,
                  cart_mass: float = 1.0, pole_mass: float = 0.1, length: float = 0.5, cart_damping: float = 0.0,
                  pole_damping: float = 0.0, gravity: float = 9.81,
                  dt: float = 0.01) -> tuple[Tensor, Tensor, Tensor, Tensor]:
  """Advance independent cart-poles with semi-implicit Euler."""
  _require_world_columns(position, velocity, theta, omega, force)
  cart_acceleration, pole_acceleration = cartpole_acceleration(
    velocity, theta, omega, force, cart_mass=cart_mass, pole_mass=pole_mass, length=length,
    cart_damping=cart_damping, pole_damping=pole_damping, gravity=gravity)
  position_next, velocity_next = semi_implicit_euler(position, velocity, cart_acceleration, dt)
  theta_next, omega_next = semi_implicit_euler(theta, omega, pole_acceleration, dt)
  return position_next, velocity_next, theta_next, omega_next


def _realized_step(step: Callable[..., tuple[Tensor, ...]], parameters: dict[str, float]) -> TinyJit:
  def run(*state: Tensor) -> tuple[Tensor, ...]:
    outputs = step(*state, **parameters)
    Tensor.realize(*outputs)
    return outputs
  return TinyJit(run)


def make_jitted_pendulum_step(**parameters: float) -> TinyJit:
  """Build a forward-only JIT with physical parameters frozen."""
  return _realized_step(pendulum_step, parameters)


def make_jitted_cartpole_step(**parameters: float) -> TinyJit:
  """Build a forward-only JIT with physical parameters frozen."""
  return _realized_step(cartpole_step, parameters)
