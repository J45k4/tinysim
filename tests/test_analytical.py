import math
import unittest

from tinygrad import Tensor, dtypes

from tinysim.analytical import (cartpole_acceleration, cartpole_step, make_jitted_cartpole_step,
                                make_jitted_pendulum_step, pendulum_acceleration, pendulum_step,
                                semi_implicit_euler, spring_acceleration, spring_step)


def scalar(tensor: Tensor) -> float:
  return float(tensor.item())


class TestIntegrator(unittest.TestCase):
  def test_semi_implicit_updates_velocity_first(self):
    position, velocity = semi_implicit_euler(Tensor([[1.0]]), Tensor([[2.0]]), Tensor([[3.0]]), 0.1)
    self.assertAlmostEqual(scalar(velocity), 2.3, places=6)
    self.assertAlmostEqual(scalar(position), 1.23, places=6)

  def test_shape_mismatch_is_rejected(self):
    with self.assertRaises(ValueError):
      semi_implicit_euler(Tensor.zeros(2, 1), Tensor.zeros(3, 1), Tensor.zeros(2, 1), 0.1)

  def test_nonfinite_timestep_is_rejected(self):
    with self.assertRaises(ValueError):
      semi_implicit_euler(Tensor.zeros(1, 1), Tensor.zeros(1, 1), Tensor.zeros(1, 1), math.nan)


class TestPendulum(unittest.TestCase):
  def test_nonfinite_physics_parameter_is_rejected(self):
    with self.assertRaises(ValueError):
      pendulum_acceleration(Tensor.zeros(1, 1), Tensor.zeros(1, 1), Tensor.zeros(1, 1), gravity=math.nan)

  def test_acceleration_matches_equation_for_batch(self):
    theta = Tensor([[0.0], [math.pi / 2]])
    omega, torque = Tensor([[2.0], [0.0]]), Tensor([[1.0], [0.0]])
    acceleration = pendulum_acceleration(theta, omega, torque, mass=2.0, length=0.5, damping=0.25)
    self.assertAlmostEqual(scalar(acceleration[0]), 1.0, places=5)
    self.assertAlmostEqual(scalar(acceleration[1]), -19.62, places=4)

  def test_step_matches_scalar_reference(self):
    theta, omega, torque = 0.3, -0.2, 0.7
    mass, length, damping, gravity, dt = 1.3, 0.8, 0.07, 9.81, 0.02
    acceleration = (torque - damping * omega - mass * gravity * length * math.sin(theta)) / (mass * length**2)
    omega_reference = omega + dt * acceleration
    theta_reference = theta + dt * omega_reference
    theta_next, omega_next = pendulum_step(
      Tensor([[theta]]), Tensor([[omega]]), Tensor([[torque]]), mass=mass, length=length,
      damping=damping, gravity=gravity, dt=dt)
    self.assertAlmostEqual(scalar(omega_next), omega_reference, places=6)
    self.assertAlmostEqual(scalar(theta_next), theta_reference, places=6)

  def test_control_gradient_matches_central_difference(self):
    theta, omega, torque = Tensor([[0.2], [-0.1]]), Tensor([[0.1], [0.2]]), Tensor([[0.3], [-0.4]])
    loss = pendulum_step(theta, omega, torque, dt=0.03)[0].square().mean()
    loss.backward()
    autodiff = scalar(torque.grad.sum())
    epsilon = 1e-2
    plus = scalar(pendulum_step(theta.detach(), omega.detach(), torque.detach() + epsilon, dt=0.03)[0].square().mean())
    minus = scalar(pendulum_step(theta.detach(), omega.detach(), torque.detach() - epsilon, dt=0.03)[0].square().mean())
    self.assertAlmostEqual(autodiff, (plus - minus) / (2 * epsilon), delta=2e-5)

  def test_jit_replay_matches_eager(self):
    for worlds in (1, 17):
      with self.subTest(worlds=worlds):
        theta, omega, torque = Tensor.full((worlds, 1), 0.2), Tensor.zeros(worlds, 1), Tensor.full((worlds, 1), 0.1)
        expected, actual, step = (theta, omega), (theta.clone().realize(), omega.clone().realize()), make_jitted_pendulum_step()
        torque = torque.realize()
        for _ in range(3):
          expected, actual = pendulum_step(*expected, torque), step(*actual, torque)
        for got, want in zip(actual, expected):
          self.assertTrue(got.isclose(want, rtol=1e-5, atol=1e-6).all().item())

  def test_short_horizon_and_batched_gradients(self):
    initial_angles = (0.15, -0.25, 0.4)

    controls = (0.1, 0.15, 0.2)

    def gradient(
        angles: tuple[float, ...], commands: tuple[float, ...]
    ) -> tuple[float, ...]:
      theta = Tensor([[value] for value in angles], dtype=dtypes.float64)
      omega = Tensor.zeros(len(angles), 1, dtype=dtypes.float64)
      control = Tensor(
          [[value] for value in commands],
          dtype=dtypes.float64,
      )
      control.requires_grad = True
      for _ in range(8):
        theta, omega = pendulum_step(
            theta, omega, control, damping=0.02, dt=0.01
        )
      result = theta.square().sum().gradient(control)[0]
      return tuple(float(value[0]) for value in result.tolist())

    batched = gradient(initial_angles, controls)
    independent = tuple(
        gradient((angle,), (command,))[0]
        for angle, command in zip(initial_angles, controls, strict=True)
    )
    for actual, expected in zip(batched, independent, strict=True):
      self.assertAlmostEqual(actual, expected, delta=1e-12)

    def loss(offset: float) -> float:
      theta = Tensor(
          [[value] for value in initial_angles], dtype=dtypes.float64
      )
      omega = Tensor.zeros(3, 1, dtype=dtypes.float64)
      control = Tensor(
          [[0.1 + offset], [0.15 + offset], [0.2 + offset]],
          dtype=dtypes.float64,
      )
      for _ in range(8):
        theta, omega = pendulum_step(
            theta, omega, control, damping=0.02, dt=0.01
        )
      return float(theta.square().sum().item())

    epsilon = 1e-5
    finite_difference = (loss(epsilon) - loss(-epsilon)) / (2 * epsilon)
    self.assertAlmostEqual(sum(batched), finite_difference, delta=1e-9)


class TestPointMassSpring(unittest.TestCase):
  def test_acceleration_and_step_match_equations(self):
    acceleration = spring_acceleration(
        Tensor([[0.7]]),
        Tensor([[-0.2]]),
        Tensor([[0.4]]),
        mass=2.0,
        stiffness=3.0,
        damping=0.5,
        rest_position=0.1,
    )
    expected = (0.4 - 3.0 * 0.6 + 0.5 * 0.2) / 2.0
    self.assertAlmostEqual(scalar(acceleration), expected, places=6)
    position, velocity = spring_step(
        Tensor([[0.7]]),
        Tensor([[-0.2]]),
        Tensor([[0.4]]),
        mass=2.0,
        stiffness=3.0,
        damping=0.5,
        rest_position=0.1,
        dt=0.02,
    )
    expected_velocity = -0.2 + 0.02 * expected
    self.assertAlmostEqual(scalar(velocity), expected_velocity, places=6)
    self.assertAlmostEqual(
        scalar(position), 0.7 + 0.02 * expected_velocity, places=6
    )


class TestCartPole(unittest.TestCase):
  def test_acceleration_matches_independent_mass_matrix_solve(self):
    velocity, theta, omega, force = 0.4, 0.2, -0.3, 1.2
    cart_mass, pole_mass, length = 1.1, 0.2, 0.7
    cart_damping, pole_damping, gravity = 0.08, 0.03, 9.81
    sine, cosine = math.sin(theta), math.cos(theta)
    a = cart_mass + pole_mass
    b, d = pole_mass * length * cosine, pole_mass * length**2
    rhs_cart = force - cart_damping * velocity + pole_mass * length * sine * omega**2
    rhs_pole = pole_mass * gravity * length * sine - pole_damping * omega
    determinant = a * d - b * b
    expected_cart = (d * rhs_cart - b * rhs_pole) / determinant
    expected_pole = (a * rhs_pole - b * rhs_cart) / determinant
    cart, pole = cartpole_acceleration(
      Tensor([[velocity]]), Tensor([[theta]]), Tensor([[omega]]), Tensor([[force]]),
      cart_mass=cart_mass, pole_mass=pole_mass, length=length, cart_damping=cart_damping,
      pole_damping=pole_damping, gravity=gravity)
    self.assertAlmostEqual(scalar(cart), expected_cart, places=5)
    self.assertAlmostEqual(scalar(pole), expected_pole, places=5)

  def test_equilibrium_has_zero_acceleration(self):
    zero = Tensor.zeros(4, 1)
    cart, pole = cartpole_acceleration(zero, zero, zero, zero)
    self.assertEqual(scalar(cart.abs().max()), 0.0)
    self.assertEqual(scalar(pole.abs().max()), 0.0)

  def test_control_gradient_matches_central_difference(self):
    position = Tensor([[0.1], [-0.2]])
    velocity, theta = Tensor([[0.2], [0.1]]), Tensor([[0.15], [-0.1]])
    omega, force = Tensor([[0.05], [-0.2]]), Tensor([[0.3], [-0.4]])
    output = cartpole_step(position, velocity, theta, omega, force, dt=0.03)
    loss = output[0].square().mean() + output[2].square().mean()
    loss.backward()
    autodiff = scalar(force.grad.sum())
    epsilon = 1e-2
    def loss_at(control: Tensor) -> float:
      state = cartpole_step(position.detach(), velocity.detach(), theta.detach(), omega.detach(), control, dt=0.03)
      return scalar(state[0].square().mean() + state[2].square().mean())
    finite_difference = (loss_at(force.detach() + epsilon) - loss_at(force.detach() - epsilon)) / (2 * epsilon)
    self.assertAlmostEqual(autodiff, finite_difference, delta=2e-5)

  def test_jit_replay_matches_eager(self):
    for worlds in (1, 17):
      with self.subTest(worlds=worlds):
        state = (Tensor.zeros(worlds, 1), Tensor.full((worlds, 1), 0.2),
                 Tensor.full((worlds, 1), 0.1), Tensor.zeros(worlds, 1))
        force = Tensor.full((worlds, 1), 0.3).realize()
        expected = state
        actual = tuple(tensor.clone().realize() for tensor in state)
        step = make_jitted_cartpole_step()
        for _ in range(3):
          expected, actual = cartpole_step(*expected, force), step(*actual, force)
        for got, want in zip(actual, expected):
          self.assertTrue(got.isclose(want, rtol=1e-5, atol=1e-6).all().item())


if __name__ == "__main__":
  unittest.main()
