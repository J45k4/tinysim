import math
import unittest

from tinygrad import Tensor

from tinysim.actuator import actuator_forces
from tinysim.compile import compile_model
from tinysim.dynamics import batched_solve, bias_forces, forward_dynamics, mass_matrix
from tinysim.integrator import semi_implicit_euler
from tinysim.model import ActuatorSpec, BodySpec, JointSpec, ModelSpec
from tinysim.state import State


def pendulum(mass=2.0, length=3.0, inertia=0.1):
    return compile_model(
        ModelSpec(
            [BodySpec("bob", mass=mass, inertia=(0.1, inertia, 0.1), com=(0.0, 0.0, -length))],
            [JointSpec("hinge", 0, "hinge", (0.0, 1.0, 0.0))],
            [ActuatorSpec("motor", "hinge", "motor", gain=2.0, control_range=(-1.0, 1.0))],
            timestep=0.01,
        )
    )


class TestDynamics(unittest.TestCase):
    def test_pendulum_analytical_acceleration_and_batch(self):
        mass, length, inertia = 2.0, 3.0, 0.0
        model = pendulum(mass, length, inertia)
        angles = [0.0, 0.4, -0.8]
        qpos = Tensor([[angle] for angle in angles])
        qvel = Tensor.zeros(3, 1)
        acceleration = forward_dynamics(model, qpos, qvel, Tensor.zeros(3, 1)).tolist()
        denominator = mass * length * length + inertia
        for angle, actual in zip(angles, acceleration, strict=True):
            expected = -mass * 9.81 * length * math.sin(angle) / denominator
            self.assertAlmostEqual(float(actual[0]), expected, places=5)

    def test_two_link_mass_matrix_matches_analytical_and_is_spd(self):
        m1, m2, length1, com1, com2, inertia1, inertia2 = 1.3, 0.7, 1.4, 0.6, 0.5, 0.2, 0.15
        model = compile_model(
            ModelSpec(
                [
                    BodySpec("one", mass=m1, inertia=(0.1, inertia1, 0.1), com=(0.0, 0.0, -com1)),
                    BodySpec("two", parent=0, mass=m2, inertia=(0.1, inertia2, 0.1), com=(0.0, 0.0, -com2)),
                ],
                [
                    JointSpec("one_hinge", 0, "hinge", (0.0, 1.0, 0.0)),
                    JointSpec("two_hinge", 1, "hinge", (0.0, 1.0, 0.0), pos=(0.0, 0.0, -length1)),
                ],
            )
        )
        relative_angle = 0.63
        matrix = mass_matrix(model, Tensor([[0.2, relative_angle]])).tolist()[0]
        coupling = m2 * length1 * com2 * math.cos(relative_angle)
        expected22 = inertia2 + m2 * com2 * com2
        expected12 = expected22 + coupling
        expected11 = inertia1 + m1 * com1 * com1 + expected22 + m2 * length1 * length1 + 2 * coupling
        self.assertAlmostEqual(float(matrix[0][0]), expected11, places=5)
        self.assertAlmostEqual(float(matrix[0][1]), expected12, places=5)
        self.assertAlmostEqual(float(matrix[1][0]), expected12, places=5)
        self.assertAlmostEqual(float(matrix[1][1]), expected22, places=5)
        self.assertGreater(float(matrix[0][0]), 0.0)
        self.assertGreater(float(matrix[0][0] * matrix[1][1] - matrix[0][1] ** 2), 0.0)

    def test_two_link_bias_matches_closed_form(self):
        m1, m2, length1, com1, com2 = 1.3, 0.7, 1.4, 0.6, 0.5
        model = compile_model(
            ModelSpec(
                [
                    BodySpec("one", mass=m1, inertia=(0.1, 0.2, 0.1), com=(0.0, 0.0, -com1)),
                    BodySpec("two", parent=0, mass=m2, inertia=(0.1, 0.15, 0.1), com=(0.0, 0.0, -com2)),
                ],
                [
                    JointSpec("one_hinge", 0, "hinge", (0.0, 1.0, 0.0)),
                    JointSpec("two_hinge", 1, "hinge", (0.0, 1.0, 0.0), pos=(0.0, 0.0, -length1)),
                ],
            )
        )
        q1, q2, v1, v2 = 0.3, -0.45, 0.7, -0.2
        actual = bias_forces(model, Tensor([[q1, q2]]), Tensor([[v1, v2]])).tolist()[0]
        coupling = m2 * length1 * com2 * math.sin(q2)
        gravity1 = 9.81 * ((m1 * com1 + m2 * length1) * math.sin(q1) + m2 * com2 * math.sin(q1 + q2))
        gravity2 = 9.81 * m2 * com2 * math.sin(q1 + q2)
        expected1 = gravity1 - coupling * (2 * v1 * v2 + v2 * v2)
        expected2 = gravity2 + coupling * v1 * v1
        self.assertAlmostEqual(float(actual[0]), expected1, places=5)
        self.assertAlmostEqual(float(actual[1]), expected2, places=5)

    def test_batched_cholesky_solve(self):
        matrix = Tensor([[[4.0, 1.0], [1.0, 3.0]], [[2.0, 0.0], [0.0, 5.0]]])
        result = batched_solve(matrix, Tensor([[1.0, 2.0], [4.0, 10.0]])).tolist()
        self.assertAlmostEqual(float(result[0][0]), 1.0 / 11.0, places=5)
        self.assertAlmostEqual(float(result[0][1]), 7.0 / 11.0, places=5)
        self.assertAlmostEqual(float(result[1][0]), 2.0, places=5)
        self.assertAlmostEqual(float(result[1][1]), 2.0, places=5)

    def test_actuation_clamp_gradient_and_semi_implicit_step(self):
        model = pendulum(mass=1.0, length=2.0, inertia=0.0 + 0.1)
        control = Tensor([[0.25]])
        qpos, qvel = Tensor([[0.0]]), Tensor([[0.0]])
        force = actuator_forces(model, qpos, qvel, control)
        acceleration = forward_dynamics(model, qpos, qvel, force)
        gradient = acceleration.sum().gradient(control)[0].item()
        expected_gradient = 2.0 / (4.0 + 0.1)
        self.assertAlmostEqual(float(gradient), expected_gradient, places=5)
        state = State(qpos, qvel, control, Tensor([0.0]))
        next_state = semi_implicit_euler(model, state, acceleration)
        self.assertAlmostEqual(float(next_state.qvel.item()), float(acceleration.item()) * 0.01, places=6)
        self.assertAlmostEqual(float(next_state.qpos.item()), float(acceleration.item()) * 0.0001, places=6)
        self.assertAlmostEqual(float(next_state.time.item()), 0.01, places=6)


if __name__ == "__main__":
    unittest.main()
