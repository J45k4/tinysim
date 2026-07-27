import math
import unittest

from tinygrad import Tensor

from tinysim.compile import compile_model
from tinysim.dynamics import (
    batched_solve,
    bias_forces,
    forward_dynamics,
    mass_matrix,
)
from tinysim.integrator import semi_implicit_euler
from tinysim.kinematics import forward_kinematics
from tinysim.model import ActuatorSpec, BodySpec, JointSpec, ModelSpec, UnsupportedModelError
from tinysim.spatial import rotate
from tinysim.state import State, make_state


def one_body(kind: str, *, gravity=(0.0, 0.0, -9.81)):
    return compile_model(
        ModelSpec(
            [BodySpec("body", mass=2.0, inertia=(0.5, 0.75, 1.0))],
            [JointSpec("joint", 0, kind)],  # type: ignore[arg-type]
            gravity=gravity,
            timestep=0.1,
        )
    )


class TestGeneralizedJoints(unittest.TestCase):
    def test_addresses_and_identity_state(self):
        ball = one_body("ball")
        free = one_body("free")
        self.assertEqual((ball.nq, ball.nv), (4, 3))
        self.assertEqual((free.nq, free.nv), (7, 6))
        self.assertEqual(make_state(ball, 2).qpos.tolist(), [[1.0, 0.0, 0.0, 0.0]] * 2)
        self.assertEqual(
            make_state(free, 2).qpos.tolist(),
            [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]] * 2,
        )

    def test_fixed_only_state_has_empty_generalized_coordinates(self):
        model = one_body("fixed")
        state = make_state(model, 3)
        self.assertEqual(state.qpos.shape, (3, 0))
        self.assertEqual(state.qvel.shape, (3, 0))
        self.assertEqual(mass_matrix(model, state.qpos).shape, (3, 0, 0))

    def test_ball_and_free_kinematics(self):
        ball = one_body("ball")
        half = math.sqrt(0.5)
        ball_kin = forward_kinematics(ball, Tensor([[half, 0.0, 0.0, half]]))
        direction = rotate(ball_kin.body_quat[:, 0], Tensor([[1.0, 0.0, 0.0]])).tolist()[0]
        self.assertAlmostEqual(float(direction[0]), 0.0, places=5)
        self.assertAlmostEqual(float(direction[1]), 1.0, places=5)

        free = one_body("free")
        free_kin = forward_kinematics(
            free, Tensor([[1.25, -2.0, 0.5, 1.0, 0.0, 0.0, 0.0]])
        )
        self.assertTrue(
            free_kin.body_pos[:, 0].isclose(
                Tensor([[1.25, -2.0, 0.5]]), atol=1e-6, rtol=0.0
            ).all().item()
        )

    def test_quaternion_integration_and_free_translation(self):
        ball = one_body("ball")
        ball_state = State(
            make_state(ball).qpos,
            Tensor([[0.0, 0.0, math.pi]]),
            Tensor.zeros(1, 0),
            Tensor.zeros(1),
        )
        ball_next = semi_implicit_euler(
            ball,
            ball_state,
            Tensor.zeros(1, 3),
        )
        self.assertAlmostEqual(float((ball_next.qpos * ball_next.qpos).sum().item()), 1.0, places=6)
        self.assertAlmostEqual(float(ball_next.qpos[0, 3].item()), math.sin(math.pi * 0.05), places=5)

        free = one_body("free")
        state = State(
            make_state(free).qpos,
            Tensor([[1.0, -2.0, 0.5, 0.0, 0.0, 1.0]]),
            Tensor.zeros(1, 0),
            Tensor.zeros(1),
        )
        result = semi_implicit_euler(free, state, Tensor.zeros(1, 6))
        self.assertTrue(
            result.qpos[:, :3].isclose(
                Tensor([[0.1, -0.2, 0.05]]), atol=1e-6, rtol=0.0
            ).all().item()
        )
        self.assertAlmostEqual(float((result.qpos[:, 3:] ** 2).sum().item()), 1.0, places=6)

    def test_single_free_body_mass_matrix_and_gravity(self):
        model = one_body("free")
        state = make_state(model, 2)
        matrix = mass_matrix(model, state.qpos).tolist()
        expected = (2.0, 2.0, 2.0, 0.5, 0.75, 1.0)
        for world in matrix:
            for row in range(6):
                for column in range(6):
                    target = expected[row] if row == column else 0.0
                    self.assertAlmostEqual(float(world[row][column]), target, places=5)
        acceleration = forward_dynamics(
            model, state.qpos, state.qvel, Tensor.zeros(2, 6)
        )
        target = Tensor([[0.0, 0.0, -9.81, 0.0, 0.0, 0.0]] * 2)
        self.assertTrue(acceleration.isclose(target, atol=1e-5, rtol=0.0).all().item())

    def test_generalized_dynamics_gradient_is_finite(self):
        model = one_body("ball", gravity=(0.0, 0.0, 0.0))
        force = Tensor([[0.2, -0.3, 0.4]])
        force.requires_grad = True
        acceleration = forward_dynamics(
            model, make_state(model).qpos, Tensor.zeros(1, 3), force
        )
        gradient = acceleration.sum().gradient(force)[0]
        self.assertTrue(gradient.isfinite().all().item())
        self.assertGreater(float(gradient.abs().sum().item()), 0.0)

    def test_free_body_euler_equation_for_anisotropic_inertia(self):
        model = one_body("free", gravity=(0.0, 0.0, 0.0))
        state = make_state(model)
        angular_velocity = (0.4, -0.3, 0.2)
        qvel = Tensor([[0.0, 0.0, 0.0, *angular_velocity]])
        acceleration = forward_dynamics(
            model, state.qpos, qvel, Tensor.zeros(1, 6)
        ).tolist()[0]
        inertia = (0.5, 0.75, 1.0)
        expected = (
            -(inertia[2] - inertia[1])
            * angular_velocity[1]
            * angular_velocity[2]
            / inertia[0],
            -(inertia[0] - inertia[2])
            * angular_velocity[2]
            * angular_velocity[0]
            / inertia[1],
            -(inertia[1] - inertia[0])
            * angular_velocity[0]
            * angular_velocity[1]
            / inertia[2],
        )
        for actual, target in zip(acceleration[3:], expected):
            self.assertAlmostEqual(float(actual), target, places=5)

    def test_independent_free_body_fast_path_matches_dense_reference(self):
        model = compile_model(
            ModelSpec(
                bodies=[
                    BodySpec("a", mass=2.0, inertia=(0.5, 0.75, 1.0)),
                    BodySpec("b", mass=1.5, inertia=(0.4, 0.6, 0.8)),
                ],
                joints=[
                    JointSpec("free_a", 0, "free"),
                    JointSpec("free_b", 1, "free"),
                ],
            )
        )
        state = make_state(model)
        qvel = Tensor([[
            0.1, -0.2, 0.3, 0.4, -0.3, 0.2,
            -0.2, 0.1, -0.4, 0.3, 0.2, -0.1,
        ]])
        force = Tensor([[
            0.4, -0.1, 0.2, -0.3, 0.5, 0.1,
            -0.2, 0.3, 0.1, 0.4, -0.1, 0.2,
        ]])
        fast = forward_dynamics(model, state.qpos, qvel, force)
        dense = batched_solve(
            mass_matrix(model, state.qpos),
            force - bias_forces(model, state.qpos, qvel),
        )
        self.assertTrue(
            fast.isclose(dense, atol=1e-5, rtol=1e-5).all().item()
        )

    def test_multidof_actuator_requires_transmission(self):
        for kind in ("ball", "free"):
            with self.subTest(kind=kind), self.assertRaisesRegex(
                UnsupportedModelError, "explicit transmission"
            ):
                compile_model(
                    ModelSpec(
                        [BodySpec("body", inertia=(1.0, 1.0, 1.0))],
                        [JointSpec("joint", 0, kind)],  # type: ignore[arg-type]
                        [ActuatorSpec("motor", "joint")],
                    )
                )


if __name__ == "__main__":
    unittest.main()
