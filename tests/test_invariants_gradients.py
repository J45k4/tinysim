import math
import unittest

from tinygrad import Tensor, dtypes

from tinysim.collision import ContactParams, sphere_plane
from tinysim.compile import compile_model
from tinysim.contact import smooth_contact
from tinysim.dynamics import forward_dynamics, mass_matrix
from tinysim.model import ActuatorSpec, BodySpec, JointSpec, ModelSpec
from tinysim.simulation import step
from tinysim.spatial import quat_conjugate, rotate
from tinysim.state import RuntimeParameters, State, make_state


def relative_error(actual: float, expected: float) -> float:
    return abs(actual - expected) / max(abs(actual), abs(expected), 1e-12)


class TestPhysicalInvariants(unittest.TestCase):
    def test_free_body_linear_momentum_and_quaternion_norm(self):
        model = compile_model(
            ModelSpec(
                [BodySpec("body", mass=2.5, inertia=(1.0, 1.0, 1.0))],
                [JointSpec("root", 0, "free")],
                gravity=(0.0, 0.0, 0.0),
                timestep=0.001,
            ),
            dtype=dtypes.float64,
        )
        base = make_state(model)
        state = State(
            base.qpos,
            Tensor([[0.4, -0.2, 0.1, 0.3, -0.1, 0.2]], dtype=dtypes.float64),
            base.ctrl,
            base.time,
            base.constraint_impulse,
            base.parameters,
        )
        initial_momentum = state.qvel[:, :3] * 2.5
        for _ in range(5):
            state = step(model, state)
        self.assertTrue(
            (state.qvel[:, :3] * 2.5)
            .isclose(initial_momentum, atol=1e-10, rtol=1e-10)
            .all()
            .item()
        )
        self.assertAlmostEqual(
            float((state.qpos[:, 3:] ** 2).sum().item()), 1.0, places=10
        )

    def test_free_body_ballistic_motion_and_angular_momentum(self):
        model = compile_model(
            ModelSpec(
                [BodySpec("body", mass=2.0, inertia=(0.2, 0.3, 0.4))],
                [JointSpec("root", 0, "free")],
                gravity=(0.0, 0.0, -9.81),
                timestep=1e-4,
            ),
            dtype=dtypes.float64,
        )
        base = make_state(model)
        state = State(
            base.qpos,
            Tensor(
                [[0.3, -0.2, 0.5, 0.7, -0.4, 0.2]],
                dtype=dtypes.float64,
            ),
            base.ctrl,
            base.time,
            base.constraint_impulse,
            base.parameters,
        )

        def angular_momentum(candidate: State) -> Tensor:
            quaternion = candidate.qpos[:, 3:7]
            local_omega = rotate(
                quat_conjugate(quaternion),
                candidate.qvel[:, 3:6],
            )
            return rotate(
                quaternion,
                model.body_inertia[0] * local_omega,
            )

        initial_angular_momentum = angular_momentum(state)
        advanced = step(model, state)
        self.assertTrue(
            angular_momentum(advanced)
            .isclose(
                initial_angular_momentum,
                atol=1e-9,
                rtol=1e-9,
            )
            .all()
            .item()
        )
        expected_velocity_z = 0.5 - 9.81e-4
        expected_position_z = expected_velocity_z * 1e-4
        self.assertAlmostEqual(
            float(advanced.qvel[0, 2].item()),
            expected_velocity_z,
            places=12,
        )
        self.assertAlmostEqual(
            float(advanced.qpos[0, 2].item()),
            expected_position_z,
            places=12,
        )

    def test_crba_matches_independent_body_jacobian_reference(self):
        from tinysim.dynamics import mass_matrix_jacobian

        model = compile_model(
            ModelSpec(
                [
                    BodySpec("base", mass=2.0, inertia=(0.5, 0.6, 0.7)),
                    BodySpec(
                        "child",
                        parent=0,
                        mass=0.7,
                        inertia=(0.1, 0.12, 0.14),
                        com=(0.1, 0.0, -0.2),
                    ),
                ],
                [
                    JointSpec("root", 0, "free"),
                    JointSpec(
                        "child_ball",
                        1,
                        "ball",
                        pos=(0.0, 0.0, -0.5),
                    ),
                ],
                gravity=(0.0, 0.0, 0.0),
            ),
            dtype=dtypes.float64,
        )
        qpos = Tensor(
            [
                [0.1, -0.2, 0.3, 0.98, 0.1, 0.05, -0.02, 0.96, 0.1, -0.2, 0.15],
                [-0.3, 0.2, 0.1, 0.92, -0.1, 0.2, 0.3, 0.9, -0.2, 0.1, 0.3],
            ],
            dtype=dtypes.float64,
        )
        actual = mass_matrix(model, qpos)
        reference = mass_matrix_jacobian(model, qpos)
        self.assertTrue(
            actual.isclose(reference, atol=1e-9, rtol=1e-9).all().item()
        )

    def test_float32_and_float64_mass_matrices_are_spd(self):
        spec = ModelSpec(
            [BodySpec("link", inertia=(0.2, 0.3, 0.4), com=(0.2, 0.0, -0.4))],
            [JointSpec("hinge", 0, "hinge", axis=(0.0, 1.0, 0.0))],
        )
        for dtype, tolerance in ((dtypes.float32, 1e-5), (dtypes.float64, 1e-10)):
            with self.subTest(dtype=dtype):
                model = compile_model(spec, dtype=dtype)
                matrix = mass_matrix(model, Tensor([[0.3]], dtype=dtype))
                self.assertTrue(
                    matrix.isclose(
                        matrix.transpose(-1, -2),
                        atol=tolerance,
                        rtol=tolerance,
                    ).all().item()
                )
                self.assertGreater(float(matrix.item()), 0.0)


class TestParameterGradients(unittest.TestCase):
    def test_initial_position_and_velocity_gradients_match_finite_difference(self):
        model = compile_model(
            ModelSpec(
                [
                    BodySpec(
                        "link",
                        inertia=(0.2, 0.3, 0.2),
                        com=(0.0, 0.0, -0.8),
                    )
                ],
                [JointSpec("hinge", 0, "hinge", axis=(0.0, 1.0, 0.0))],
                timestep=0.01,
            ),
            dtype=dtypes.float64,
        )
        base = make_state(model)
        qpos = Tensor([[0.25]], dtype=dtypes.float64)
        qvel = Tensor([[-0.15]], dtype=dtypes.float64)
        qpos.requires_grad = True
        qvel.requires_grad = True
        state = State(
            qpos,
            qvel,
            base.ctrl,
            base.time,
            base.constraint_impulse,
            base.parameters,
        )
        output = step(model, state).qpos.sum()
        position_gradient, velocity_gradient = output.gradient(qpos, qvel)

        def evaluate(position: float, velocity: float) -> float:
            candidate = State(
                Tensor([[position]], dtype=dtypes.float64),
                Tensor([[velocity]], dtype=dtypes.float64),
                base.ctrl,
                base.time,
                base.constraint_impulse,
                base.parameters,
            )
            return float(step(model, candidate).qpos.item())

        epsilon = 1e-5
        position_fd = (
            evaluate(0.25 + epsilon, -0.15)
            - evaluate(0.25 - epsilon, -0.15)
        ) / (2 * epsilon)
        velocity_fd = (
            evaluate(0.25, -0.15 + epsilon)
            - evaluate(0.25, -0.15 - epsilon)
        ) / (2 * epsilon)
        self.assertLess(
            relative_error(float(position_gradient.item()), position_fd),
            1e-7,
        )
        self.assertLess(
            relative_error(float(velocity_gradient.item()), velocity_fd),
            1e-7,
        )

    def test_mass_and_actuator_gain_gradients_match_finite_difference(self):
        spec = ModelSpec(
            [BodySpec("slider", mass=2.0, inertia=(1.0, 1.0, 1.0))],
            [JointSpec("slide", 0, "slide", axis=(1.0, 0.0, 0.0))],
            [ActuatorSpec("motor", "slide", gain=3.0)],
            gravity=(0.0, 0.0, 0.0),
            timestep=0.01,
        )
        model = compile_model(spec, dtype=dtypes.float64)
        base = make_state(model)
        mass = Tensor([[2.0]], dtype=dtypes.float64)
        gain = Tensor([[3.0]], dtype=dtypes.float64)
        mass.requires_grad = True
        gain.requires_grad = True
        parameters = RuntimeParameters(
            mass, base.parameters.geom_friction, gain
        )
        state = State(
            base.qpos,
            base.qvel,
            base.ctrl,
            base.time,
            base.constraint_impulse,
            parameters,
        )
        loss = step(model, state, Tensor([[0.4]], dtype=dtypes.float64)).qvel.sum()
        mass_gradient, gain_gradient = loss.gradient(mass, gain)

        def evaluate(mass_value: float, gain_value: float) -> float:
            params = RuntimeParameters(
                Tensor([[mass_value]], dtype=dtypes.float64),
                base.parameters.geom_friction,
                Tensor([[gain_value]], dtype=dtypes.float64),
            )
            candidate = State(
                base.qpos,
                base.qvel,
                base.ctrl,
                base.time,
                base.constraint_impulse,
                params,
            )
            return float(
                step(
                    model,
                    candidate,
                    Tensor([[0.4]], dtype=dtypes.float64),
                ).qvel.item()
            )

        epsilon = 1e-5
        mass_fd = (
            evaluate(2.0 + epsilon, 3.0)
            - evaluate(2.0 - epsilon, 3.0)
        ) / (2 * epsilon)
        gain_fd = (
            evaluate(2.0, 3.0 + epsilon)
            - evaluate(2.0, 3.0 - epsilon)
        ) / (2 * epsilon)
        self.assertLess(
            relative_error(float(mass_gradient.item()), mass_fd), 1e-7
        )
        self.assertLess(
            relative_error(float(gain_gradient.item()), gain_fd), 1e-7
        )

    def test_inertia_gradient_matches_finite_difference(self):
        def acceleration(inertia_y: float, differentiable: bool = False):
            model = compile_model(
                ModelSpec(
                    [BodySpec("body", inertia=(0.4, inertia_y, 0.6))],
                    [JointSpec("hinge", 0, "hinge", axis=(0.0, 1.0, 0.0))],
                    gravity=(0.0, 0.0, 0.0),
                ),
                dtype=dtypes.float64,
            )
            if differentiable:
                model.body_inertia.requires_grad = True
            output = forward_dynamics(
                model,
                Tensor.zeros(1, 1, dtype=dtypes.float64),
                Tensor.zeros(1, 1, dtype=dtypes.float64),
                Tensor.ones(1, 1, dtype=dtypes.float64),
            )
            return model, output

        model, output = acceleration(0.5, True)
        gradient = output.sum().gradient(model.body_inertia)[0][0, 1].item()
        epsilon = 1e-5
        finite = (
            acceleration(0.5 + epsilon)[1].item()
            - acceleration(0.5 - epsilon)[1].item()
        ) / (2 * epsilon)
        self.assertLess(relative_error(float(gradient), float(finite)), 1e-7)

    def test_contact_parameter_gradients_match_finite_difference(self):
        center = Tensor([[0.0, 0.0, 0.8]], dtype=dtypes.float64)
        geometry = sphere_plane(
            center,
            1.0,
            Tensor([0.0, 0.0, 0.0], dtype=dtypes.float64),
            Tensor([0.0, 0.0, 1.0], dtype=dtypes.float64),
        )
        stiffness = Tensor([40.0], dtype=dtypes.float64)
        friction = Tensor([0.3], dtype=dtypes.float64)
        penetration_smoothing = Tensor([0.01], dtype=dtypes.float64)
        velocity_smoothing = Tensor([0.05], dtype=dtypes.float64)
        stiffness.requires_grad = True
        friction.requires_grad = True
        penetration_smoothing.requires_grad = True
        velocity_smoothing.requires_grad = True

        def loss(k, mu, penetration_epsilon, velocity_epsilon):
            return smooth_contact(
                geometry,
                Tensor([[1.0, 0.0, -0.2]], dtype=dtypes.float64),
                Tensor.zeros(1, 3, dtype=dtypes.float64),
                ContactParams(
                    stiffness=k,
                    damping=1.0,
                    friction=mu,
                    penetration_smoothing=penetration_epsilon,
                    force_smoothing=1e-5,
                    velocity_smoothing=velocity_epsilon,
                ),
            ).force_a.sum()

        output = loss(
            stiffness,
            friction,
            penetration_smoothing,
            velocity_smoothing,
        )
        gradients = output.gradient(
            stiffness,
            friction,
            penetration_smoothing,
            velocity_smoothing,
        )
        epsilon = 1e-5
        base = (40.0, 0.3, 0.01, 0.05)

        def evaluate(values: tuple[float, float, float, float]) -> float:
            return float(
                loss(
                    *(Tensor([value], dtype=dtypes.float64) for value in values)
                ).item()
            )

        for index, (name, gradient) in enumerate(
            zip(
                (
                    "stiffness",
                    "friction",
                    "penetration_smoothing",
                    "velocity_smoothing",
                ),
                gradients,
                strict=True,
            )
        ):
            plus, minus = list(base), list(base)
            plus[index] += epsilon
            minus[index] -= epsilon
            finite_difference = (
                evaluate(tuple(plus)) - evaluate(tuple(minus))
            ) / (2 * epsilon)
            with self.subTest(parameter=name):
                self.assertLess(
                    relative_error(float(gradient.item()), finite_difference),
                    1e-6,
                )


if __name__ == "__main__":
    unittest.main()
