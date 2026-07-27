import math
import unittest

from tinygrad import Tensor

from tinysim.collision import ContactParams
from tinysim.systems import make_jitted_sphere_plane_step, sphere_plane_step


class TestBouncingBall(unittest.TestCase):
    def test_free_fall_step_matches_semi_implicit_euler(self):
        position = Tensor([[0.2, -0.1, 2.0]])
        velocity = Tensor([[1.0, 2.0, -0.5]])
        timestep = 0.01
        position_next, velocity_next = sphere_plane_step(
            position, velocity, gravity=(0.0, 0.0, -10.0), timestep=timestep
        )
        expected_velocity = [[1.0, 2.0, -0.6]]
        expected_position = [[0.21, -0.08, 1.994]]
        self.assertTrue(velocity_next.isclose(Tensor(expected_velocity), atol=1e-6, rtol=0.0).all().item())
        self.assertTrue(position_next.isclose(Tensor(expected_position), atol=1e-6, rtol=0.0).all().item())

    def test_equal_batch_members_remain_equal(self):
        position = Tensor([[0.1, -0.2, 0.2]] * 8)
        velocity = Tensor([[0.3, 0.1, -1.0]] * 8)
        position_next, velocity_next = sphere_plane_step(position, velocity)
        self.assertTrue((position_next == position_next[0].expand(position_next.shape)).all().item())
        self.assertTrue((velocity_next == velocity_next[0].expand(velocity_next.shape)).all().item())

    def test_long_rollout_is_finite_and_settles_near_surface(self):
        position = Tensor([[0.0, 0.0, 1.0], [0.1, -0.1, 1.5]]).realize()
        velocity = Tensor.zeros(2, 3).realize()
        step = make_jitted_sphere_plane_step()
        for _ in range(2000):
            position, velocity = step(position, velocity)
        values = position.tolist() + velocity.tolist()
        self.assertTrue(all(math.isfinite(value) for row in values for value in row))
        # Compliant contact rests with a small penetration; these deliberately
        # loose bounds detect explosions and failure to dissipate, not tuning.
        for height, speed in zip(position[:, 2].tolist(), velocity[:, 2].abs().tolist()):
            self.assertGreater(height, 0.23)
            self.assertLess(height, 0.27)
            self.assertLess(speed, 0.05)

    def test_contact_parameter_stability_envelope(self):
        # Defined CPU stability envelope for the explicit smooth-contact step.
        cases = (
            (2_500.0, 70.0, 0.001),
            (5_000.0, 100.0, 0.002),
            (8_000.0, 140.0, 0.001),
        )
        for stiffness, damping, timestep in cases:
            with self.subTest(
                stiffness=stiffness,
                damping=damping,
                timestep=timestep,
            ):
                params = ContactParams(
                    stiffness=stiffness,
                    damping=damping,
                    friction=0.5,
                    penetration_smoothing=1e-4,
                    force_smoothing=1e-6,
                    velocity_smoothing=1e-3,
                )
                position = Tensor([[0.0, 0.0, 0.75]]).realize()
                velocity = Tensor.zeros(1, 3).realize()
                step = make_jitted_sphere_plane_step(
                    timestep=timestep,
                    contact_params=params,
                )
                for _ in range(600):
                    position, velocity = step(position, velocity)
                values = position.tolist()[0] + velocity.tolist()[0]
                self.assertTrue(all(math.isfinite(value) for value in values))
                self.assertGreater(values[2], 0.15)
                self.assertLess(values[2], 2.0)
                self.assertLess(max(abs(value) for value in values[3:]), 20.0)

    def test_position_gradient_matches_finite_difference_inside_contact(self):
        params = ContactParams(
            stiffness=300.0,
            damping=4.0,
            friction=0.0,
            penetration_smoothing=0.01,
            force_smoothing=1e-5,
        )
        height = Tensor([0.2])
        position = Tensor.stack(Tensor.zeros(1), Tensor.zeros(1), height, dim=-1)
        velocity = Tensor.zeros(1, 3)
        output = sphere_plane_step(
            position, velocity, radius=0.25, timestep=0.01, contact_params=params
        )[1][:, 2].sum()
        output.backward()
        analytical = height.grad.item()

        def vertical_velocity(z: float) -> float:
            return sphere_plane_step(
                Tensor([[0.0, 0.0, z]]),
                Tensor.zeros(1, 3),
                radius=0.25,
                timestep=0.01,
                contact_params=params,
            )[1][0, 2].item()

        epsilon = 1e-3
        finite_difference = (
            vertical_velocity(0.2 + epsilon) - vertical_velocity(0.2 - epsilon)
        ) / (2 * epsilon)
        self.assertAlmostEqual(analytical, finite_difference, delta=2e-3)

    def test_jit_replay_matches_eager(self):
        position = Tensor([[0.0, 0.0, 0.2], [0.1, -0.2, 1.0]])
        velocity = Tensor([[0.2, 0.0, -0.5], [0.0, 0.1, 0.2]])
        expected = (position, velocity)
        actual = (position.clone().realize(), velocity.clone().realize())
        step = make_jitted_sphere_plane_step()
        for _ in range(4):
            expected = sphere_plane_step(*expected)
            actual = step(*actual)
        for got, want in zip(actual, expected):
            self.assertTrue(got.isclose(want, rtol=1e-5, atol=1e-6).all().item())


if __name__ == "__main__":
    unittest.main()
