import unittest

from tinygrad import Tensor

from tinysim.constraint import solve_contact_impulses
from tinysim.solver import projected_jacobi


class TestSolvers(unittest.TestCase):
    def test_projected_jacobi_nonnegative(self):
        matrix = Tensor([[[2.0, 0.0], [0.0, 4.0]]])
        rhs = Tensor([[2.0, -4.0]])
        result = projected_jacobi(matrix, rhs, iterations=2).tolist()
        self.assertAlmostEqual(float(result[0][0]), 1.0, places=6)
        self.assertEqual(float(result[0][1]), 0.0)

    def test_single_contact_impulse(self):
        inverse_mass = Tensor([[[0.5]]])
        jacobian = Tensor([[[1.0]]])
        impulses, delta_velocity = solve_contact_impulses(
            inverse_mass,
            jacobian,
            normal_velocity=Tensor([[-2.0]]),
            penetration=Tensor([[0.0]]),
            active=Tensor([[True]]),
            timestep=0.01,
            iterations=2,
        )
        self.assertGreater(float(impulses.item()), 3.99)
        self.assertAlmostEqual(float(delta_velocity.item()), 2.0, places=4)

    def test_inactive_contact_is_zero(self):
        impulses, delta_velocity = solve_contact_impulses(
            Tensor([[[1.0]]]),
            Tensor([[[1.0]]]),
            normal_velocity=Tensor([[-2.0]]),
            penetration=Tensor([[0.1]]),
            active=Tensor([[False]]),
            timestep=0.01,
        )
        self.assertEqual(float(impulses.item()), 0.0)
        self.assertEqual(float(delta_velocity.item()), 0.0)

    def test_fixed_iteration_convergence_is_measured(self):
        matrix = Tensor([[[2.0, 0.5], [0.5, 1.5]]])
        rhs = Tensor([[1.0, 0.75]])
        target = Tensor([[0.4090909091, 0.3636363636]])
        errors = []
        for iterations in (1, 2, 4, 8):
            solution = projected_jacobi(
                matrix, rhs, iterations=iterations
            )
            errors.append(float((solution - target).abs().max().item()))
        self.assertTrue(
            all(later < earlier for earlier, later in zip(errors, errors[1:]))
        )
        self.assertLess(errors[-1], 1e-4)

    def test_warm_start_changes_fixed_iteration_result(self):
        matrix = Tensor([[[2.0, 0.5], [0.5, 1.5]]])
        rhs = Tensor([[1.0, 0.75]])
        cold = projected_jacobi(matrix, rhs, iterations=1)
        warm = projected_jacobi(
            matrix,
            rhs,
            iterations=1,
            initial=Tensor([[0.4, 0.35]]),
        )
        target = Tensor([[0.4090909091, 0.3636363636]])
        self.assertLess(
            float((warm - target).abs().max().item()),
            float((cold - target).abs().max().item()),
        )


if __name__ == "__main__":
    unittest.main()
