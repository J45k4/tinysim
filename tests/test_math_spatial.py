import math
import unittest

from tinygrad import Tensor

from tinysim.math import cross, dot, skew
from tinysim.spatial import (
    axis_angle,
    force_cross,
    motion_cross,
    quat_integrate,
    quat_mul,
    quat_to_matrix,
    rotate,
)


def assert_vector_close(test: unittest.TestCase, actual, expected, places=5):
    for got, want in zip(actual, expected, strict=True):
        test.assertAlmostEqual(float(got), float(want), places=places)


class TestMathSpatial(unittest.TestCase):
    def test_cross_and_skew_agree(self):
        left = Tensor([[1.0, 2.0, 3.0], [-2.0, 1.0, 0.5]])
        right = Tensor([[4.0, -5.0, 6.0], [3.0, 2.0, -1.0]])
        direct = cross(left, right)
        matrix = (skew(left) @ right.unsqueeze(-1)).squeeze(-1)
        for direct_row, matrix_row in zip(direct.tolist(), matrix.tolist(), strict=True):
            assert_vector_close(self, direct_row, matrix_row)
        self.assertAlmostEqual(float(dot(direct[0], left[0]).item()), 0.0, places=5)
        self.assertAlmostEqual(float(dot(direct[0], right[0]).item()), 0.0, places=5)

    def test_quaternion_rotation_and_composition(self):
        z90 = axis_angle(Tensor([[0.0, 0.0, 1.0]]), Tensor([math.pi / 2]))
        y90 = axis_angle(Tensor([[0.0, 1.0, 0.0]]), Tensor([math.pi / 2]))
        rotated = rotate(z90, Tensor([[1.0, 0.0, 0.0]])).tolist()[0]
        assert_vector_close(self, rotated, [0.0, 1.0, 0.0])
        composed = quat_mul(y90, z90)
        sequential = rotate(y90, rotate(z90, Tensor([[1.0, 0.0, 0.0]])))
        assert_vector_close(self, rotate(composed, Tensor([[1.0, 0.0, 0.0]])).tolist()[0], sequential.tolist()[0])

    def test_quaternion_matrix_is_orthogonal(self):
        quaternion = axis_angle(Tensor([[1.0, 2.0, -3.0]]), Tensor([0.73]))
        rotation = quat_to_matrix(quaternion)
        product = (rotation.transpose(-1, -2) @ rotation).tolist()[0]
        for row in range(3):
            for column in range(3):
                self.assertAlmostEqual(float(product[row][column]), float(row == column), places=5)

    def test_quaternion_integration(self):
        integrated = quat_integrate(
            Tensor([[1.0, 0.0, 0.0, 0.0]]),
            Tensor([[0.0, 0.0, math.pi]]),
            0.5,
        )
        rotated = rotate(integrated, Tensor([[1.0, 0.0, 0.0]])).tolist()[0]
        assert_vector_close(self, rotated, [0.0, 1.0, 0.0])

    def test_force_cross_is_negative_transpose(self):
        motion = Tensor([[0.2, -0.3, 0.4, 1.0, -2.0, 3.0]])
        total = force_cross(motion) + motion_cross(motion).transpose(-1, -2)
        self.assertLess(max(abs(float(x)) for row in total.tolist()[0] for x in row), 1e-6)


if __name__ == "__main__":
    unittest.main()
