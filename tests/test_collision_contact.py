import math
import unittest

from tinygrad import Tensor

from tinysim.collision import (
    ContactParams,
    all_pairs,
    box_box,
    box_plane,
    capsule_capsule,
    capsule_plane,
    fixed_pairs,
    sphere_capsule,
    sphere_plane,
    sphere_sphere,
)
from tinysim.contact import smooth_contact


def values(tensor: Tensor):
    return tensor.tolist()


class TestCollisionPrimitives(unittest.TestCase):
    def test_sphere_plane_separated_touching_penetrating_batch(self):
        result = sphere_plane(
            Tensor([[0.0, 0.0, 2.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.25]]),
            1.0,
            Tensor([0.0, 0.0, 0.0]),
            Tensor([0.0, 0.0, 2.0]),
        )
        self.assertEqual(values(result.distance), [1.0, 0.0, -0.75])
        self.assertEqual(values(result.active), [False, True, True])
        self.assertEqual(values(result.normal), [[0.0, 0.0, -1.0]] * 3)
        self.assertEqual(values(result.point_b), [[0.0, 0.0, 0.0]] * 3)

    def test_sphere_sphere_witnesses_and_concentric_fallback(self):
        result = sphere_sphere(
            Tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            Tensor([1.0, 1.0]),
            Tensor([[3.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            Tensor([1.0, 2.0]),
        )
        self.assertEqual(values(result.distance), [1.0, -3.0])
        self.assertEqual(values(result.normal), [[1.0, 0.0, 0.0]] * 2)
        self.assertEqual(values(result.point_a), [[1.0, 0.0, 0.0]] * 2)
        self.assertEqual(values(result.point_b), [[2.0, 0.0, 0.0], [-2.0, 0.0, 0.0]])

    def test_sphere_capsule_projection_and_degenerate_segment(self):
        result = sphere_capsule(
            Tensor([[2.0, 0.0, 1.0], [0.0, 0.0, 0.0]]),
            0.5,
            Tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            Tensor([[0.0, 0.0, 2.0], [0.0, 0.0, 0.0]]),
            0.5,
        )
        self.assertEqual(values(result.distance), [1.0, -1.0])
        self.assertEqual(values(result.normal), [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        for field in (result.distance, result.normal, result.point_a, result.point_b):
            flattened = (
                x for row in values(field) for x in (row if isinstance(row, list) else [row])
            )
            self.assertTrue(all(math.isfinite(x) for x in flattened))

    def test_capsule_plane_selects_low_endpoint(self):
        result = capsule_plane(
            Tensor([[0.0, 0.0, 2.0], [0.0, 0.0, 0.25]]),
            Tensor([[0.0, 0.0, 4.0], [0.0, 0.0, 2.0]]),
            0.5,
            Tensor([0.0, 0.0, 0.0]),
            Tensor([0.0, 0.0, 1.0]),
        )
        self.assertEqual(values(result.distance), [1.5, -0.25])
        self.assertEqual(values(result.normal), [[0.0, 0.0, -1.0]] * 2)
        self.assertEqual(values(result.point_a), [[0.0, 0.0, 1.5], [0.0, 0.0, -0.25]])

    def test_capsule_capsule_and_box_plane(self):
        capsules = capsule_capsule(
            Tensor([[0.0, 0.0, -1.0]]),
            Tensor([[0.0, 0.0, 1.0]]),
            0.25,
            Tensor([[0.4, 0.0, -1.0]]),
            Tensor([[0.4, 0.0, 1.0]]),
            0.25,
        )
        self.assertAlmostEqual(float(capsules.distance.item()), -0.1, places=6)
        self.assertTrue(capsules.active.item())

        box = box_plane(
            Tensor([[0.0, 0.0, 0.4]]),
            Tensor([[0.2, 0.3, 0.5]]),
            Tensor([[1.0, 0.0, 0.0, 0.0]]),
            Tensor([0.0, 0.0, 0.0]),
            Tensor([0.0, 0.0, 1.0]),
        )
        self.assertAlmostEqual(float(box.distance.item()), -0.1, places=6)
        self.assertEqual(values(box.normal), [[0.0, 0.0, -1.0]])

    def test_box_plane_returns_rotated_support_vertex(self):
        # An oblique normal makes every local support coordinate observable.
        center = Tensor([[0.3, -0.2, 1.7]])
        half_size = Tensor([[1.0, 0.5, 0.25]])
        quaternion = Tensor([[0.9659258263, 0.0, 0.0, 0.2588190451]])
        normal = Tensor([1.0, 2.0, 3.0])
        box = box_plane(
            center,
            half_size,
            quaternion,
            Tensor([0.0, 0.0, 0.0]),
            normal,
        )
        point = values(box.point_a)[0]
        cx, cy, cz = values(center)[0]
        dx, dy, dz = point[0] - cx, point[1] - cy, point[2] - cz
        # Inverse of the 30-degree local-to-world rotation about z.
        cosine, sine = 0.8660254038, 0.5
        local = (cosine * dx + sine * dy, -sine * dx + cosine * dy, dz)
        for actual, expected in zip(map(abs, local), (1.0, 0.5, 0.25)):
            self.assertAlmostEqual(actual, expected, places=5)

        nx, ny, nz = (component / math.sqrt(14.0) for component in (1.0, 2.0, 3.0))
        plane_height = point[0] * nx + point[1] * ny + point[2] * nz
        self.assertAlmostEqual(plane_height, float(box.distance.item()), places=5)
        projected = values(box.point_b)[0]
        self.assertAlmostEqual(projected[0] * nx + projected[1] * ny + projected[2] * nz, 0.0, places=5)

    def test_box_box_sat_reports_gap_and_penetration(self):
        identity = Tensor([[1.0, 0.0, 0.0, 0.0]])
        size = Tensor([[1.0, 0.5, 0.25]])
        separated = box_box(
            Tensor([[0.0, 0.0, 0.0]]),
            size,
            identity,
            Tensor([[2.5, 0.0, 0.0]]),
            size,
            identity,
        )
        penetrating = box_box(
            Tensor([[0.0, 0.0, 0.0]]),
            size,
            identity,
            Tensor([[1.75, 0.0, 0.0]]),
            size,
            identity,
        )
        self.assertAlmostEqual(float(separated.distance.item()), 0.5, places=6)
        self.assertFalse(bool(separated.active.item()))
        self.assertAlmostEqual(float(penetrating.distance.item()), -0.25, places=6)
        self.assertTrue(bool(penetrating.active.item()))
        self.assertEqual(values(penetrating.normal), [[1.0, 0.0, 0.0]])
        witness_gap = (
            (penetrating.point_b - penetrating.point_a)
            * penetrating.normal
        ).sum(axis=-1)
        self.assertAlmostEqual(
            float(witness_gap.item()),
            float(penetrating.distance.item()),
            places=6,
        )

    def test_fixed_pair_layout(self):
        self.assertEqual(all_pairs(3).indices, ((0, 1), (0, 2), (1, 2)))
        self.assertEqual(values(fixed_pairs([(2, 0)], geometry_count=3).tensor()), [[2, 0]])
        with self.assertRaises(ValueError):
            fixed_pairs([(0, 1), (1, 0)])


class TestSmoothContact(unittest.TestCase):
    params = ContactParams(
        stiffness=100.0,
        damping=2.0,
        friction=0.5,
        penetration_smoothing=1e-5,
        force_smoothing=1e-6,
        velocity_smoothing=0.1,
    )

    def test_inactive_is_zero_and_active_forces_are_equal_opposite(self):
        geometry = sphere_plane(
            Tensor([[0.0, 0.0, 2.0], [0.0, 0.0, 0.5]]),
            1.0,
            Tensor([0.0, 0.0, 0.0]),
            Tensor([0.0, 0.0, 1.0]),
        )
        forces = smooth_contact(
            geometry,
            Tensor([[0.0, 0.0, 0.0], [1.0, 0.0, -2.0]]),
            Tensor.zeros(2, 3),
            self.params,
        )
        self.assertEqual(values(forces.force_a[0]), [0.0, 0.0, 0.0])
        summed = values(forces.force_a + forces.force_b)
        self.assertEqual(summed, [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        active_force = values(forces.force_a[1])
        self.assertLess(active_force[0], 0.0)
        self.assertGreater(active_force[2], 0.0)
        self.assertTrue(all(math.isfinite(x) for row in values(forces.force_a) for x in row))

    def test_contact_gradient_matches_finite_difference_away_from_activation(self):
        params = ContactParams(
            stiffness=30.0,
            damping=0.0,
            friction=0.0,
            penetration_smoothing=0.02,
            force_smoothing=1e-5,
        )
        height = Tensor([0.7])
        center = Tensor.stack(Tensor.zeros(1), Tensor.zeros(1), height, dim=-1)
        geometry = sphere_plane(
            center, 1.0, Tensor([0.0, 0.0, 0.0]), Tensor([0.0, 0.0, 1.0])
        )
        output = smooth_contact(geometry, Tensor.zeros(1, 3), Tensor.zeros(1, 3), params)
        output.force_a[:, 2].sum().backward()
        analytical = height.grad.item()

        def force_at(z: float) -> float:
            query = sphere_plane(
                Tensor([[0.0, 0.0, z]]),
                1.0,
                Tensor([0.0, 0.0, 0.0]),
                Tensor([0.0, 0.0, 1.0]),
            )
            return smooth_contact(query, Tensor.zeros(1, 3), Tensor.zeros(1, 3), params).force_a[0, 2].item()

        step = 1e-3
        finite_difference = (force_at(0.7 + step) - force_at(0.7 - step)) / (2 * step)
        self.assertAlmostEqual(analytical, finite_difference, delta=2e-2)

    def test_documented_near_contact_gradient_inside_active_region(self):
        # Stay 1 mm inside contact and use a 0.1 mm finite-difference step so
        # both samples remain on the same activity-mask branch.
        params = ContactParams(
            stiffness=30.0,
            damping=0.0,
            friction=0.0,
            penetration_smoothing=0.01,
            force_smoothing=1e-5,
        )
        height = Tensor([0.999])
        geometry = sphere_plane(
            Tensor.stack(Tensor.zeros(1), Tensor.zeros(1), height, dim=-1),
            1.0,
            Tensor([0.0, 0.0, 0.0]),
            Tensor([0.0, 0.0, 1.0]),
        )
        force = smooth_contact(
            geometry,
            Tensor.zeros(1, 3),
            Tensor.zeros(1, 3),
            params,
        ).force_a[0, 2]
        force.backward()
        analytical = float(height.grad.item())

        def evaluate(value: float) -> float:
            query = sphere_plane(
                Tensor([[0.0, 0.0, value]]),
                1.0,
                Tensor([0.0, 0.0, 0.0]),
                Tensor([0.0, 0.0, 1.0]),
            )
            return float(
                smooth_contact(
                    query,
                    Tensor.zeros(1, 3),
                    Tensor.zeros(1, 3),
                    params,
                ).force_a[0, 2].item()
            )

        epsilon = 1e-4
        finite_difference = (
            evaluate(0.999 + epsilon) - evaluate(0.999 - epsilon)
        ) / (2 * epsilon)
        self.assertAlmostEqual(analytical, finite_difference, delta=5e-2)


if __name__ == "__main__":
    unittest.main()
