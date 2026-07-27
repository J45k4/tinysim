import math
import unittest

from tinygrad import Tensor

from tinysim.compile import compile_model
from tinysim.kinematics import forward_kinematics
from tinysim.model import BodySpec, JointSpec, ModelSpec, UnsupportedModelError
from tinysim.spatial import rotate


class TestModelKinematics(unittest.TestCase):
    def test_compiler_preserves_parent_before_child_topology(self):
        model = compile_model(
            ModelSpec(
                bodies=[
                    BodySpec("root", -1),
                    BodySpec("left", 0),
                    BodySpec("right", 0),
                    BodySpec("tip", 1),
                ],
                joints=[
                    JointSpec("root_fixed", 0, "fixed"),
                    JointSpec("left_hinge", 1),
                    JointSpec("right_slide", 2, "slide"),
                    JointSpec("tip_hinge", 3),
                ],
            )
        )
        self.assertEqual(model.body_parent, (-1, 0, 0, 1))
        self.assertEqual(model.nq, 3)

    def test_rejects_non_topological_tree(self):
        with self.assertRaises(UnsupportedModelError):
            compile_model(
                ModelSpec(
                    [BodySpec("child", 1), BodySpec("parent", -1)],
                    [JointSpec("a", 0), JointSpec("b", 1)],
                )
            )

    def test_hinge_and_child_transform(self):
        model = compile_model(
            ModelSpec(
                [
                    BodySpec("upper", -1),
                    BodySpec("lower", 0),
                ],
                [
                    JointSpec("shoulder", 0, "hinge", (0.0, 1.0, 0.0)),
                    JointSpec("elbow", 1, "hinge", (0.0, 1.0, 0.0), pos=(0.0, 0.0, -2.0)),
                ],
            )
        )
        qpos = Tensor([[math.pi / 2, 0.0], [0.0, math.pi / 2]])
        result = forward_kinematics(model, qpos)
        first_tip = result.body_pos.tolist()[0][1]
        self.assertAlmostEqual(float(first_tip[0]), -2.0, places=5)
        self.assertAlmostEqual(float(first_tip[2]), 0.0, places=5)
        local_down = Tensor([[0.0, 0.0, -1.0]])
        second_lower_direction = rotate(result.body_quat[1:2, 1], local_down).tolist()[0]
        self.assertAlmostEqual(float(second_lower_direction[0]), -1.0, places=5)
        self.assertAlmostEqual(float(second_lower_direction[2]), 0.0, places=5)

    def test_slide_is_batched(self):
        model = compile_model(
            ModelSpec(
                [BodySpec("slider")],
                [JointSpec("slide", 0, "slide", (1.0, 0.0, 0.0), pos=(0.0, 2.0, 0.0))],
            )
        )
        positions = forward_kinematics(model, Tensor([[0.0], [1.5], [-2.0]])).body_pos.tolist()
        self.assertEqual(len(positions), 3)
        self.assertAlmostEqual(float(positions[0][0][0]), 0.0)
        self.assertAlmostEqual(float(positions[1][0][0]), 1.5)
        self.assertAlmostEqual(float(positions[2][0][0]), -2.0)
        self.assertAlmostEqual(float(positions[1][0][1]), 2.0)


if __name__ == "__main__":
    unittest.main()
