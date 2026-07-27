import unittest

from tinygrad import Tensor

from tinysim.articulated_contact import contacts
from tinysim.compile import compile_model
from tinysim.kinematics import forward_kinematics
from tinysim.model import (
    BodySpec,
    ContactSpec,
    GeomSpec,
    JointSpec,
    ModelSpec,
)
from tinysim.simulation import Simulator, step
from tinysim.state import State, make_state


def ball_model(mode: str):
    return compile_model(
        ModelSpec(
            [BodySpec("ball", mass=1.0, inertia=(0.1, 0.1, 0.1))],
            [JointSpec("free", 0, "free")],
            timestep=0.002,
            geoms=[
                GeomSpec("ball_geom", 0, "sphere", (0.25,)),
                GeomSpec("floor", -1, "plane", (0.0, 0.0, 0.1)),
            ],
            contact=ContactSpec(
                mode=mode,  # type: ignore[arg-type]
                stiffness=5_000.0,
                damping=100.0,
                friction=0.0,
                solver_iterations=8,
            ),
        )
    )


def penetrating_state(model, *, velocity=-1.0):
    state = make_state(model)
    return State(
        Tensor([[0.0, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0]]),
        Tensor([[0.0, 0.0, velocity, 0.0, 0.0, 0.0]]),
        state.ctrl,
        state.time,
        state.constraint_impulse,
    )


class TestArticulatedContact(unittest.TestCase):
    def test_compiler_builds_fixed_pair_and_depth_schedule(self):
        model = ball_model("smooth")
        self.assertEqual(model.collision_pairs, ((0, 1),))
        self.assertEqual(model.body_depth, (0,))
        self.assertEqual(model.depth_bodies, ((0,),))
        self.assertEqual(model.reverse_depth_bodies, ((0,),))

    def test_contact_jacobian_maps_free_translation(self):
        model = ball_model("smooth")
        state = penetrating_state(model, velocity=0.0)
        batch = contacts(model, forward_kinematics(model, state.qpos))
        self.assertEqual(batch.geometry.distance.shape, (1, 1))
        self.assertAlmostEqual(float(batch.geometry.distance.item()), -0.05, places=6)
        jacobian = batch.constraint_jacobian.tolist()[0][0]
        self.assertAlmostEqual(float(jacobian[2]), 1.0, places=6)
        for index in (0, 1, 3, 4, 5):
            self.assertAlmostEqual(float(jacobian[index]), 0.0, places=6)

    def test_compiler_and_contact_admit_box_box_pair(self):
        model = compile_model(
            ModelSpec(
                bodies=[
                    BodySpec("a", inertia=(0.1, 0.1, 0.1)),
                    BodySpec("b", inertia=(0.1, 0.1, 0.1)),
                ],
                joints=[
                    JointSpec("free_a", 0, "free"),
                    JointSpec("free_b", 1, "free"),
                ],
                geoms=[
                    GeomSpec("box_a", 0, "box", (0.5, 0.5, 0.5)),
                    GeomSpec("box_b", 1, "box", (0.5, 0.5, 0.5)),
                ],
                contact=ContactSpec(mode="smooth"),
            )
        )
        qpos = Tensor(
            [[
                0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                1.5, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
            ]]
        )
        batch = contacts(model, forward_kinematics(model, qpos))
        self.assertEqual(model.collision_pairs, ((0, 1),))
        self.assertAlmostEqual(
            float(batch.geometry.distance.item()),
            0.5,
            places=6,
        )

    def test_smooth_contact_is_in_articulated_step_and_differentiable(self):
        model = ball_model("smooth")
        state = penetrating_state(model)
        result = step(model, state)
        self.assertGreater(float(result.qvel[0, 2].item()), -1.0)

        height = Tensor([0.2])
        qpos = Tensor.stack(
            Tensor.zeros(1),
            Tensor.zeros(1),
            height,
            Tensor.ones(1),
            Tensor.zeros(1),
            Tensor.zeros(1),
            Tensor.zeros(1),
            dim=-1,
        )
        differentiable = State(
            qpos,
            state.qvel,
            state.ctrl,
            state.time,
            state.constraint_impulse,
        )
        step(model, differentiable).qvel[:, 2].sum().backward()
        self.assertIsNotNone(height.grad)
        self.assertTrue(height.grad.isfinite().all().item())

    def test_constraint_contact_warm_starts_and_jit_replays(self):
        model = ball_model("constraint")
        state = penetrating_state(model)
        result = step(model, state)
        self.assertGreater(float(result.qvel[0, 2].item()), 0.0)
        self.assertIsNotNone(result.constraint_impulse)
        self.assertGreater(float(result.constraint_impulse.item()), 0.0)

        simulator = Simulator(model, 1)
        state = penetrating_state(model)
        for _ in range(3):
            state = simulator.inference_step(state)
        self.assertTrue(state.qpos.isfinite().all().item())
        self.assertTrue(state.qvel.isfinite().all().item())

    def test_constraint_contact_supports_static_weight(self):
        model = ball_model("constraint")
        state = penetrating_state(model, velocity=0.0)
        state = State(
            Tensor([[0.0, 0.0, 0.25, 1.0, 0.0, 0.0, 0.0]]),
            state.qvel,
            state.ctrl,
            state.time,
            state.constraint_impulse,
            state.parameters,
        )
        result = step(model, state)
        self.assertAlmostEqual(float(result.qvel[0, 2].item()), 0.0, places=5)
        self.assertGreater(float(result.constraint_impulse.item()), 0.0)

    def test_joint_limit_is_a_warm_started_unilateral_constraint(self):
        model = compile_model(
            ModelSpec(
                [BodySpec("link", inertia=(1.0, 1.0, 1.0))],
                [
                    JointSpec(
                        "hinge",
                        0,
                        "hinge",
                        limit=(-0.5, 0.5),
                    )
                ],
                gravity=(0.0, 0.0, 0.0),
                timestep=0.01,
                contact=ContactSpec(
                    mode="constraint",
                    solver_iterations=4,
                    stabilization=0.2,
                ),
            )
        )
        base = make_state(model)
        state = State(
            Tensor([[0.6]]),
            Tensor([[1.0]]),
            base.ctrl,
            base.time,
            base.constraint_impulse,
        )
        result = step(model, state)
        self.assertLess(float(result.qvel.item()), 0.0)
        self.assertEqual(result.constraint_impulse.shape, (1, 2))
        self.assertGreater(float(result.constraint_impulse[0, 1].item()), 0.0)


if __name__ == "__main__":
    unittest.main()
