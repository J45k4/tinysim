import unittest

from tinygrad import Tensor

from tinysim.environment import DomainRandomization, Environment
from tinysim.model import ActuatorSpec, BodySpec, JointSpec, ModelSpec
from tinysim.simulation import Simulator
from tinysim.state import State
from tinysim.robots import quadruped_spec


def pendulum_spec() -> ModelSpec:
    return ModelSpec(
        name="pendulum",
        bodies=[
            BodySpec(
                name="bob",
                parent=-1,
                mass=1.0,
                inertia=(0.01, 1.01, 1.01),
                com=(0.0, 0.0, -1.0),
            )
        ],
        joints=[
            JointSpec(
                name="hinge",
                body=0,
                kind="hinge",
                axis=(0.0, 1.0, 0.0),
                damping=0.05,
            )
        ],
        actuators=[ActuatorSpec(name="motor", joint="hinge")],
        timestep=0.01,
    )


class TestSimulator(unittest.TestCase):
    def test_eager_step_is_batched(self):
        simulator = Simulator.compile(pendulum_spec(), worlds=4)
        initial = simulator.make_state()
        state = State(
            qpos=Tensor.full((4, 1), 0.2),
            qvel=initial.qvel,
            ctrl=initial.ctrl,
            time=initial.time,
        )
        result = simulator.step(state, Tensor.zeros(4, 1))
        self.assertEqual(result.qpos.shape, (4, 1))
        self.assertTrue(bool((result.time == 0.01).all().item()))
        self.assertLess(float(result.qpos[0, 0].item()), 0.2)

    def test_inference_replay_matches_eager_and_is_detached(self):
        simulator = Simulator.compile(pendulum_spec(), worlds=2)
        qpos = Tensor.full((2, 1), 0.2).realize()
        zero_velocity = Tensor.zeros(2, 1).realize()
        control = Tensor.full((2, 1), 0.1).realize()
        time = Tensor.zeros(2).realize()
        eager_state = State(qpos, zero_velocity, control, time)
        expected = simulator.step(eager_state, control)
        actual = None
        for _ in range(3):
            state = State(
                qpos.clone().realize(),
                zero_velocity.clone().realize(),
                control.clone().realize(),
                time.clone().realize(),
            )
            actual = simulator.inference_step(state, state.ctrl)
        self.assertIsNotNone(actual)
        assert actual is not None
        self.assertTrue(actual.qpos.isclose(expected.qpos).all().item())
        self.assertTrue(actual.qvel.isclose(expected.qvel).all().item())
        actual.qpos.sum().backward()
        self.assertIsNone(control.grad)

    def test_environment_contract(self):
        environment = Environment(
            Simulator.compile(pendulum_spec(), worlds=3)
        )
        state, observation = environment.reset()
        self.assertEqual(observation.shape, (3, 2))
        transition = environment.step(state, Tensor.zeros(3, 1))
        self.assertEqual(transition.reward.shape, (3,))
        self.assertEqual(transition.terminated.shape, (3,))

    def test_environment_rejects_nonboolean_termination(self):
        environment = Environment(
            Simulator.compile(pendulum_spec(), worlds=2),
            termination=lambda state: Tensor.full((2,), 0.5),
        )
        state, _ = environment.reset()
        with self.assertRaisesRegex(TypeError, "boolean"):
            environment.step(state, Tensor.zeros(2, 1))

    def test_domain_randomization_and_masked_reset_stay_batched(self):
        environment = Environment(
            Simulator.compile(pendulum_spec(), worlds=4),
            randomization=DomainRandomization(
                mass_scale=(0.8, 1.2),
                actuator_gain_scale=(0.5, 1.5),
            ),
        )
        state, _ = environment.reset(seed=12)
        self.assertEqual(state.parameters.body_mass.shape, (4, 1))
        self.assertGreater(
            len(set(float(value[0]) for value in state.parameters.body_mass.tolist())),
            1,
        )
        advanced = environment.step(state, Tensor.ones(4, 1)).state
        reset, _ = environment.reset(
            advanced,
            Tensor([True, False, True, False]),
            seed=13,
        )
        self.assertEqual(float(reset.time[0].item()), 0.0)
        self.assertGreater(float(reset.time[1].item()), 0.0)
        self.assertEqual(float(reset.qpos[0, 0].item()), 0.0)
        self.assertNotEqual(
            float(reset.parameters.body_mass[0, 0].item()),
            float(advanced.parameters.body_mass[0, 0].item()),
        )

    def test_same_device_policy_rollout(self):
        environment = Environment(
            Simulator.compile(pendulum_spec(), worlds=3)
        )
        state, _ = environment.reset()
        result = environment.rollout(
            state,
            lambda observation: observation[:, :1] * 0.0,
            steps=3,
            inference=True,
        )
        self.assertEqual(result.qpos.shape, (3, 1))
        self.assertAlmostEqual(float(result.time[0].item()), 0.03, places=6)

    def test_representative_locomotion_topology(self):
        model = Simulator.compile(
            quadruped_spec(contact_mode="none"), worlds=2
        ).model
        self.assertEqual(model.nbody, 5)
        self.assertEqual((model.nq, model.nv, model.nu), (11, 10, 4))
        self.assertGreaterEqual(len(model.depth_bodies), 2)

    def test_representative_locomotion_smooth_contact_step(self):
        simulator = Simulator.compile(quadruped_spec(), worlds=3)
        state = simulator.make_state()
        state.qpos[:, 2].assign(0.55)
        advanced = simulator.step(state, simulator.zeros_control())
        self.assertEqual(advanced.qpos.shape, (3, 11))
        self.assertEqual(advanced.qvel.shape, (3, 10))
        self.assertTrue(advanced.qpos.isfinite().all().item())
        self.assertTrue(advanced.qvel.isfinite().all().item())

    def test_locomotion_policy_actuates_joints(self):
        simulator = Simulator.compile(quadruped_spec(), worlds=2)
        environment = Environment(simulator)
        state, _ = environment.reset()
        state.qpos[:, 2].assign(0.55)
        pattern = Tensor([[0.25, -0.25, -0.25, 0.25]]).expand(2, 4)
        advanced = environment.rollout(
            state,
            lambda observation: observation[:, :1] * 0.0 + pattern,
            steps=2,
            inference=False,
        )
        self.assertGreater(
            float(advanced.qvel[:, 6:].abs().max().item()),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
