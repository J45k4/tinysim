import math
from pathlib import Path
import unittest

from tinygrad import Tensor, dtypes

from tinysim import (
    ActuatorSpec,
    BodySpec,
    GeomSpec,
    JointSpec,
    ModelSpec,
    Simulator,
    UnsupportedModelError,
    compile_model,
    load_mjcf,
)
from tinysim.constraint import contact_system
from tinysim.collision import ContactParams
from tinysim.dynamics import bias_forces, mass_matrix
from tinysim.kinematics import forward_kinematics
from tinysim.state import State, validate_state


def pendulum() -> ModelSpec:
    return ModelSpec(
        bodies=[BodySpec("bob", inertia=(0.0, 0.0, 0.0), com=(0.0, 0.0, -1.0))],
        joints=[JointSpec("hinge", 0, axis=(0.0, 1.0, 0.0))],
        actuators=[ActuatorSpec("motor", "hinge")],
        timestep=0.01,
    )


class TestModelValidation(unittest.TestCase):
    def test_rejects_nonfinite_python_contact_parameters(self):
        for kwargs in (
            {"stiffness": math.nan},
            {"friction": math.inf},
            {"penetration_smoothing": math.nan},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, "finite"):
                ContactParams(**kwargs)

    def test_rejects_malformed_vectors(self):
        with self.assertRaisesRegex(ValueError, "COM must contain 3"):
            compile_model(
                ModelSpec(
                    [BodySpec("body", com=(0.0, 0.0))],
                    [JointSpec("joint", 0)],
                )
            )
        with self.assertRaisesRegex(ValueError, "axis must contain 3"):
            compile_model(
                ModelSpec(
                    [BodySpec("body")],
                    [JointSpec("joint", 0, axis=(1.0, 0.0))],
                )
            )

    def test_rejects_nonfinite_actuator(self):
        model = pendulum()
        with self.assertRaisesRegex(ValueError, "finite and nonnegative"):
            compile_model(
                ModelSpec(
                    model.bodies,
                    model.joints,
                    [ActuatorSpec("motor", "hinge", gain=math.nan)],
                )
            )

    def test_rejects_dead_damping_on_motor_and_velocity_actuators(self):
        model = pendulum()
        for kind in ("motor", "velocity"):
            with self.subTest(kind=kind), self.assertRaisesRegex(
                UnsupportedModelError, "damping is only defined for position"
            ):
                compile_model(
                    ModelSpec(
                        model.bodies,
                        model.joints,
                        [
                            ActuatorSpec(
                                "drive",
                                "hinge",
                                kind=kind,
                                damping=0.1,
                            )
                        ],
                    )
                )

    def test_rejects_nonfloating_model_dtype(self):
        with self.assertRaisesRegex(TypeError, "float32 or float64"):
            compile_model(pendulum(), dtype=dtypes.int)

    def test_generalized_integrator_rejects_nonfinite_timestep(self):
        simulator = Simulator.compile(pendulum())
        state = simulator.make_state()
        from tinysim.integrator import semi_implicit_euler

        with self.assertRaisesRegex(ValueError, "finite and positive"):
            semi_implicit_euler(
                simulator.model,
                state,
                state.qvel.zeros_like(),
                timestep=math.nan,
            )

    def test_rejects_zero_generalized_inertia(self):
        with self.assertRaisesRegex(UnsupportedModelError, "zero generalized inertia"):
            compile_model(
                ModelSpec(
                    [BodySpec("point", inertia=(0.0, 0.0, 0.0), com=(0.0, 0.0, 0.0))],
                    [JointSpec("hinge", 0)],
                )
            )


class TestRuntimeValidation(unittest.TestCase):
    def test_default_simulator_preserves_control_gradient(self):
        simulator = Simulator.compile(pendulum(), worlds=1)
        control = Tensor([[1.0]])
        result = simulator.step(simulator.make_state(), control)
        result.qpos.sum().backward()
        self.assertIsNotNone(control.grad)
        self.assertGreater(abs(float(control.grad.item())), 1e-8)

    def test_state_dtype_must_match_model(self):
        model = compile_model(pendulum())
        state = State(
            Tensor.zeros(1, 1, dtype=dtypes.float64),
            Tensor.zeros(1, 1, dtype=dtypes.float64),
            Tensor.zeros(1, 1, dtype=dtypes.float64),
            Tensor.zeros(1, dtype=dtypes.float64),
        )
        with self.assertRaisesRegex(TypeError, "model dtype"):
            validate_state(model, state)

    def test_public_kinematics_and_dynamics_require_model_dtype(self):
        model = compile_model(pendulum())
        qpos = Tensor.zeros(1, 1, dtype=dtypes.float64)
        qvel = Tensor.zeros(1, 1, dtype=dtypes.float64)
        with self.assertRaisesRegex(TypeError, "compiled model"):
            forward_kinematics(model, qpos)
        with self.assertRaisesRegex(TypeError, "compiled model"):
            mass_matrix(model, qpos)
        with self.assertRaisesRegex(TypeError, "compiled model"):
            bias_forces(model, qpos, qvel)

    def test_constraint_world_shape_is_exact(self):
        with self.assertRaisesRegex(ValueError, r"shape \(2, 1\)"):
            contact_system(
                Tensor.eye(1).unsqueeze(0).expand(2, 1, 1),
                Tensor.ones(2, 1, 1),
                Tensor.zeros(1, 1),
                Tensor.zeros(1, 1),
                Tensor.zeros(1, 1),
                timestep=0.01,
            )

    def test_constraint_rejects_nonsquare_mass_and_wrong_contact_count(self):
        with self.assertRaisesRegex(ValueError, "square"):
            contact_system(
                Tensor.zeros(1, 2, 3),
                Tensor.zeros(1, 1, 3),
                Tensor.zeros(1, 1),
                Tensor.zeros(1, 1),
                Tensor.zeros(1, 1),
                timestep=0.01,
            )

    def test_constraint_requires_boolean_activity(self):
        with self.assertRaisesRegex(TypeError, "boolean"):
            contact_system(
                Tensor.eye(1).unsqueeze(0),
                Tensor.ones(1, 1, 1),
                Tensor.zeros(1, 1),
                Tensor.zeros(1, 1),
                Tensor.ones(1, 1),
                timestep=0.01,
            )
        with self.assertRaisesRegex(ValueError, r"shape \(1, 2\)"):
            contact_system(
                Tensor.eye(1).unsqueeze(0),
                Tensor.zeros(1, 2, 1),
                Tensor.zeros(1, 1),
                Tensor.zeros(1, 1),
                Tensor.zeros(1, 1),
                timestep=0.01,
            )

    def test_imported_and_direct_pendulum_step_match(self):
        imported = load_mjcf(Path(__file__).parent / "fixtures" / "pendulum.xml")
        direct = ModelSpec(
            name="pendulum",
            bodies=[
                BodySpec(
                    "link",
                    mass=2.0,
                    inertia=(0.2, 0.2, 0.02),
                    com=(0.0, 0.0, -0.5),
                )
            ],
            joints=[
                JointSpec(
                    "hinge",
                    0,
                    axis=(0.0, 1.0, 0.0),
                    pos=(0.0, 0.0, 1.0),
                    damping=0.05,
                )
            ],
            actuators=[
                ActuatorSpec(
                    "drive",
                    0,
                    gain=2.0,
                    control_range=(-1.0, 1.0),
                )
            ],
            geoms=[
                GeomSpec(
                    "floor",
                    -1,
                    "plane",
                    (0.0, 0.0, 0.1),
                    friction=1.0,
                ),
                GeomSpec(
                    "rod",
                    0,
                    "capsule",
                    (0.05, 0.5),
                    pos=(0.0, 0.0, -0.5),
                    friction=1.0,
                ),
            ],
            gravity=(0.0, 0.0, -9.8),
            timestep=0.01,
        )
        imported_sim = Simulator.compile(imported.spec, worlds=1)
        direct_sim = Simulator.compile(direct, worlds=1)
        imported_model, direct_model = imported_sim.model, direct_sim.model
        for field in (
            "bodies",
            "joints",
            "actuators",
            "body_parent",
            "body_depth",
            "depth_bodies",
            "reverse_depth_bodies",
            "joint_for_body",
            "joint_qpos",
            "joint_dof",
            "joint_nq",
            "joint_nv",
            "dof_joint",
            "dof_body",
            "dof_axis_index",
            "dof_angular",
            "actuator_dof",
            "geoms",
            "collision_pairs",
            "contact",
            "timestep",
        ):
            with self.subTest(field=field):
                self.assertEqual(
                    getattr(imported_model, field),
                    getattr(direct_model, field),
                )
        for field in (
            "body_mass",
            "body_inertia",
            "body_com",
            "joint_axis",
            "joint_pos",
            "joint_quat",
            "dof_damping",
            "dof_limit",
            "dof_identity",
            "actuator_gain",
            "actuator_damping",
            "actuator_control_range",
            "actuator_force_range",
            "geom_size",
            "geom_pos",
            "geom_quat",
            "geom_friction",
            "gravity",
        ):
            with self.subTest(field=field):
                self.assertTrue(
                    getattr(imported_model, field)
                    .isclose(getattr(direct_model, field))
                    .all()
                    .item()
                )
        imported_state = imported_sim.make_state()
        direct_state = direct_sim.make_state()
        angle = Tensor([[0.25]])
        imported_state = State(angle, imported_state.qvel, imported_state.ctrl, imported_state.time)
        direct_state = State(angle, direct_state.qvel, direct_state.ctrl, direct_state.time)
        control = Tensor([[0.1]])
        imported_next = imported_sim.step(imported_state, control)
        direct_next = direct_sim.step(direct_state, control)
        self.assertTrue(imported_next.qpos.isclose(direct_next.qpos).all().item())
        self.assertTrue(imported_next.qvel.isclose(direct_next.qvel).all().item())


if __name__ == "__main__":
    unittest.main()
