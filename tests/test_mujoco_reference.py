from dataclasses import replace
import unittest
from unittest import mock

from tinygrad import Tensor, dtypes

from tinysim.articulated_contact import contacts
from tinysim.compile import compile_model
from tinysim.importers import load_mjcf
from tinysim.kinematics import forward_kinematics
from tinysim.model import (
    ActuatorSpec,
    BodySpec,
    ContactSpec,
    GeomSpec,
    JointSpec,
    ModelSpec,
)
from tinysim.reference.mujoco import (
    MujocoCompatibilityError,
    MujocoReference,
    MujocoUnavailableError,
    ReferenceSnapshot,
    model_to_mjcf,
    mujoco_status,
)


PENDULUM_XML = """
<mujoco model="reference_pendulum">
  <option timestep="0.01" gravity="0 0 -9.81"/>
  <worldbody>
    <body name="link">
      <inertial pos="0 0 -0.7" mass="1.2" diaginertia="0.08 0.12 0.08"/>
      <joint name="hinge" type="hinge" axis="0 1 0"/>
    </body>
  </worldbody>
  <actuator><motor name="motor" joint="hinge" gear="1.7"/></actuator>
</mujoco>
"""


def two_link() -> ModelSpec:
    return ModelSpec(
        name="reference_two_link",
        bodies=[
            BodySpec(
                "one",
                mass=1.3,
                inertia=(0.1, 0.2, 0.1),
                com=(0.0, 0.0, -0.6),
            ),
            BodySpec(
                "two",
                parent=0,
                mass=0.7,
                inertia=(0.1, 0.15, 0.1),
                com=(0.0, 0.0, -0.5),
            ),
        ],
        joints=[
            JointSpec("one_hinge", 0, axis=(0.0, 1.0, 0.0)),
            JointSpec(
                "two_hinge",
                1,
                axis=(0.0, 1.0, 0.0),
                pos=(0.0, 0.0, -1.4),
            ),
        ],
        actuators=[
            ActuatorSpec("one_motor", "one_hinge", gain=1.1),
            ActuatorSpec("two_motor", "two_hinge", gain=0.8),
        ],
        timestep=0.005,
    )


class EchoBackend:
    """Test double for adapter plumbing; it is not a physics oracle."""

    name = "test-echo-not-mujoco"

    def __init__(self):
        self.result: ReferenceSnapshot | None = None
        self.inputs = None

    def evaluate(self, qpos, qvel, control):
        self.inputs = (qpos, qvel, control)
        if self.result is None:
            raise RuntimeError("test backend has no result")
        return self.result


class TestMujocoAdapterContracts(unittest.TestCase):
    def test_import_is_optional_and_unavailable_error_is_explicit(self):
        namespace = type("Namespace", (), {"__path__": ("/checkout/mujoco",)})()
        with mock.patch(
            "tinysim.reference.mujoco.importlib.import_module",
            return_value=namespace,
        ):
            status = mujoco_status()
            self.assertFalse(status.available)
            self.assertIn("not canonical bindings", status.reason or "")
            with self.assertRaisesRegex(
                MujocoUnavailableError, "not canonical bindings"
            ):
                MujocoReference(load_mjcf(PENDULUM_XML).spec)

    def test_serializer_preserves_tree_frames_and_force_semantics(self):
        xml = model_to_mjcf(two_link())
        self.assertIn('body name="two" pos="0 0 -1.3999999999999999', xml)
        self.assertIn('joint name="two_hinge" type="hinge"', xml)
        self.assertIn('general name="one_motor"', xml)
        self.assertIn('gainprm="1.1000000000000001"', xml)
        self.assertNotIn("<motor", xml)

    def test_echo_backend_proves_comparison_wiring_not_mujoco(self):
        backend = EchoBackend()
        reference = MujocoReference(
            load_mjcf(PENDULUM_XML).spec, backend=backend
        )
        expected = reference.tinysim_snapshot((0.35,), (-0.2,), (0.4,))
        backend.result = expected
        comparison = reference.compare((0.35,), (-0.2,), (0.4,))
        self.assertEqual(
            backend.inputs, ((0.35,), (-0.2,), (0.4,))
        )
        self.assertEqual(comparison.errors.maximum, 0.0)
        comparison.assert_within(0.0)

        backend.result = replace(
            expected,
            acceleration=(expected.acceleration[0] + 0.25,),
        )
        comparison = reference.compare((0.35,), (-0.2,), (0.4,))
        self.assertAlmostEqual(comparison.errors.acceleration, 0.25)
        with self.assertRaisesRegex(AssertionError, "acceleration=0.25"):
            comparison.assert_within(1e-9)

    def test_backend_shape_is_checked_before_comparison(self):
        backend = EchoBackend()
        reference = MujocoReference(two_link(), backend=backend)
        expected = reference.tinysim_snapshot(
            (0.1, -0.2), (0.3, -0.4), (0.0, 0.0)
        )
        backend.result = replace(expected, mass_matrix=((1.0,),))
        with self.assertRaisesRegex(
            MujocoCompatibilityError,
            r"mass_matrix must have shape \[2, 2\]",
        ):
            reference.compare(
                (0.1, -0.2), (0.3, -0.4), (0.0, 0.0)
            )

    def test_zero_inertia_is_an_explicit_unsupported_boundary(self):
        model = ModelSpec(
            [BodySpec("body", inertia=(0.0, 0.0, 0.0))],
            [JointSpec("slide", 0, kind="slide")],
        )
        with self.assertRaisesRegex(
            MujocoCompatibilityError, "strictly positive diagonal inertia"
        ):
            model_to_mjcf(model)

    def test_ball_and_free_velocity_basis_mismatch_is_rejected(self):
        for kind in ("ball", "free"):
            with self.subTest(kind=kind):
                model = ModelSpec(
                    [BodySpec("body", inertia=(0.2, 0.3, 0.4))],
                    [JointSpec(f"{kind}_joint", 0, kind=kind)],
                )
                with self.assertRaisesRegex(
                    MujocoCompatibilityError,
                    r"different angular basis.*explicit basis transform",
                ):
                    MujocoReference(model, backend=EchoBackend())


_STATUS = mujoco_status()


@unittest.skipUnless(
    _STATUS.available,
    f"canonical MuJoCo bindings unavailable: {_STATUS.reason}",
)
class TestRealMujocoReference(unittest.TestCase):
    """These tests are validation only when canonical bindings execute them."""

    def test_imported_pendulum_intermediates_and_step(self):
        reference = MujocoReference(load_mjcf(PENDULUM_XML).spec)
        comparison = reference.compare((0.35,), (-0.2,), (0.4,))
        self.assertTrue(
            comparison.backend_name.startswith("mujoco-python-")
        )
        comparison.assert_within(1e-9)

    def test_two_link_intermediates_and_step(self):
        reference = MujocoReference(two_link())
        comparison = reference.compare(
            (0.3, -0.45), (0.7, -0.2), (0.15, -0.1)
        )
        self.assertTrue(
            comparison.backend_name.startswith("mujoco-python-")
        )
        comparison.assert_within(1e-9)

    def test_two_link_float32_tolerance(self):
        comparison = MujocoReference(
            two_link(), dtype=dtypes.float32
        ).compare(
            (0.3, -0.45), (0.7, -0.2), (0.15, -0.1)
        )
        comparison.assert_within(2e-6)

    def test_sphere_plane_contact_geometry_and_normal_jacobian(self):
        import mujoco
        import numpy

        xml = """
        <mujoco><option gravity="0 0 0"/><worldbody>
          <geom name="floor" type="plane" size="0 0 1"/>
          <body name="ball" pos="0 0 .2"><freejoint/>
            <geom name="sphere" type="sphere" size=".25"/>
          </body>
        </worldbody></mujoco>
        """
        mj_model = mujoco.MjModel.from_xml_string(xml)
        mj_data = mujoco.MjData(mj_model)
        mujoco.mj_forward(mj_model, mj_data)
        self.assertEqual(mj_data.ncon, 1)
        mj_contact = mj_data.contact[0]
        body_id = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_BODY, "ball"
        )
        point_jacobian = numpy.zeros((3, mj_model.nv))
        rotation_jacobian = numpy.zeros((3, mj_model.nv))
        mujoco.mj_jac(
            mj_model,
            mj_data,
            point_jacobian,
            rotation_jacobian,
            mj_contact.pos,
            body_id,
        )
        mj_normal_jacobian = mj_contact.frame[:3] @ point_jacobian

        model = compile_model(
            ModelSpec(
                [BodySpec("ball", inertia=(0.1, 0.1, 0.1))],
                [JointSpec("root", 0, "free")],
                geoms=[
                    GeomSpec("floor", -1, "plane", (0.0, 0.0, 1.0)),
                    GeomSpec("sphere", 0, "sphere", (0.25,)),
                ],
                contact=ContactSpec(mode="constraint"),
                gravity=(0.0, 0.0, 0.0),
            ),
            dtype=dtypes.float64,
        )
        qpos = Tensor(
            [[0.0, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0]],
            dtype=dtypes.float64,
        )
        batch = contacts(model, forward_kinematics(model, qpos))
        geometry = batch.geometry
        self.assertAlmostEqual(
            float(geometry.distance.item()), float(mj_contact.dist), places=12
        )
        for actual, expected in zip(
            geometry.normal.tolist()[0][0],
            mj_contact.frame[:3],
            strict=True,
        ):
            self.assertAlmostEqual(actual, expected, places=12)
        for actual, expected in zip(
            geometry.position.tolist()[0][0],
            mj_contact.pos,
            strict=True,
        ):
            self.assertAlmostEqual(actual, expected, places=12)
        for actual, expected in zip(
            batch.constraint_jacobian.tolist()[0][0],
            mj_normal_jacobian,
            strict=True,
        ):
            self.assertAlmostEqual(actual, expected, places=12)


if __name__ == "__main__":
    unittest.main()
