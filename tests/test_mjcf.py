from pathlib import Path
import math
import unittest

from tinysim.compile import compile_model
from tinysim.importers import load_mjcf
from tinysim.model import ModelSpec, UnsupportedModelError


FIXTURES = Path(__file__).parent / "fixtures"
EXAMPLES = Path(__file__).parents[1] / "examples" / "verification"


class TestMJCFImporter(unittest.TestCase):
    def test_pendulum_file(self):
        imported = load_mjcf(FIXTURES / "pendulum.xml")
        self.assertEqual(imported.spec.name, "pendulum")
        self.assertEqual(imported.spec.timestep, 0.01)
        self.assertEqual(imported.spec.gravity, (0.0, 0.0, -9.8))
        self.assertEqual(imported.spec.bodies[0].mass, 2.0)
        self.assertEqual(imported.spec.bodies[0].com, (0.0, 0.0, -0.5))
        self.assertEqual(imported.spec.joints[0].pos, (0.0, 0.0, 1.0))
        self.assertEqual(imported.spec.joints[0].axis, (0.0, 1.0, 0.0))
        self.assertEqual(
            [geom.kind for geom in imported.spec.geoms],
            ["plane", "capsule"],
        )
        self.assertEqual(imported.spec.geoms[0].body, -1)
        self.assertEqual(imported.spec.geoms[1].body, 0)
        self.assertEqual(imported.keyframes, (("displaced", (0.25,)),))

    def test_nested_body_offsets_are_rebased_to_joint_frames(self):
        imported = load_mjcf(FIXTURES / "hierarchy.xml")
        self.assertEqual([body.parent for body in imported.spec.bodies], [-1, 0])
        self.assertEqual(imported.spec.joints[0].pos, (1.2, 0.0, 0.0))
        self.assertAlmostEqual(imported.spec.joints[1].pos[0], 1.9)
        self.assertEqual(imported.spec.joints[1].pos[1:], (0.0, 0.0))
        self.assertEqual(imported.spec.bodies[1].com, (0.0, 0.0, 0.0))
        self.assertEqual(imported.spec.geoms[0].pos, (0.2, 0.0, 0.0))
        self.assertAlmostEqual(imported.spec.geoms[1].pos[0], 0.2)
        self.assertEqual(imported.spec.geoms[1].pos[1:], (0.0, 0.0))
        compiled = compile_model(imported.spec)
        self.assertEqual(compiled.body_parent, (-1, 0))

    def test_actuators_resolve_joint_names_and_parameters(self):
        imported = load_mjcf(FIXTURES / "pendulum.xml")
        actuator = imported.spec.actuators[0]
        self.assertEqual(actuator.joint, 0)
        self.assertEqual(actuator.kind, "motor")
        self.assertEqual(actuator.gain, 2.0)
        self.assertEqual(actuator.control_range, (-1.0, 1.0))
        self.assertEqual(compile_model(imported.spec).actuator_dof, (0,))

    def test_motor_force_range_is_transformed_to_generalized_force(self):
        imported = load_mjcf(
            """
            <mujoco><worldbody><body name="slider">
              <inertial mass="1" diaginertia="1 1 1"/>
              <joint name="slide" type="slide"/>
            </body></worldbody><actuator>
              <motor name="drive" joint="slide" gear="2"
                     forcelimited="true" forcerange="-1 1"/>
            </actuator></mujoco>
            """
        )
        actuator = imported.spec.actuators[0]
        self.assertEqual(actuator.gain, 2.0)
        self.assertEqual(actuator.force_range, (-2.0, 2.0))
        with self.assertRaisesRegex(
            UnsupportedModelError, "negative motor gear"
        ):
            load_mjcf(
                """
                <mujoco><worldbody><body name="slider">
                  <inertial mass="1" diaginertia="1 1 1"/>
                  <joint name="slide" type="slide"/>
                </body></worldbody><actuator>
                  <motor name="drive" joint="slide" gear="-2"/>
                </actuator></mujoco>
                """
            )

    def test_position_actuator_and_named_defaults(self):
        imported = load_mjcf(
            """
            <mujoco>
              <default>
                <joint damping="0.2"/>
                <default class="servo">
                  <joint axis="0 1 0"/>
                  <geom type="sphere" size="0.1"/>
                  <position kp="8" kv="0.5" ctrlrange="-2 2"/>
                </default>
              </default>
              <worldbody>
                <body name="arm" childclass="servo">
                  <inertial mass="1" diaginertia="1 1 1"/>
                  <joint name="joint"/>
                  <geom name="visual"/>
                </body>
              </worldbody>
              <actuator>
                <position name="servo" joint="joint" class="servo"/>
              </actuator>
            </mujoco>
            """
        )
        self.assertEqual(imported.spec.joints[0].axis, (0.0, 1.0, 0.0))
        self.assertEqual(imported.spec.joints[0].damping, 0.2)
        self.assertEqual(imported.spec.geoms[0].kind, "sphere")
        actuator = imported.spec.actuators[0]
        self.assertEqual((actuator.gain, actuator.damping), (8.0, 0.5))
        self.assertEqual(actuator.control_range, (-2.0, 2.0))

    def test_body_without_joint_becomes_explicit_fixed_relationship(self):
        imported = load_mjcf(
            """
            <mujoco><worldbody><body name="fixed" pos="1 2 3">
              <inertial mass="1" diaginertia="1 1 1"/>
            </body></worldbody></mujoco>
            """
        )
        self.assertEqual(imported.spec.joints[0].kind, "fixed")
        self.assertEqual(imported.spec.joints[0].pos, (1.0, 2.0, 3.0))

    def test_free_joint_and_quaternion_keyframe(self):
        imported = load_mjcf(
            """
            <mujoco><worldbody><body name="arm">
              <inertial mass="1" diaginertia="1 1 1"/>
              <joint name="free" type="free"/>
            </body></worldbody>
            <keyframe><key name="pose" qpos="1 2 3 1 0 0 0"/></keyframe>
            </mujoco>
            """
        )
        self.assertEqual(imported.spec.joints[0].kind, "free")
        self.assertEqual(imported.keyframes[0][1], (1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0))
        compiled = compile_model(imported.spec)
        self.assertEqual((compiled.nq, compiled.nv), (7, 6))

    def test_explicit_contact_pair_and_body_exclusion(self):
        imported = load_mjcf(
            """
            <mujoco>
              <worldbody>
                <body name="a"><inertial mass="1" diaginertia="1 1 1"/>
                  <geom name="a_ball" type="sphere" size="0.2"/>
                </body>
                <body name="b"><inertial mass="1" diaginertia="1 1 1"/>
                  <geom name="b_ball" type="sphere" size="0.2"/>
                </body>
                <body name="c"><inertial mass="1" diaginertia="1 1 1"/>
                  <geom name="c_ball" type="sphere" size="0.2"/>
                </body>
              </worldbody>
              <contact>
                <pair geom1="a_ball" geom2="c_ball"/>
                <exclude body1="a" body2="b"/>
              </contact>
            </mujoco>
            """
        )
        self.assertEqual(imported.spec.collision_pairs, ((0, 2), (1, 2)))
        self.assertEqual(imported.spec.collision_exclusions, ((0, 1),))
        self.assertEqual(
            compile_model(imported.spec).collision_pairs,
            ((0, 2), (1, 2)),
        )

    def test_explicit_pair_adds_to_ordinary_candidates(self):
        imported = load_mjcf(
            """
            <mujoco><worldbody>
              <body name="a"><inertial mass="1" diaginertia="1 1 1"/>
                <geom name="ga" type="sphere" size="0.1"/></body>
              <body name="b"><inertial mass="1" diaginertia="1 1 1"/>
                <geom name="gb" type="sphere" size="0.1"/></body>
              <body name="c"><inertial mass="1" diaginertia="1 1 1"/>
                <geom name="gc" type="sphere" size="0.1"/></body>
            </worldbody><contact><pair geom1="ga" geom2="gc"/></contact>
            </mujoco>
            """
        )
        self.assertEqual(
            compile_model(imported.spec).collision_pairs,
            ((0, 1), (0, 2), (1, 2)),
        )

    def test_modelspec_from_mjcf_public_constructor(self):
        spec = ModelSpec.from_mjcf(Path(__file__).parent / "fixtures" / "pendulum.xml")
        self.assertEqual(spec.name, "pendulum")
        self.assertEqual([geom.kind for geom in spec.geoms], ["plane", "capsule"])

    def test_degree_converts_hinge_range_but_not_keyframe_qpos(self):
        imported = load_mjcf(
            """
            <mujoco><worldbody><body name="arm">
              <inertial mass="1" diaginertia="1 1 1"/>
              <joint name="hinge" limited="true" range="-90 90"/>
            </body></worldbody>
            <keyframe><key name="quarter" qpos="90"/></keyframe>
            </mujoco>
            """
        )
        self.assertAlmostEqual(imported.spec.joints[0].limit[0], -math.pi / 2)
        self.assertAlmostEqual(imported.spec.joints[0].limit[1], math.pi / 2)
        self.assertEqual(imported.keyframes[0][1][0], 90.0)

    def test_explicit_radian_hinge_ranges_are_preserved(self):
        imported = load_mjcf(
            """
            <mujoco><compiler angle="radian"/>
            <worldbody><body name="arm">
              <inertial mass="1" diaginertia="1 1 1"/>
              <joint name="hinge" limited="true" range="-1 1"/>
            </body></worldbody></mujoco>
            """
        )
        self.assertEqual(imported.spec.joints[0].limit, (-1.0, 1.0))

    def test_friction_uses_mjcf_sliding_default_and_rejects_leaky_terms(self):
        default = load_mjcf(
            """
            <mujoco><worldbody><geom name="floor" type="plane" size="0 0 1"/>
            </worldbody></mujoco>
            """
        )
        self.assertEqual(default.spec.geoms[0].friction, 1.0)
        with self.assertRaisesRegex(
            UnsupportedModelError, "torsional and rolling friction"
        ):
            load_mjcf(
                """
                <mujoco><worldbody>
                  <geom name="floor" type="plane" size="0 0 1"
                        friction="1 0.005 0.0001"/>
                </worldbody></mujoco>
                """
            )

    def test_imported_examples_compile(self):
        expected = {
            "pendulum.xml": (1, 1, 1),
            "cartpole.xml": (2, 2, 1),
            "two_link.xml": (2, 2, 0),
        }
        for filename, dimensions in expected.items():
            with self.subTest(filename=filename):
                model = compile_model(load_mjcf(EXAMPLES / filename).spec)
                self.assertEqual((model.nq, model.nv, model.nu), dimensions)

    def test_unsupported_feature_is_not_silently_ignored(self):
        with self.assertRaisesRegex(
            UnsupportedModelError,
            r"/mujoco/worldbody/body\[@name='arm'\].*unsupported attribute.*quat",
        ):
            load_mjcf(
                """
                <mujoco><worldbody><body name="arm" quat="1 0 0 0">
                  <inertial mass="1" diaginertia="1 1 1"/>
                </body></worldbody></mujoco>
                """
            )

    def test_unsupported_nested_feature_is_not_silently_ignored(self):
        with self.assertRaisesRegex(
            UnsupportedModelError,
            r"/mujoco/option/flag.*cannot contain child elements",
        ):
            load_mjcf(
                """
                <mujoco>
                  <option><flag gravity="disable"/></option>
                  <worldbody/>
                </mujoco>
                """
            )

    def test_unknown_actuator_joint_reports_actuator_path(self):
        with self.assertRaisesRegex(
            ValueError,
            r"/mujoco/actuator/motor\[@name='drive'\].*unknown joint 'missing'",
        ):
            load_mjcf(
                """
                <mujoco>
                  <worldbody><body name="arm">
                    <inertial mass="1" diaginertia="1 1 1"/>
                    <joint name="hinge"/>
                  </body></worldbody>
                  <actuator><motor name="drive" joint="missing"/></actuator>
                </mujoco>
                """
            )


if __name__ == "__main__":
    unittest.main()
