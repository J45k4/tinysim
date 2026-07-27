"""Static articulated benchmark models.

These are performance fixtures, not claims of biomechanical fidelity.  Their
topology and sizes are recorded in every benchmark result.
"""

from tinysim.model import ActuatorSpec, BodySpec, JointSpec, ModelSpec


def quadruped_like_spec() -> ModelSpec:
    """Return a fixed-base, four-leg tree with eight actuated hinges."""

    bodies = [BodySpec("torso", mass=5.0, inertia=(0.3, 0.5, 0.6))]
    joints = [JointSpec("root", 0, "fixed")]
    actuators: list[ActuatorSpec] = []
    hip_positions = (
        (0.35, 0.22, 0.0),
        (0.35, -0.22, 0.0),
        (-0.35, 0.22, 0.0),
        (-0.35, -0.22, 0.0),
    )
    for leg, hip_position in enumerate(hip_positions):
        hip = len(bodies)
        bodies.append(
            BodySpec(
                f"hip_{leg}",
                parent=0,
                mass=0.7,
                inertia=(0.03, 0.03, 0.02),
                com=(0.0, 0.0, -0.15),
            )
        )
        joints.append(
            JointSpec(
                f"hip_joint_{leg}",
                hip,
                "hinge",
                axis=(0.0, 1.0, 0.0),
                pos=hip_position,
                damping=0.05,
            )
        )
        knee = len(bodies)
        bodies.append(
            BodySpec(
                f"shin_{leg}",
                parent=hip,
                mass=0.5,
                inertia=(0.02, 0.02, 0.01),
                com=(0.0, 0.0, -0.18),
            )
        )
        joints.append(
            JointSpec(
                f"knee_joint_{leg}",
                knee,
                "hinge",
                axis=(0.0, 1.0, 0.0),
                pos=(0.0, 0.0, -0.3),
                damping=0.05,
            )
        )
        actuators.extend(
            (
                ActuatorSpec(f"hip_motor_{leg}", f"hip_joint_{leg}"),
                ActuatorSpec(f"knee_motor_{leg}", f"knee_joint_{leg}"),
            )
        )
    return ModelSpec(
        bodies,
        joints,
        actuators,
        timestep=0.002,
        name="quadruped_like_8dof",
    )


def humanoid_scale_spec() -> ModelSpec:
    """Return a fixed-base 12-DoF humanoid-shaped articulated tree.

    Twelve DoFs are enough to exercise a materially larger dense solve than
    the quadruped fixture while remaining capturable by tinygrad's CPU backend,
    whose generated programs currently accept at most 31 buffer arguments.
    """

    bodies = [BodySpec("pelvis", mass=8.0, inertia=(0.4, 0.35, 0.45))]
    joints = [JointSpec("root", 0, "fixed")]
    actuators: list[ActuatorSpec] = []

    def add_link(
        name: str,
        parent: int,
        position: tuple[float, float, float],
        axis: tuple[float, float, float],
        mass: float,
        com: tuple[float, float, float],
    ) -> int:
        body = len(bodies)
        inertia = max(0.01, mass * 0.025)
        bodies.append(
            BodySpec(
                name,
                parent=parent,
                mass=mass,
                inertia=(inertia, inertia, inertia),
                com=com,
            )
        )
        joint_name = f"{name}_joint"
        joints.append(
            JointSpec(
                joint_name,
                body,
                "hinge",
                axis=axis,
                pos=position,
                damping=0.05,
            )
        )
        actuators.append(ActuatorSpec(f"{name}_motor", joint_name))
        return body

    spine = add_link("spine_0", 0, (0.0, 0.0, 0.2), (0.0, 1.0, 0.0), 3.0, (0.0, 0.0, 0.12))
    chest = add_link("spine_1", spine, (0.0, 0.0, 0.25), (1.0, 0.0, 0.0), 4.0, (0.0, 0.0, 0.12))
    head = len(bodies)
    bodies.append(
        BodySpec(
            "head",
            parent=chest,
            mass=3.5,
            inertia=(0.09, 0.09, 0.09),
            com=(0.0, 0.0, 0.08),
        )
    )
    joints.append(JointSpec("head_fixed", head, "fixed", pos=(0.0, 0.0, 0.3)))

    for side, sign in (("left", 1.0), ("right", -1.0)):
        shoulder = add_link(
            f"{side}_shoulder_pitch",
            chest,
            (0.0, 0.22 * sign, 0.22),
            (0.0, 1.0, 0.0),
            1.2,
            (0.0, 0.1 * sign, 0.0),
        )
        add_link(
            f"{side}_elbow",
            shoulder,
            (0.0, 0.3 * sign, 0.0),
            (1.0, 0.0, 0.0),
            0.7,
            (0.0, 0.1 * sign, 0.0),
        )

    for side, sign in (("left", 1.0), ("right", -1.0)):
        hip_pitch = add_link(
            f"{side}_hip_pitch",
            0,
            (0.0, 0.12 * sign, -0.08),
            (0.0, 1.0, 0.0),
            3.0,
            (0.0, 0.0, -0.18),
        )
        knee = add_link(
            f"{side}_knee",
            hip_pitch,
            (0.0, 0.0, -0.4),
            (0.0, 1.0, 0.0),
            2.2,
            (0.0, 0.0, -0.18),
        )
        add_link(
            f"{side}_ankle_pitch",
            knee,
            (0.0, 0.0, -0.36),
            (0.0, 1.0, 0.0),
            0.7,
            (0.0, 0.0, -0.06),
        )

    return ModelSpec(
        bodies,
        joints,
        actuators,
        timestep=0.002,
        name="humanoid_scale_14body_12dof",
    )
