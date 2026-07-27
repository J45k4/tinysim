"""Small directly-authored robotics models used by examples and benchmarks."""

from .model import (
    ActuatorSpec,
    BodySpec,
    ContactSpec,
    GeomSpec,
    JointSpec,
    ModelSpec,
)


def quadruped_spec(*, contact_mode: str = "smooth") -> ModelSpec:
    """Returns a compact free-base, four-leg locomotion test model."""

    bodies = [
        BodySpec(
            "trunk",
            mass=5.0,
            inertia=(0.20, 0.35, 0.40),
        )
    ]
    joints = [JointSpec("root", 0, "free")]
    actuators = []
    geoms = [
        GeomSpec("trunk_geom", 0, "box", (0.35, 0.20, 0.12)),
        GeomSpec("floor", -1, "plane", (0.0, 0.0, 0.1)),
    ]
    hips = (
        (0.25, 0.18, 0.0),
        (0.25, -0.18, 0.0),
        (-0.25, 0.18, 0.0),
        (-0.25, -0.18, 0.0),
    )
    for index, position in enumerate(hips):
        body = len(bodies)
        bodies.append(
            BodySpec(
                f"leg_{index}",
                parent=0,
                mass=0.5,
                inertia=(0.015, 0.015, 0.008),
                com=(0.0, 0.0, -0.22),
            )
        )
        joint_name = f"hip_{index}"
        joints.append(
            JointSpec(
                joint_name,
                body,
                "hinge",
                axis=(0.0, 1.0, 0.0),
                pos=position,
                damping=0.1,
                limit=(-1.2, 1.2),
            )
        )
        actuators.append(
            ActuatorSpec(
                f"hip_motor_{index}",
                joint_name,
                gain=4.0,
                control_range=(-1.0, 1.0),
                force_range=(-8.0, 8.0),
            )
        )
        geoms.append(
            GeomSpec(
                f"leg_geom_{index}",
                body,
                "capsule",
                (0.06, 0.22),
                pos=(0.0, 0.0, -0.22),
                friction=0.8,
            )
        )
    return ModelSpec(
        bodies=bodies,
        joints=joints,
        actuators=actuators,
        geoms=geoms,
        timestep=0.002,
        name="tiny_quadruped",
        contact=ContactSpec(
            mode=contact_mode,  # type: ignore[arg-type]
            stiffness=8_000.0,
            damping=150.0,
            friction=1.0,
            solver_iterations=12,
        ),
    )
