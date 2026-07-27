"""Validation and static-topology compilation."""

from dataclasses import dataclass
import math

from tinygrad import Device, Tensor, dtypes

from .model import (
    ActuatorSpec,
    BodySpec,
    ContactSpec,
    GeomSpec,
    JointSpec,
    ModelSpec,
    SUPPORTED_COLLISION_PAIRS,
    UnsupportedModelError,
)


@dataclass(frozen=True)
class CompiledModel:
    name: str
    bodies: tuple[BodySpec, ...]
    joints: tuple[JointSpec, ...]
    actuators: tuple[ActuatorSpec, ...]
    body_parent: tuple[int, ...]
    body_depth: tuple[int, ...]
    depth_bodies: tuple[tuple[int, ...], ...]
    reverse_depth_bodies: tuple[tuple[int, ...], ...]
    joint_for_body: tuple[int, ...]
    joint_qpos: tuple[int, ...]
    joint_dof: tuple[int, ...]
    joint_nq: tuple[int, ...]
    joint_nv: tuple[int, ...]
    dof_joint: tuple[int, ...]
    dof_body: tuple[int, ...]
    dof_axis_index: tuple[int, ...]
    dof_angular: tuple[bool, ...]
    actuator_dof: tuple[int, ...]
    geoms: tuple[GeomSpec, ...]
    collision_pairs: tuple[tuple[int, int], ...]
    contact: ContactSpec
    body_mass: Tensor
    body_inertia: Tensor
    body_com: Tensor
    joint_axis: Tensor
    joint_pos: Tensor
    joint_quat: Tensor
    dof_damping: Tensor
    dof_limit: Tensor
    dof_identity: Tensor
    actuator_gain: Tensor
    actuator_damping: Tensor
    actuator_control_range: Tensor
    actuator_force_range: Tensor
    geom_size: Tensor
    geom_pos: Tensor
    geom_quat: Tensor
    geom_friction: Tensor
    gravity: Tensor
    timestep: float
    device: str
    dtype: object

    @property
    def nbody(self) -> int:
        return len(self.bodies)

    @property
    def njoint(self) -> int:
        return len(self.joints)

    @property
    def nq(self) -> int:
        return sum(self.joint_nq)

    @property
    def nv(self) -> int:
        return len(self.dof_joint)

    @property
    def nu(self) -> int:
        return len(self.actuators)

    @property
    def nconstraint(self) -> int:
        if self.contact.mode != "constraint":
            return 0
        return len(self.collision_pairs) + 2 * sum(
            joint.limit is not None for joint in self.joints
        )


def _finite(values: tuple[float, ...], label: str) -> None:
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{label} must contain finite values")


def _vector(values: tuple[float, ...], size: int, label: str) -> None:
    if len(values) != size:
        raise ValueError(f"{label} must contain {size} values")
    _finite(values, label)


def _unit(values: tuple[float, ...], size: int, label: str) -> tuple[float, ...]:
    _vector(values, size, label)
    length = math.sqrt(sum(value * value for value in values))
    if length < 1e-12:
        raise ValueError(f"{label} must be nonzero")
    return tuple(value / length for value in values)


def compile_model(
    spec: ModelSpec,
    *,
    device: str | None = None,
    dtype: object = dtypes.float32,
) -> CompiledModel:
    """Validates and flattens a static reduced-coordinate body tree.

    Body order is topological. Every body has exactly one explicit joint;
    quaternion joints use distinct qpos and velocity widths.
    """
    if dtype not in (dtypes.float32, dtypes.float64):
        raise TypeError("TinySim models require float32 or float64 dtype")
    if not spec.bodies:
        raise ValueError("model must contain at least one body")
    if spec.timestep <= 0 or not math.isfinite(spec.timestep):
        raise ValueError("timestep must be finite and positive")
    _vector(spec.gravity, 3, "gravity")
    names: set[str] = set()
    body_depth: list[int] = []
    for index, body in enumerate(spec.bodies):
        if not body.name or body.name in names:
            raise ValueError(f"body names must be unique and nonempty: {body.name!r}")
        names.add(body.name)
        if body.parent >= index or body.parent < -1:
            raise UnsupportedModelError("bodies must be in parent-before-child order")
        body_depth.append(0 if body.parent == -1 else body_depth[body.parent] + 1)
        if body.mass <= 0 or not math.isfinite(body.mass):
            raise ValueError(f"body {body.name!r} mass must be finite and positive")
        _vector(body.inertia, 3, f"body {body.name!r} inertia")
        if any(value < 0 or not math.isfinite(value) for value in body.inertia):
            raise ValueError(f"body {body.name!r} diagonal inertia must be nonnegative")
        if any(body.inertia[i] > sum(body.inertia) - body.inertia[i] + 1e-12 for i in range(3)):
            raise ValueError(f"body {body.name!r} inertia violates the triangle inequality")
        _vector(body.com, 3, f"body {body.name!r} COM")

    normalized_joints: list[JointSpec] = []
    joint_for_body = [-1] * len(spec.bodies)
    joint_names: dict[str, int] = {}
    joint_qpos: list[int] = []
    joint_dof: list[int] = []
    joint_nq: list[int] = []
    joint_nv: list[int] = []
    dof_joint: list[int] = []
    dof_body: list[int] = []
    dof_axis_index: list[int] = []
    dof_angular: list[bool] = []
    damping: list[float] = []
    qpos_size = 0
    for index, joint in enumerate(spec.joints):
        if joint.kind not in ("fixed", "hinge", "slide", "ball", "free"):
            raise UnsupportedModelError(f"unsupported joint kind {joint.kind!r}")
        if joint.body < 0 or joint.body >= len(spec.bodies):
            raise ValueError(f"joint {joint.name!r} references invalid body")
        if joint_for_body[joint.body] != -1:
            raise UnsupportedModelError("each body must have exactly one joint")
        if not joint.name or joint.name in joint_names:
            raise ValueError(f"joint names must be unique and nonempty: {joint.name!r}")
        if joint.damping < 0 or not math.isfinite(joint.damping):
            raise ValueError(f"joint {joint.name!r} damping must be nonnegative")
        if joint.limit is not None:
            _vector(joint.limit, 2, f"joint {joint.name!r} limit")
            if joint.limit[0] > joint.limit[1]:
                raise ValueError(f"joint {joint.name!r} limit is reversed")
            if joint.kind not in ("hinge", "slide"):
                raise UnsupportedModelError(
                    "joint limits currently require a scalar hinge or slide"
                )
        axis = _unit(joint.axis, 3, f"joint {joint.name!r} axis")
        quat = _unit(joint.quat, 4, f"joint {joint.name!r} quaternion")
        _vector(joint.pos, 3, f"joint {joint.name!r} position")
        if joint.kind == "hinge":
            body = spec.bodies[joint.body]
            rotational = sum(inertia * component * component for inertia, component in zip(body.inertia, axis))
            lever = (
                axis[1] * body.com[2] - axis[2] * body.com[1],
                axis[2] * body.com[0] - axis[0] * body.com[2],
                axis[0] * body.com[1] - axis[1] * body.com[0],
            )
            if rotational + body.mass * sum(value * value for value in lever) <= 1e-12:
                raise UnsupportedModelError(
                    f"joint {joint.name!r} has zero generalized inertia on its own body"
                )
        if joint.kind in ("ball", "free") and any(value <= 1e-12 for value in spec.bodies[joint.body].inertia):
            raise UnsupportedModelError(
                f"joint {joint.name!r} requires positive rotational inertia on its own body"
            )
        normalized_joints.append(
            JointSpec(
                joint.name,
                joint.body,
                joint.kind,
                axis,
                joint.pos,
                quat,
                joint.damping,
                joint.limit,
            )
        )
        joint_for_body[joint.body] = index
        joint_names[joint.name] = index
        nq, nv = {
            "fixed": (0, 0),
            "hinge": (1, 1),
            "slide": (1, 1),
            "ball": (4, 3),
            "free": (7, 6),
        }[joint.kind]
        joint_qpos.append(qpos_size if nq else -1)
        joint_dof.append(len(dof_joint) if nv else -1)
        joint_nq.append(nq)
        joint_nv.append(nv)
        qpos_size += nq
        for component in range(nv):
            dof_joint.append(index)
            dof_body.append(joint.body)
            if joint.kind == "free":
                dof_axis_index.append(component if component < 3 else component - 3)
                dof_angular.append(component >= 3)
            elif joint.kind == "ball":
                dof_axis_index.append(component)
                dof_angular.append(True)
            else:
                dof_axis_index.append(0)
                dof_angular.append(joint.kind == "hinge")
            damping.append(joint.damping)
    if any(index == -1 for index in joint_for_body):
        missing = [spec.bodies[i].name for i, index in enumerate(joint_for_body) if index == -1]
        raise UnsupportedModelError(f"every body needs one joint; missing {missing}")

    normalized_actuators: list[ActuatorSpec] = []
    actuator_dof: list[int] = []
    actuator_names: set[str] = set()
    for actuator in spec.actuators:
        if not actuator.name or actuator.name in actuator_names:
            raise ValueError(f"actuator names must be unique and nonempty: {actuator.name!r}")
        actuator_names.add(actuator.name)
        joint_index = joint_names.get(actuator.joint, -1) if isinstance(actuator.joint, str) else actuator.joint
        if not isinstance(joint_index, int) or not 0 <= joint_index < len(spec.joints):
            raise ValueError(f"actuator {actuator.name!r} references invalid joint")
        dof = joint_dof[joint_index]
        if dof == -1:
            raise UnsupportedModelError("actuators cannot target fixed joints")
        if joint_nv[joint_index] != 1:
            raise UnsupportedModelError(
                "actuators on ball/free joints require an explicit transmission"
            )
        if actuator.kind not in ("motor", "position", "velocity"):
            raise UnsupportedModelError(f"unsupported actuator kind {actuator.kind!r}")
        if (
            actuator.gain < 0
            or actuator.damping < 0
            or not math.isfinite(actuator.gain)
            or not math.isfinite(actuator.damping)
        ):
            raise ValueError("actuator gain and damping must be finite and nonnegative")
        if actuator.kind != "position" and actuator.damping != 0.0:
            raise UnsupportedModelError(
                f"actuator {actuator.name!r}: damping is only defined for "
                "position servos"
            )
        for limits, label in ((actuator.control_range, "control"), (actuator.force_range, "force")):
            if limits is not None and (
                len(limits) != 2
                or limits[0] > limits[1]
                or not all(math.isfinite(x) for x in limits)
            ):
                raise ValueError(f"actuator {label} range is invalid")
        normalized_actuators.append(actuator)
        actuator_dof.append(dof)

    normalized_geoms: list[GeomSpec] = []
    geom_names: dict[str, int] = {}
    size_width = 3
    expected_sizes = {"plane": 3, "sphere": 1, "capsule": 2, "box": 3}
    for index, geom in enumerate(spec.geoms):
        if geom.kind not in expected_sizes:
            raise UnsupportedModelError(f"unsupported geom kind {geom.kind!r}")
        if geom.body < -1 or geom.body >= len(spec.bodies):
            raise ValueError(f"geom {geom.name!r} references invalid body")
        if geom.name is not None:
            if not geom.name or geom.name in geom_names:
                raise ValueError(f"geom names must be unique and nonempty: {geom.name!r}")
            geom_names[geom.name] = index
        _vector(geom.size, expected_sizes[geom.kind], f"geom {geom.name!r} size")
        if geom.kind == "plane":
            if any(value < 0 for value in geom.size):
                raise ValueError(f"geom {geom.name!r} plane size must be nonnegative")
        elif any(value <= 0 for value in geom.size):
            raise ValueError(f"geom {geom.name!r} size must be positive")
        _vector(geom.pos, 3, f"geom {geom.name!r} position")
        quat = _unit(geom.quat, 4, f"geom {geom.name!r} quaternion")
        if geom.friction < 0 or not math.isfinite(geom.friction):
            raise ValueError(f"geom {geom.name!r} friction must be finite and nonnegative")
        normalized_geoms.append(
            GeomSpec(geom.name, geom.body, geom.kind, geom.size, geom.pos, quat, geom.friction)
        )

    supported_pairs = SUPPORTED_COLLISION_PAIRS

    def pair_index(value: int | str) -> int:
        if isinstance(value, str):
            if value not in geom_names:
                raise ValueError(f"collision pair references unknown geom {value!r}")
            return geom_names[value]
        if not isinstance(value, int) or value < 0 or value >= len(normalized_geoms):
            raise ValueError(f"collision pair references invalid geom {value!r}")
        return value

    excluded_pairs = {
        (min(pair_index(a), pair_index(b)), max(pair_index(a), pair_index(b)))
        for a, b in spec.collision_exclusions
    }
    if any(a == b for a, b in excluded_pairs):
        raise ValueError("collision exclusion cannot contain the same geom twice")
    raw_pairs = (
        [(pair_index(a), pair_index(b)) for a, b in spec.collision_pairs]
        if spec.collision_pairs
        else [
            (a, b)
            for a in range(len(normalized_geoms))
            for b in range(a + 1, len(normalized_geoms))
            if normalized_geoms[a].body != normalized_geoms[b].body
            and not (normalized_geoms[a].body == normalized_geoms[b].body == -1)
            and (a, b) not in excluded_pairs
            and (
                (
                    normalized_geoms[a].kind,
                    normalized_geoms[b].kind,
                )
                in supported_pairs
                or (
                    normalized_geoms[b].kind,
                    normalized_geoms[a].kind,
                )
                in supported_pairs
            )
        ]
    )
    collision_pairs: list[tuple[int, int]] = []
    seen_pairs: set[tuple[int, int]] = set()
    for a, b in raw_pairs:
        if a == b:
            raise ValueError("collision pair cannot contain the same geom twice")
        key = (min(a, b), max(a, b))
        if key in excluded_pairs:
            raise ValueError(f"collision pair {(a, b)} is also excluded")
        if key in seen_pairs:
            raise ValueError(f"duplicate collision pair {(a, b)}")
        kinds = (normalized_geoms[a].kind, normalized_geoms[b].kind)
        if kinds not in supported_pairs and kinds[::-1] not in supported_pairs:
            raise UnsupportedModelError(f"unsupported collision pair {kinds}")
        collision_pairs.append((a, b))
        seen_pairs.add(key)

    contact = spec.contact
    if contact.mode not in ("none", "smooth", "constraint"):
        raise ValueError(f"unsupported contact mode {contact.mode!r}")
    finite_contact = (
        contact.margin,
        contact.stiffness,
        contact.damping,
        contact.friction,
        contact.penetration_smoothing,
        contact.force_smoothing,
        contact.velocity_smoothing,
        contact.stabilization,
        contact.regularization,
    )
    if not all(math.isfinite(value) for value in finite_contact):
        raise ValueError("contact parameters must be finite")
    if contact.margin < 0 or min(finite_contact[1:]) < 0:
        raise ValueError("contact parameters must be nonnegative")
    if min(
        contact.penetration_smoothing,
        contact.force_smoothing,
        contact.velocity_smoothing,
    ) <= 0:
        raise ValueError("contact smoothing scales must be positive")
    if contact.solver_iterations < 1:
        raise ValueError("solver iterations must be positive")

    maximum_depth = max(body_depth)
    depth_bodies = tuple(
        tuple(index for index, depth in enumerate(body_depth) if depth == level)
        for level in range(maximum_depth + 1)
    )

    selected_device = device or Device.DEFAULT
    tensor = lambda values: Tensor(values, dtype=dtype, device=selected_device)
    control_ranges = [a.control_range or (-math.inf, math.inf) for a in normalized_actuators]
    force_ranges = [a.force_range or (-math.inf, math.inf) for a in normalized_actuators]
    return CompiledModel(
        name=spec.name,
        bodies=tuple(spec.bodies),
        joints=tuple(normalized_joints),
        actuators=tuple(normalized_actuators),
        body_parent=tuple(body.parent for body in spec.bodies),
        body_depth=tuple(body_depth),
        depth_bodies=depth_bodies,
        reverse_depth_bodies=tuple(reversed(depth_bodies)),
        joint_for_body=tuple(joint_for_body),
        joint_qpos=tuple(joint_qpos),
        joint_dof=tuple(joint_dof),
        joint_nq=tuple(joint_nq),
        joint_nv=tuple(joint_nv),
        dof_joint=tuple(dof_joint),
        dof_body=tuple(dof_body),
        dof_axis_index=tuple(dof_axis_index),
        dof_angular=tuple(dof_angular),
        actuator_dof=tuple(actuator_dof),
        geoms=tuple(normalized_geoms),
        collision_pairs=tuple(collision_pairs),
        contact=contact,
        body_mass=tensor([body.mass for body in spec.bodies]),
        body_inertia=tensor([body.inertia for body in spec.bodies]),
        body_com=tensor([body.com for body in spec.bodies]),
        joint_axis=tensor([joint.axis for joint in normalized_joints]),
        joint_pos=tensor([joint.pos for joint in normalized_joints]),
        joint_quat=tensor([joint.quat for joint in normalized_joints]),
        dof_damping=(
            tensor(damping)
            if damping
            else Tensor.zeros(0, dtype=dtype, device=selected_device)
        ),
        dof_limit=tensor([
            normalized_joints[joint].limit or (-math.inf, math.inf)
            for joint in dof_joint
        ]) if dof_joint else Tensor.zeros(
            0, 2, dtype=dtype, device=selected_device
        ),
        dof_identity=tensor([
            [1.0 if row == column else 0.0 for column in range(len(dof_joint))]
            for row in range(len(dof_joint))
        ]) if dof_joint else Tensor.zeros(
            0, 0, dtype=dtype, device=selected_device
        ),
        actuator_gain=(
            tensor([a.gain for a in normalized_actuators])
            if normalized_actuators
            else Tensor.zeros(0, dtype=dtype, device=selected_device)
        ),
        actuator_damping=(
            tensor([a.damping for a in normalized_actuators])
            if normalized_actuators
            else Tensor.zeros(0, dtype=dtype, device=selected_device)
        ),
        actuator_control_range=(
            tensor(control_ranges)
            if control_ranges
            else Tensor.zeros(0, 2, dtype=dtype, device=selected_device)
        ),
        actuator_force_range=(
            tensor(force_ranges)
            if force_ranges
            else Tensor.zeros(0, 2, dtype=dtype, device=selected_device)
        ),
        geom_size=tensor([
            tuple(geom.size) + (0.0,) * (size_width - len(geom.size))
            for geom in normalized_geoms
        ]) if normalized_geoms else Tensor.zeros(
            0, size_width, dtype=dtype, device=selected_device
        ),
        geom_pos=(
            tensor([geom.pos for geom in normalized_geoms])
            if normalized_geoms
            else Tensor.zeros(0, 3, dtype=dtype, device=selected_device)
        ),
        geom_quat=(
            tensor([geom.quat for geom in normalized_geoms])
            if normalized_geoms
            else Tensor.zeros(0, 4, dtype=dtype, device=selected_device)
        ),
        geom_friction=(
            tensor([geom.friction for geom in normalized_geoms])
            if normalized_geoms
            else Tensor.zeros(0, dtype=dtype, device=selected_device)
        ),
        gravity=tensor(spec.gravity),
        timestep=spec.timestep,
        device=selected_device,
        dtype=dtype,
    )
