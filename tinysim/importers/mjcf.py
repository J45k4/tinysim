"""Strict native importer for TinySim's deliberately small MJCF subset."""

from dataclasses import dataclass
import math
from os import PathLike
from pathlib import Path
import xml.etree.ElementTree as ET

from ..model import (
    ActuatorSpec,
    BodySpec,
    GeomSpec,
    JointSpec,
    ModelSpec,
    SUPPORTED_COLLISION_PAIRS,
    UnsupportedModelError,
)


Keyframe = tuple[str, tuple[float, ...]]
_Defaults = dict[str, dict[str, dict[str, str]]]


@dataclass(frozen=True)
class ImportedModel:
    """The runtime authoring spec plus importer-only keyframe metadata."""

    spec: ModelSpec
    keyframes: tuple[Keyframe, ...]


_DEFAULT_TAGS = ("joint", "geom", "motor", "position")
_ALLOWED_DEFAULT_ATTRIBUTES = {
    "joint": {"type", "axis", "pos", "damping", "limited", "range"},
    "geom": {"type", "size", "pos", "quat", "friction"},
    "motor": {"gear", "ctrllimited", "ctrlrange", "forcelimited", "forcerange"},
    "position": {"kp", "kv", "ctrllimited", "ctrlrange", "forcelimited", "forcerange"},
}


def _unsupported(path: str, detail: str) -> None:
    raise UnsupportedModelError(f"{path}: {detail}")


def _check_attributes(element: ET.Element, allowed: set[str], path: str) -> None:
    unsupported = sorted(set(element.attrib) - allowed)
    if unsupported:
        _unsupported(path, f"unsupported attribute(s): {', '.join(unsupported)}")


def _check_leaf(element: ET.Element, path: str) -> None:
    if len(element):
        child = element[0]
        _unsupported(_path(path, child), f"{element.tag!r} cannot contain child elements")


def _numbers(value: str, count: int | None, path: str, attribute: str) -> tuple[float, ...]:
    try:
        result = tuple(float(part) for part in value.split())
    except ValueError as error:
        raise ValueError(f"{path}: {attribute} must contain numbers") from error
    if count is not None and len(result) != count:
        raise ValueError(f"{path}: {attribute} must contain {count} numbers")
    if not all(math.isfinite(part) for part in result):
        raise ValueError(f"{path}: {attribute} must contain finite numbers")
    return result


def _scalar(attributes: dict[str, str], name: str, default: float, path: str) -> float:
    if name not in attributes:
        return default
    return _numbers(attributes[name], 1, path, name)[0]


def _vector(
    attributes: dict[str, str],
    name: str,
    default: tuple[float, ...],
    path: str,
) -> tuple[float, ...]:
    return default if name not in attributes else _numbers(attributes[name], len(default), path, name)


def _path(parent: str, element: ET.Element) -> str:
    result = f"{parent}/{element.tag}"
    if "name" in element.attrib:
        result += f"[@name={element.attrib['name']!r}]"
    elif "class" in element.attrib:
        result += f"[@class={element.attrib['class']!r}]"
    return result


def _copy_class(defaults: _Defaults, name: str) -> dict[str, dict[str, str]]:
    return {tag: dict(defaults[name][tag]) for tag in _DEFAULT_TAGS}


def _parse_defaults(root: ET.Element) -> _Defaults:
    defaults: _Defaults = {"main": {tag: {} for tag in _DEFAULT_TAGS}}
    sections = [child for child in root if child.tag == "default"]
    if len(sections) > 1:
        _unsupported("/mujoco/default", "multiple top-level default sections are unsupported")
    if not sections:
        return defaults

    def visit(element: ET.Element, parent_class: str | None, path: str) -> None:
        _check_attributes(element, {"class"}, path)
        class_name = element.get("class")
        if parent_class is None:
            class_name = class_name or "main"
            if class_name != "main":
                _unsupported(path, "the top-level default must be the main class")
            values = _copy_class(defaults, "main")
        else:
            if class_name is None:
                _unsupported(path, "nested default requires a class name")
            if class_name in defaults:
                raise ValueError(f"{path}: duplicate default class {class_name!r}")
            values = _copy_class(defaults, parent_class)

        for child in element:
            child_path = _path(path, child)
            if child.tag == "default":
                continue
            if child.tag not in _DEFAULT_TAGS:
                _unsupported(child_path, f"default for {child.tag!r} is unsupported")
            _check_attributes(child, _ALLOWED_DEFAULT_ATTRIBUTES[child.tag], child_path)
            _check_leaf(child, child_path)
            values[child.tag].update(child.attrib)
        defaults[class_name] = values
        for child in element:
            if child.tag == "default":
                visit(child, class_name, _path(path, child))

    visit(sections[0], None, "/mujoco/default")
    return defaults


def _effective(
    element: ET.Element,
    tag: str,
    inherited_class: str,
    defaults: _Defaults,
    path: str,
) -> dict[str, str]:
    class_name = element.get("class", inherited_class)
    if class_name not in defaults:
        raise ValueError(f"{path}: unknown default class {class_name!r}")
    attributes = dict(defaults[class_name][tag])
    attributes.update({key: value for key, value in element.attrib.items() if key != "class"})
    return attributes


def _quat(attributes: dict[str, str], path: str) -> tuple[float, float, float, float]:
    values = _vector(attributes, "quat", (1.0, 0.0, 0.0, 0.0), path)
    length = math.sqrt(sum(value * value for value in values))
    if length < 1e-12:
        raise ValueError(f"{path}: quat must be nonzero")
    return tuple(value / length for value in values)  # type: ignore[return-value]


def _geom(
    element: ET.Element,
    body: int,
    offset: tuple[float, float, float],
    inherited_class: str,
    defaults: _Defaults,
    path: str,
) -> GeomSpec:
    _check_attributes(
        element,
        {"name", "type", "size", "pos", "quat", "friction", "class"},
        path,
    )
    _check_leaf(element, path)
    attributes = _effective(element, "geom", inherited_class, defaults, path)
    kind = attributes.get("type", "sphere")
    sizes = {"plane": 3, "sphere": 1, "capsule": 2, "box": 3}
    if kind not in sizes:
        _unsupported(path, f"unsupported geom type {kind!r}")
    if "size" not in attributes:
        raise ValueError(f"{path}: primitive geom requires size")
    size = _numbers(attributes["size"], sizes[kind], path, "size")
    if kind == "plane":
        if any(value < 0 for value in size):
            raise ValueError(f"{path}: plane size must be nonnegative")
    elif any(value <= 0 for value in size):
        raise ValueError(f"{path}: geom size must be positive")
    source_pos = _vector(attributes, "pos", (0.0, 0.0, 0.0), path)
    pos = tuple(source_pos[index] - offset[index] for index in range(3))
    friction = 1.0
    if "friction" in attributes:
        friction_values = _numbers(
            attributes["friction"], 3, path, "friction"
        )
        if friction_values[1:] != (0.0, 0.0):
            _unsupported(
                path,
                "torsional and rolling friction are unsupported; set the "
                "second and third friction components to zero",
            )
        friction = friction_values[0]
    if friction < 0:
        raise ValueError(f"{path}: friction must be nonnegative")
    return GeomSpec(
        element.get("name"),
        body,
        kind,
        size,
        pos,
        _quat(attributes, path),
        friction,
    )


def _range(
    attributes: dict[str, str],
    prefix: str,
    path: str,
) -> tuple[float, float] | None:
    limited_name, range_name = f"{prefix}limited", f"{prefix}range"
    limited = attributes.get(limited_name, "auto")
    if limited not in ("true", "false", "auto"):
        raise ValueError(f"{path}: {limited_name} must be true, false, or auto")
    values = None
    if range_name in attributes:
        parsed = _numbers(attributes[range_name], 2, path, range_name)
        if parsed[0] > parsed[1]:
            raise ValueError(f"{path}: {range_name} lower bound exceeds upper bound")
        values = (parsed[0], parsed[1])
    if limited == "true" and values is None:
        raise ValueError(f"{path}: {limited_name}=true requires {range_name}")
    return None if limited == "false" else values


def _joint_limit(
    attributes: dict[str, str], path: str
) -> tuple[float, float] | None:
    limited = attributes.get("limited", "auto")
    if limited not in ("true", "false", "auto"):
        raise ValueError(f"{path}: limited must be true, false, or auto")
    values = None
    if "range" in attributes:
        parsed = _numbers(attributes["range"], 2, path, "range")
        if parsed[0] > parsed[1]:
            raise ValueError(f"{path}: range lower bound exceeds upper bound")
        values = (parsed[0], parsed[1])
    if limited == "true" and values is None:
        raise ValueError(f"{path}: limited=true requires range")
    return None if limited == "false" else values


def _motor_gain(attributes: dict[str, str], path: str) -> float:
    if "gear" not in attributes:
        return 1.0
    gear = _numbers(attributes["gear"], None, path, "gear")
    if len(gear) == 1 or (
        len(gear) == 6 and all(value == 0.0 for value in gear[1:])
    ):
        if gear[0] < 0:
            _unsupported(
                path,
                "negative motor gear requires signed transmission support",
            )
        return gear[0]
    _unsupported(path, "only scalar joint gear is supported")
    raise AssertionError("unreachable")


def _read_source(source: str | PathLike[str]) -> tuple[str, str]:
    if isinstance(source, PathLike):
        path = Path(source)
        return path.read_text(encoding="utf-8"), str(path)
    if source.lstrip().startswith("<"):
        return source, "<string>"
    path = Path(source)
    return path.read_text(encoding="utf-8"), str(path)


def load_mjcf(source: str | PathLike[str]) -> ImportedModel:
    """Loads XML text or a path using only the supported native MJCF subset.

    TinySim's current body frame is its joint frame. MJCF body and joint
    translations are therefore combined while inertial and geom positions are
    rebased, preserving their world transforms. Rotated body/joint frames are
    rejected until the core model can represent them without loss.
    """
    text, source_name = _read_source(source)
    try:
        root = ET.fromstring(text)
    except ET.ParseError as error:
        raise ValueError(f"{source_name}: invalid MJCF XML: {error}") from error
    if root.tag != "mujoco":
        _unsupported(f"/{root.tag}", "root element must be 'mujoco'")
    _check_attributes(root, {"model"}, "/mujoco")
    defaults = _parse_defaults(root)

    supported_sections = {
        "compiler",
        "option",
        "default",
        "worldbody",
        "contact",
        "actuator",
        "keyframe",
    }
    for child in root:
        if child.tag not in supported_sections:
            _unsupported(_path("/mujoco", child), f"unsupported top-level element {child.tag!r}")

    compilers = [child for child in root if child.tag == "compiler"]
    if len(compilers) > 1:
        _unsupported("/mujoco/compiler", "multiple compiler sections are unsupported")
    angle_unit = "degree"
    if compilers:
        compiler = compilers[0]
        _check_attributes(compiler, {"angle"}, "/mujoco/compiler")
        _check_leaf(compiler, "/mujoco/compiler")
        angle_unit = compiler.get("angle", angle_unit)
        if angle_unit not in ("degree", "radian"):
            raise ValueError("/mujoco/compiler: angle must be degree or radian")

    options = [child for child in root if child.tag == "option"]
    if len(options) > 1:
        _unsupported("/mujoco/option", "multiple option sections are unsupported")
    gravity = (0.0, 0.0, -9.81)
    timestep = 0.002
    if options:
        option = options[0]
        _check_attributes(option, {"gravity", "timestep"}, "/mujoco/option")
        _check_leaf(option, "/mujoco/option")
        gravity = _vector(option.attrib, "gravity", gravity, "/mujoco/option")  # type: ignore[assignment]
        timestep = _scalar(option.attrib, "timestep", timestep, "/mujoco/option")
        if timestep <= 0:
            raise ValueError("/mujoco/option: timestep must be positive")

    worldbodies = [child for child in root if child.tag == "worldbody"]
    if len(worldbodies) != 1:
        raise ValueError("/mujoco/worldbody: exactly one worldbody is required")
    worldbody = worldbodies[0]
    _check_attributes(worldbody, set(), "/mujoco/worldbody")

    bodies: list[BodySpec] = []
    joints: list[JointSpec] = []
    geoms: list[GeomSpec] = []

    def parse_body(
        element: ET.Element,
        parent: int,
        parent_joint_offset: tuple[float, float, float],
        inherited_class: str,
        path: str,
    ) -> None:
        _check_attributes(element, {"name", "pos", "childclass"}, path)
        name = element.get("name")
        if not name:
            raise ValueError(f"{path}: body requires a name")
        body_class = element.get("childclass", inherited_class)
        if body_class not in defaults:
            raise ValueError(f"{path}: unknown childclass {body_class!r}")
        body_pos = _vector(element.attrib, "pos", (0.0, 0.0, 0.0), path)

        inertials = [child for child in element if child.tag == "inertial"]
        explicit_joints = [child for child in element if child.tag == "joint"]
        if len(inertials) != 1:
            detail = "exactly one inertial is required; geom inertia inference is unsupported"
            _unsupported(f"{path}/inertial", detail)
        if len(explicit_joints) > 1:
            _unsupported(f"{path}/joint", "only one joint per body is supported")

        joint_element = explicit_joints[0] if explicit_joints else None
        if joint_element is None:
            joint_name = f"{name}_fixed"
            joint_kind = "fixed"
            joint_axis = (0.0, 0.0, 1.0)
            joint_local_pos = (0.0, 0.0, 0.0)
            damping = 0.0
        else:
            joint_path = _path(path, joint_element)
            _check_attributes(
                joint_element,
                {
                    "name",
                    "type",
                    "axis",
                    "pos",
                    "damping",
                    "limited",
                    "range",
                    "class",
                },
                joint_path,
            )
            _check_leaf(joint_element, joint_path)
            attributes = _effective(joint_element, "joint", body_class, defaults, joint_path)
            joint_name = joint_element.get("name")
            if not joint_name:
                raise ValueError(f"{joint_path}: joint requires a name")
            joint_kind = attributes.get("type", "hinge")
            if joint_kind not in ("hinge", "slide", "ball", "free"):
                _unsupported(joint_path, f"unsupported joint type {joint_kind!r}")
            joint_axis = _vector(attributes, "axis", (0.0, 0.0, 1.0), joint_path)
            if sum(value * value for value in joint_axis) < 1e-24:
                raise ValueError(f"{joint_path}: axis must be nonzero")
            joint_local_pos = _vector(attributes, "pos", (0.0, 0.0, 0.0), joint_path)
            damping = _scalar(attributes, "damping", 0.0, joint_path)
            if damping < 0:
                raise ValueError(f"{joint_path}: damping must be nonnegative")
            limit = _joint_limit(attributes, joint_path)
            if limit is not None and joint_kind not in ("hinge", "slide"):
                _unsupported(
                    joint_path,
                    "limits on ball/free joints require a different representation",
                )
            if limit is not None and joint_kind == "hinge" and angle_unit == "degree":
                limit = tuple(math.radians(value) for value in limit)
        if joint_element is None:
            limit = None

        inertial = inertials[0]
        inertial_path = _path(path, inertial)
        _check_attributes(inertial, {"mass", "diaginertia", "pos"}, inertial_path)
        _check_leaf(inertial, inertial_path)
        if "mass" not in inertial.attrib or "diaginertia" not in inertial.attrib:
            raise ValueError(f"{inertial_path}: mass and diaginertia are required")
        mass = _scalar(inertial.attrib, "mass", 0.0, inertial_path)
        inertia = _numbers(inertial.attrib["diaginertia"], 3, inertial_path, "diaginertia")
        if mass <= 0 or any(value <= 0 for value in inertia):
            raise ValueError(f"{inertial_path}: mass and diagonal inertia must be positive")
        inertial_pos = _vector(inertial.attrib, "pos", (0.0, 0.0, 0.0), inertial_path)
        com = tuple(inertial_pos[index] - joint_local_pos[index] for index in range(3))

        body_index = len(bodies)
        bodies.append(BodySpec(name, parent, mass, inertia, com))  # type: ignore[arg-type]
        runtime_pos = tuple(
            body_pos[index] - parent_joint_offset[index] + joint_local_pos[index]
            for index in range(3)
        )
        joints.append(
            JointSpec(
                joint_name,
                body_index,
                joint_kind,  # type: ignore[arg-type]
                joint_axis,  # type: ignore[arg-type]
                runtime_pos,  # type: ignore[arg-type]
                damping=damping,
                limit=limit,
            )
        )

        allowed_children = {"inertial", "joint", "geom", "body"}
        for child in element:
            child_path = _path(path, child)
            if child.tag not in allowed_children:
                _unsupported(child_path, f"unsupported body element {child.tag!r}")
            if child.tag == "geom":
                geoms.append(
                    _geom(child, body_index, joint_local_pos, body_class, defaults, child_path)
                )
            elif child.tag == "body":
                parse_body(child, body_index, joint_local_pos, body_class, child_path)

    for child in worldbody:
        child_path = _path("/mujoco/worldbody", child)
        if child.tag == "geom":
            geoms.append(_geom(child, -1, (0.0, 0.0, 0.0), "main", defaults, child_path))
        elif child.tag == "body":
            parse_body(child, -1, (0.0, 0.0, 0.0), "main", child_path)
        else:
            _unsupported(child_path, f"unsupported worldbody element {child.tag!r}")

    joint_indices = {joint.name: index for index, joint in enumerate(joints)}
    if len(joint_indices) != len(joints):
        raise ValueError("/mujoco/worldbody: joint names must be unique")
    actuators: list[ActuatorSpec] = []
    for section in (child for child in root if child.tag == "actuator"):
        section_path = "/mujoco/actuator"
        _check_attributes(section, set(), section_path)
        for element in section:
            path = _path(section_path, element)
            if element.tag not in ("motor", "position"):
                _unsupported(path, f"unsupported actuator type {element.tag!r}")
            allowed = {
                "name", "joint", "class", "ctrllimited", "ctrlrange",
                "forcelimited", "forcerange",
            }
            allowed |= {"gear"} if element.tag == "motor" else {"kp", "kv"}
            _check_attributes(element, allowed, path)
            _check_leaf(element, path)
            attributes = _effective(element, element.tag, "main", defaults, path)
            name, joint_name = element.get("name"), element.get("joint")
            if not name or not joint_name:
                raise ValueError(f"{path}: actuator requires name and joint")
            if joint_name not in joint_indices:
                raise ValueError(f"{path}: unknown joint {joint_name!r}")
            if joints[joint_indices[joint_name]].kind == "fixed":
                _unsupported(path, "actuator cannot target a fixed joint")
            if element.tag == "motor":
                gain, damping = _motor_gain(attributes, path), 0.0
            else:
                gain = _scalar(attributes, "kp", 1.0, path)
                damping = _scalar(attributes, "kv", 0.0, path)
            if gain < 0 or damping < 0:
                raise ValueError(f"{path}: gain and damping must be nonnegative")
            force_range = _range(attributes, "force", path)
            if force_range is not None and element.tag == "motor":
                force_range = (
                    gain * force_range[0],
                    gain * force_range[1],
                )
            actuators.append(
                ActuatorSpec(
                    name,
                    joint_indices[joint_name],
                    element.tag,  # type: ignore[arg-type]
                    gain,
                    damping,
                    _range(attributes, "ctrl", path),
                    force_range,
                )
            )

    geom_indices = {
        geom.name: index for index, geom in enumerate(geoms)
        if geom.name is not None
    }
    body_indices = {body.name: index for index, body in enumerate(bodies)}
    collision_pairs: list[tuple[int, int]] = []
    collision_exclusions: list[tuple[int, int]] = []
    contact_sections = [child for child in root if child.tag == "contact"]
    if len(contact_sections) > 1:
        _unsupported("/mujoco/contact", "multiple contact sections are unsupported")
    if contact_sections:
        section = contact_sections[0]
        _check_attributes(section, set(), "/mujoco/contact")
        for element in section:
            path = _path("/mujoco/contact", element)
            if element.tag == "pair":
                _check_attributes(element, {"name", "geom1", "geom2"}, path)
                _check_leaf(element, path)
                names = (element.get("geom1"), element.get("geom2"))
                if any(name not in geom_indices for name in names):
                    raise ValueError(f"{path}: pair references unknown named geom")
                collision_pairs.append(
                    (geom_indices[names[0]], geom_indices[names[1]])  # type: ignore[index]
                )
            elif element.tag == "exclude":
                _check_attributes(element, {"name", "body1", "body2"}, path)
                _check_leaf(element, path)
                names = (element.get("body1"), element.get("body2"))
                if any(name not in body_indices for name in names):
                    raise ValueError(f"{path}: exclude references unknown body")
                body_pair = {
                    body_indices[names[0]], body_indices[names[1]]  # type: ignore[index]
                }
                for a in range(len(geoms)):
                    for b in range(a + 1, len(geoms)):
                        if {geoms[a].body, geoms[b].body} == body_pair:
                            collision_exclusions.append((a, b))
            else:
                _unsupported(path, f"unsupported contact element {element.tag!r}")

    excluded = {
        (min(a, b), max(a, b)) for a, b in collision_exclusions
    }
    automatic_pairs = [
        (a, b)
        for a in range(len(geoms))
        for b in range(a + 1, len(geoms))
        if geoms[a].body != geoms[b].body
        and not (geoms[a].body == geoms[b].body == -1)
        and (a, b) not in excluded
        and (
            (geoms[a].kind, geoms[b].kind) in SUPPORTED_COLLISION_PAIRS
            or (geoms[b].kind, geoms[a].kind) in SUPPORTED_COLLISION_PAIRS
        )
    ]
    collision_pairs = list(
        dict.fromkeys((*automatic_pairs, *collision_pairs))
    )

    qpos_count = sum(
        {"fixed": 0, "hinge": 1, "slide": 1, "ball": 4, "free": 7}[joint.kind]
        for joint in joints
    )
    keyframes: list[Keyframe] = []
    for section in (child for child in root if child.tag == "keyframe"):
        section_path = "/mujoco/keyframe"
        _check_attributes(section, set(), section_path)
        for element in section:
            path = _path(section_path, element)
            if element.tag != "key":
                _unsupported(path, f"unsupported keyframe element {element.tag!r}")
            _check_attributes(element, {"name", "qpos"}, path)
            _check_leaf(element, path)
            name = element.get("name")
            if not name:
                raise ValueError(f"{path}: key requires a name")
            qpos = _numbers(element.get("qpos", ""), qpos_count, path, "qpos")
            keyframes.append((name, qpos))

    spec = ModelSpec(
        bodies=tuple(bodies),
        joints=tuple(joints),
        actuators=tuple(actuators),
        gravity=gravity,
        timestep=timestep,
        name=root.get("model", "model"),
        geoms=tuple(geoms),
        collision_pairs=tuple(collision_pairs),
        collision_exclusions=tuple(collision_exclusions),
    )
    return ImportedModel(spec, tuple(keyframes))
