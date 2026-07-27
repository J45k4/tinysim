"""One-step diagnostics against optional canonical MuJoCo Python bindings.

The adapter deliberately rebuilds a contact-free MuJoCo model from TinySim's
compiled representation. This compares dynamics conventions without making
MuJoCo a parser or runtime dependency. A comparison only constitutes MuJoCo
validation when ``backend_name`` starts with ``"mujoco-python-"``; injected
backends exist solely to test this adapter's contracts.
"""

from dataclasses import dataclass
import importlib
import math
from types import ModuleType
from typing import Any, Protocol, Sequence, runtime_checkable
import xml.etree.ElementTree as ET

from tinygrad import Tensor, dtypes

from ..actuator import actuator_forces
from ..compile import CompiledModel, compile_model
from ..dynamics import bias_forces, forward_dynamics, mass_matrix
from ..integrator import semi_implicit_euler
from ..kinematics import forward_kinematics
from ..model import ModelSpec
from ..spatial import rotate
from ..state import State


Vector = tuple[float, ...]
Matrix = tuple[Vector, ...]
Vectors3 = tuple[tuple[float, float, float], ...]
Vectors4 = tuple[tuple[float, float, float, float], ...]


class MujocoUnavailableError(RuntimeError):
    """Raised when canonical MuJoCo Python bindings cannot be loaded."""


class MujocoCompatibilityError(ValueError):
    """Raised when a model or backend result cannot be compared faithfully."""


@dataclass(frozen=True)
class MujocoStatus:
    available: bool
    version: str | None
    reason: str | None


@dataclass(frozen=True)
class ReferenceSnapshot:
    """Contact-free dynamics intermediates and the result of one Euler step."""

    body_position: Vectors3
    body_quaternion: Vectors4
    body_com_position: Vectors3
    mass_matrix: Matrix
    bias_force: Vector
    actuator_force: Vector
    acceleration: Vector
    next_qpos: Vector
    next_qvel: Vector


@runtime_checkable
class ReferenceBackend(Protocol):
    """Backend boundary used by :class:`MujocoReference`."""

    name: str

    def evaluate(
        self, qpos: Vector, qvel: Vector, control: Vector
    ) -> ReferenceSnapshot:
        """Return reference intermediates for one state and one step."""


@dataclass(frozen=True)
class ComparisonErrors:
    body_position: float
    body_quaternion: float
    body_com_position: float
    mass_matrix: float
    bias_force: float
    actuator_force: float
    acceleration: float
    next_qpos: float
    next_qvel: float

    @property
    def maximum(self) -> float:
        return max(vars(self).values(), default=0.0)


@dataclass(frozen=True)
class MujocoComparison:
    """TinySim and reference results plus field-by-field absolute errors."""

    backend_name: str
    tinysim: ReferenceSnapshot
    reference: ReferenceSnapshot
    errors: ComparisonErrors

    def assert_within(self, atol: float) -> None:
        if atol < 0 or not math.isfinite(atol):
            raise ValueError("atol must be finite and nonnegative")
        failures = {
            name: error
            for name, error in vars(self.errors).items()
            if error > atol
        }
        if failures:
            detail = ", ".join(
                f"{name}={error:.6g}" for name, error in failures.items()
            )
            raise AssertionError(
                f"{self.backend_name} differs from TinySim (atol={atol:g}): {detail}"
            )


_REQUIRED_BINDING_ATTRIBUTES = (
    "MjData",
    "MjModel",
    "mj_forward",
    "mj_fullM",
    "mj_name2id",
    "mj_step",
    "mjtObj",
)


def _binding_status(module: ModuleType | Any) -> MujocoStatus:
    missing = [
        attribute
        for attribute in _REQUIRED_BINDING_ATTRIBUTES
        if not hasattr(module, attribute)
    ]
    if missing:
        locations = ", ".join(str(path) for path in getattr(module, "__path__", ()))
        resolution = (
            f"namespace package at {locations}"
            if locations
            else repr(getattr(module, "__file__", None))
        )
        return MujocoStatus(
            False,
            None,
            "import mujoco resolved to "
            f"{resolution}, not canonical bindings; missing {', '.join(missing)}",
        )
    try:
        importlib.import_module("numpy")
    except (ImportError, ModuleNotFoundError) as error:
        return MujocoStatus(
            False,
            None,
            f"MuJoCo bindings require NumPy, but importing numpy failed: {error}",
        )
    return MujocoStatus(True, str(getattr(module, "__version__", "unknown")), None)


def mujoco_status() -> MujocoStatus:
    """Report binding availability without importing MuJoCo at package import."""
    try:
        module = importlib.import_module("mujoco")
    except (ImportError, ModuleNotFoundError, OSError) as error:
        return MujocoStatus(False, None, f"cannot import mujoco: {error}")
    return _binding_status(module)


def _require_bindings() -> tuple[Any, Any]:
    try:
        module = importlib.import_module("mujoco")
    except (ImportError, ModuleNotFoundError, OSError) as error:
        raise MujocoUnavailableError(f"cannot import mujoco: {error}") from error
    status = _binding_status(module)
    if not status.available:
        raise MujocoUnavailableError(status.reason)
    return module, importlib.import_module("numpy")


def _numbers(values: Sequence[float], count: int, label: str) -> Vector:
    if len(values) != count:
        raise ValueError(f"{label} must contain {count} values")
    result = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{label} must contain finite values")
    return result


def _format(values: Sequence[float]) -> str:
    return " ".join(format(float(value), ".17g") for value in values)


def model_to_mjcf(model: CompiledModel | ModelSpec) -> str:
    """Serialize the shared contact-free subset into diagnostic MJCF.

    TinySim body frames coincide with joint frames, so each joint transform is
    represented by a MuJoCo body transform and the joint itself is at the body
    origin. ``general`` actuators preserve TinySim's gain, damping, and
    post-gain force-clamping semantics.
    """
    compiled = model if isinstance(model, CompiledModel) else compile_model(
        model, device="CPU", dtype=dtypes.float64
    )
    incompatible = [
        joint.name for joint in compiled.joints
        if joint.kind in ("ball", "free")
    ]
    if incompatible:
        names = ", ".join(repr(name) for name in incompatible)
        raise MujocoCompatibilityError(
            "raw MuJoCo generalized velocities use a different angular basis "
            f"for ball/free joint(s) {names}; comparison requires an explicit "
            "basis transform that TinySim does not yet implement"
        )
    for body in compiled.bodies:
        if any(value <= 0 for value in body.inertia):
            raise MujocoCompatibilityError(
                f"body {body.name!r} needs strictly positive diagonal inertia "
                "for MuJoCo comparison"
            )

    root = ET.Element("mujoco", {"model": compiled.name})
    ET.SubElement(root, "compiler", {"angle": "radian", "autolimits": "true"})
    ET.SubElement(
        root,
        "option",
        {
            "timestep": format(compiled.timestep, ".17g"),
            "gravity": _format(compiled.gravity.tolist()),
            "integrator": "Euler",
        },
    )
    world = ET.SubElement(root, "worldbody")
    body_elements: list[ET.Element] = []
    for index, body in enumerate(compiled.bodies):
        joint = compiled.joints[compiled.joint_for_body[index]]
        parent = world if body.parent == -1 else body_elements[body.parent]
        body_element = ET.SubElement(
            parent,
            "body",
            {
                "name": body.name,
                "pos": _format(joint.pos),
                "quat": _format(joint.quat),
            },
        )
        ET.SubElement(
            body_element,
            "inertial",
            {
                "pos": _format(body.com),
                "mass": format(body.mass, ".17g"),
                "diaginertia": _format(body.inertia),
            },
        )
        if joint.kind != "fixed":
            ET.SubElement(
                body_element,
                "joint",
                {
                    "name": joint.name,
                    "type": joint.kind,
                    "axis": _format(joint.axis),
                    "damping": format(joint.damping, ".17g"),
                },
            )
        body_elements.append(body_element)

    if compiled.actuators:
        actuators = ET.SubElement(root, "actuator")
        for index, actuator in enumerate(compiled.actuators):
            attributes = {
                "name": actuator.name,
                "joint": compiled.joints[
                    compiled.dof_joint[compiled.actuator_dof[index]]
                ].name,
                "dyntype": "none",
                "gaintype": "fixed",
                "gainprm": format(actuator.gain, ".17g"),
            }
            if actuator.kind == "motor":
                attributes["biastype"] = "none"
            else:
                attributes["biastype"] = "affine"
                attributes["biasprm"] = (
                    _format((0.0, -actuator.gain, -actuator.damping))
                    if actuator.kind == "position"
                    else _format((0.0, 0.0, -actuator.gain))
                )
            if actuator.control_range is not None:
                attributes["ctrllimited"] = "true"
                attributes["ctrlrange"] = _format(actuator.control_range)
            if actuator.force_range is not None:
                attributes["forcelimited"] = "true"
                attributes["forcerange"] = _format(actuator.force_range)
            ET.SubElement(actuators, "general", attributes)
    return ET.tostring(root, encoding="unicode")


def _vector(values: Any) -> Vector:
    return tuple(float(value) for value in values)


def _matrix(values: Any) -> Matrix:
    return tuple(_vector(row) for row in values)


def _vectors(values: Any, width: int) -> tuple[Any, ...]:
    rows = tuple(_vector(row) for row in values)
    if any(len(row) != width for row in rows):
        raise MujocoCompatibilityError(
            f"reference backend returned vectors with width other than {width}"
        )
    return rows


class _PythonBindingsBackend:
    def __init__(self, model: CompiledModel, xml: str):
        self._mj, self._np = _require_bindings()
        try:
            self._model = self._mj.MjModel.from_xml_string(xml)
        except Exception as error:
            raise MujocoCompatibilityError(
                f"MuJoCo rejected generated diagnostic MJCF: {error}"
            ) from error
        self._data = self._mj.MjData(self._model)
        expected = (model.nq, model.nv, model.nu)
        actual = (self._model.nq, self._model.nv, self._model.nu)
        if actual != expected:
            raise MujocoCompatibilityError(
                f"generated MuJoCo dimensions {actual} do not match TinySim {expected}"
            )
        self._body_ids = tuple(
            self._mj.mj_name2id(
                self._model, self._mj.mjtObj.mjOBJ_BODY, body.name
            )
            for body in model.bodies
        )
        if any(body_id < 0 for body_id in self._body_ids):
            raise MujocoCompatibilityError("MuJoCo lost a named TinySim body")
        version = getattr(self._mj, "__version__", "unknown")
        self.name = f"mujoco-python-{version}"

    def _full_mass_matrix(self) -> Any:
        dense = self._np.empty(
            (self._model.nv, self._model.nv), dtype=self._np.float64
        )
        try:
            self._mj.mj_fullM(self._model, self._data, dense)
        except TypeError:
            # MuJoCo <= 3.3 exposed the older (model, dst, packed_qM) binding.
            self._mj.mj_fullM(self._model, dense, self._data.qM)
        return dense

    def evaluate(
        self, qpos: Vector, qvel: Vector, control: Vector
    ) -> ReferenceSnapshot:
        self._data.qpos[:] = qpos
        self._data.qvel[:] = qvel
        self._data.ctrl[:] = control
        self._mj.mj_forward(self._model, self._data)
        body_position = _vectors(
            self._data.xpos[list(self._body_ids)].copy(), 3
        )
        body_quaternion = _vectors(
            self._data.xquat[list(self._body_ids)].copy(), 4
        )
        body_com_position = _vectors(
            self._data.xipos[list(self._body_ids)].copy(), 3
        )
        matrix = _matrix(self._full_mass_matrix())
        bias = _vector(
            (self._data.qfrc_bias - self._data.qfrc_passive).copy()
        )
        actuator = _vector(self._data.qfrc_actuator.copy())
        acceleration = _vector(self._data.qacc.copy())
        self._mj.mj_step(self._model, self._data)
        return ReferenceSnapshot(
            body_position,
            body_quaternion,
            body_com_position,
            matrix,
            bias,
            actuator,
            acceleration,
            _vector(self._data.qpos.copy()),
            _vector(self._data.qvel.copy()),
        )


def _shape(snapshot: ReferenceSnapshot, model: CompiledModel) -> None:
    fields = (
        ("body_position", snapshot.body_position, model.nbody, 3),
        ("body_quaternion", snapshot.body_quaternion, model.nbody, 4),
        ("body_com_position", snapshot.body_com_position, model.nbody, 3),
        ("mass_matrix", snapshot.mass_matrix, model.nv, model.nv),
    )
    for name, values, rows, columns in fields:
        if len(values) != rows or any(len(row) != columns for row in values):
            raise MujocoCompatibilityError(
                f"reference {name} must have shape [{rows}, {columns}]"
            )
    for name in (
        "bias_force",
        "actuator_force",
        "acceleration",
        "next_qpos",
        "next_qvel",
    ):
        values = getattr(snapshot, name)
        expected = model.nq if name == "next_qpos" else model.nv
        if len(values) != expected:
            raise MujocoCompatibilityError(
                f"reference {name} must contain {expected} values"
            )
    scalars = (
        value
        for name in vars(snapshot)
        for row in getattr(snapshot, name)
        for value in (row if isinstance(row, tuple) else (row,))
    )
    if not all(math.isfinite(float(value)) for value in scalars):
        raise MujocoCompatibilityError("reference snapshot contains non-finite values")


def _maximum_difference(left: Any, right: Any) -> float:
    if isinstance(left, tuple):
        return max(
            (_maximum_difference(a, b) for a, b in zip(left, right, strict=True)),
            default=0.0,
        )
    return abs(float(left) - float(right))


def _quaternion_difference(left: Vectors4, right: Vectors4) -> float:
    errors = []
    for a, b in zip(left, right, strict=True):
        direct = max(abs(x - y) for x, y in zip(a, b, strict=True))
        negated = max(abs(x + y) for x, y in zip(a, b, strict=True))
        errors.append(min(direct, negated))
    return max(errors, default=0.0)


class MujocoReference:
    """Compare one contact-free TinySim world with a MuJoCo backend."""

    def __init__(
        self,
        model: CompiledModel | ModelSpec,
        *,
        dtype: object = dtypes.float64,
        backend: ReferenceBackend | None = None,
    ):
        self.model = model if isinstance(model, CompiledModel) else compile_model(
            model, device="CPU", dtype=dtype
        )
        self.xml = model_to_mjcf(self.model)
        self.backend = backend or _PythonBindingsBackend(self.model, self.xml)
        if not isinstance(self.backend, ReferenceBackend):
            raise TypeError("backend must implement name and evaluate")

    @property
    def backend_name(self) -> str:
        return self.backend.name

    def tinysim_snapshot(
        self,
        qpos: Sequence[float],
        qvel: Sequence[float],
        control: Sequence[float] = (),
    ) -> ReferenceSnapshot:
        position = _numbers(qpos, self.model.nq, "qpos")
        velocity = _numbers(qvel, self.model.nv, "qvel")
        command = _numbers(control, self.model.nu, "control")
        tensor = lambda values: Tensor(
            [values], dtype=self.model.dtype, device=self.model.device
        )
        qpos_tensor, qvel_tensor, control_tensor = (
            tensor(position),
            tensor(velocity),
            tensor(command),
        )
        kinematics = forward_kinematics(self.model, qpos_tensor)
        com = tuple(
            kinematics.body_pos[:, body]
            + rotate(
                kinematics.body_quat[:, body],
                self.model.body_com[body].unsqueeze(0),
            )
            for body in range(self.model.nbody)
        )
        force = actuator_forces(
            self.model, qpos_tensor, qvel_tensor, control_tensor
        )
        acceleration = forward_dynamics(
            self.model, qpos_tensor, qvel_tensor, force
        )
        state = State(
            qpos_tensor,
            qvel_tensor,
            control_tensor,
            Tensor.zeros(
                1, dtype=self.model.dtype, device=self.model.device
            ),
        )
        next_state = semi_implicit_euler(
            self.model, state, acceleration, control=control_tensor
        )
        snapshot = ReferenceSnapshot(
            _vectors(kinematics.body_pos.tolist()[0], 3),
            _vectors(kinematics.body_quat.tolist()[0], 4),
            _vectors([value.tolist()[0] for value in com], 3),
            _matrix(mass_matrix(self.model, qpos_tensor, kinematics=kinematics).tolist()[0]),
            _vector(
                bias_forces(
                    self.model,
                    qpos_tensor,
                    qvel_tensor,
                    kinematics=kinematics,
                ).tolist()[0]
            ),
            _vector(force.tolist()[0]),
            _vector(acceleration.tolist()[0]),
            _vector(next_state.qpos.tolist()[0]),
            _vector(next_state.qvel.tolist()[0]),
        )
        _shape(snapshot, self.model)
        return snapshot

    def compare(
        self,
        qpos: Sequence[float],
        qvel: Sequence[float],
        control: Sequence[float] = (),
    ) -> MujocoComparison:
        position = _numbers(qpos, self.model.nq, "qpos")
        velocity = _numbers(qvel, self.model.nv, "qvel")
        command = _numbers(control, self.model.nu, "control")
        tinysim = self.tinysim_snapshot(position, velocity, command)
        reference = self.backend.evaluate(position, velocity, command)
        _shape(reference, self.model)
        errors = ComparisonErrors(
            _maximum_difference(tinysim.body_position, reference.body_position),
            _quaternion_difference(
                tinysim.body_quaternion, reference.body_quaternion
            ),
            _maximum_difference(
                tinysim.body_com_position, reference.body_com_position
            ),
            _maximum_difference(tinysim.mass_matrix, reference.mass_matrix),
            _maximum_difference(tinysim.bias_force, reference.bias_force),
            _maximum_difference(tinysim.actuator_force, reference.actuator_force),
            _maximum_difference(tinysim.acceleration, reference.acceleration),
            _maximum_difference(tinysim.next_qpos, reference.next_qpos),
            _maximum_difference(tinysim.next_qvel, reference.next_qvel),
        )
        return MujocoComparison(self.backend.name, tinysim, reference, errors)
