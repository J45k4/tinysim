"""Deterministic numerical and visual release evidence."""

from argparse import ArgumentParser
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from math import cos, isfinite, sqrt
from pathlib import Path
import time

from tinygrad import Device, Tensor

from .analytical import (
    cartpole_step,
    make_jitted_cartpole_step,
    make_jitted_pendulum_step,
    pendulum_step,
)
from .importers import load_mjcf
from .model import BodySpec, JointSpec, ModelSpec
from .provenance import project_source_sha256, source_tree_sha256
from .render import (
    encode_mp4,
    render_articulated,
    render_batched_pendulums,
    render_bouncing_ball,
    render_cartpole,
    render_free_body,
    render_pendulum,
    write_ppm,
)
from .simulation import Simulator
from .state import State
from .systems import make_jitted_sphere_plane_step, sphere_plane_step
from .trajectory import (
    Trajectory,
    load_trajectory,
    playback_indices,
    save_trajectory,
)


@dataclass(frozen=True)
class VerificationReport:
    passed: bool
    scenario: str
    checks: dict[str, bool]
    metrics: dict[str, float]
    artifacts: dict[str, str]
    trajectory_sha256: str
    environment: dict[str, str]
    configuration: dict[str, float | int | str]


FrameRenderer = Callable[[Sequence[float]], bytes]


def _tinygrad_revision() -> str:
    lock_path = Path(__file__).resolve().parent.parent / "research-lock.json"
    try:
        return str(json.loads(lock_path.read_text(encoding="utf-8"))["tinygrad"]["commit"])
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        return "unknown"


def _source_revision() -> str:
    return source_tree_sha256(Path(__file__).resolve().parent)


def _environment(tensor: Tensor) -> dict[str, str]:
    source_revision = _source_revision()
    project_revision = project_source_sha256(
        Path(__file__).resolve().parent.parent
    )
    return {
        "backend": Device.DEFAULT,
        "device": tensor.device,
        "dtype": str(tensor.dtype),
        "tinygrad_commit": _tinygrad_revision(),
        "tinysim_revision": f"source-sha256:{source_revision}",
        "tinysim_source_sha256": source_revision,
        "tinysim_project_source_sha256": project_revision,
    }


def _jit_call_count(jitted: object) -> int:
    captured = getattr(jitted, "captured", None)
    return 0 if captured is None else len(captured.linear.src)


def _bounded_jit(jitted: object) -> tuple[bool, int]:
    calls = _jit_call_count(jitted)
    # A single physics step should remain a small captured graph. More
    # importantly, rollout length never appears in this count.
    return 0 < calls <= 128, calls


def _finite(rows: Sequence[Sequence[float]]) -> bool:
    return all(isfinite(value) for row in rows for value in row)


def _write_evidence(
    output: str | Path,
    trajectory: Trajectory,
    *,
    checks: dict[str, bool],
    metrics: dict[str, float],
    configuration: dict[str, float | int | str],
    reference_tensor: Tensor,
    render_frame: FrameRenderer,
    record: str | Path | None,
    width: int,
    height: int,
    fps: int,
    rollout_wall_seconds: float,
) -> VerificationReport:
    if width < 16 or height < 16:
        raise ValueError("frame dimensions must be at least 16 pixels")
    if fps <= 0:
        raise ValueError("fps must be positive")
    trajectory.validate()
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = trajectory.scenario
    trajectory_path = output_dir / f"{stem}.trajectory.json"
    save_trajectory(trajectory, trajectory_path)
    trajectory_digest = sha256(trajectory_path.read_bytes()).hexdigest()
    rendering_trajectory = load_trajectory(trajectory_path)

    indices = playback_indices(
        len(rendering_trajectory.qpos),
        timestep=rendering_trajectory.timestep,
        fps=fps,
    )
    artifacts = {"trajectory": str(trajectory_path)}
    if record is not None:
        encode_mp4(
            (
                render_frame(rendering_trajectory.qpos[index])
                for index in indices
            ),
            record,
            width=width,
            height=height,
            fps=fps,
        )
        artifacts["video"] = str(Path(record))
    else:
        first_frame = output_dir / f"{stem}.first.ppm"
        last_frame = output_dir / f"{stem}.last.ppm"
        write_ppm(
            first_frame,
            render_frame(rendering_trajectory.qpos[0]),
            width=width,
            height=height,
        )
        write_ppm(
            last_frame,
            render_frame(rendering_trajectory.qpos[-1]),
            width=width,
            height=height,
        )
        artifacts.update(first_frame=str(first_frame), last_frame=str(last_frame))

    simulated_seconds = (len(trajectory.qpos) - 1) * trajectory.timestep
    if rollout_wall_seconds <= 0 or not isfinite(rollout_wall_seconds):
        raise ValueError("rollout wall time must be finite and positive")
    report_metrics = {
        **metrics,
        "steps": float(len(trajectory.qpos) - 1),
        "simulated_seconds": simulated_seconds,
        "trajectory_recording_wall_seconds": rollout_wall_seconds,
        "trajectory_recording_realtime_factor": (
            simulated_seconds / rollout_wall_seconds
        ),
        "video_frames": float(len(indices)),
        "video_playback_seconds": len(indices) / fps,
        "video_duration_error_seconds": abs(len(indices) / fps - simulated_seconds),
    }
    report = VerificationReport(
        passed=all(checks.values()),
        scenario=trajectory.scenario,
        checks=checks,
        metrics=report_metrics,
        artifacts=artifacts,
        trajectory_sha256=trajectory_digest,
        environment=_environment(reference_tensor),
        configuration={
            **configuration,
            "timing_scope": "simulation_plus_per_step_host_trajectory_recording",
            "width": width,
            "height": height,
            "fps": fps,
        },
    )
    manifest_path = output_dir / f"{stem}.manifest.json"
    manifest_path.write_text(
        json.dumps(asdict(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return VerificationReport(
        **{
            **asdict(report),
            "artifacts": {**artifacts, "manifest": str(manifest_path)},
        }
    )


def _gradient_error(theta: float, *, dt: float) -> float:
    angle = Tensor([[theta]])
    velocity = Tensor([[0.0]])
    control = Tensor([[0.0]])
    loss = pendulum_step(
        angle, velocity, control, damping=0.0, dt=dt
    )[0].square().mean()
    loss.backward()
    autodiff = float(control.grad.item())
    epsilon = 1e-2
    plus = float(
        pendulum_step(
            angle.detach(),
            velocity.detach(),
            control.detach() + epsilon,
            damping=0.0,
            dt=dt,
        )[0]
        .square()
        .mean()
        .item()
    )
    minus = float(
        pendulum_step(
            angle.detach(),
            velocity.detach(),
            control.detach() - epsilon,
            damping=0.0,
            dt=dt,
        )[0]
        .square()
        .mean()
        .item()
    )
    finite_difference = (plus - minus) / (2.0 * epsilon)
    return abs(autodiff - finite_difference) / max(
        abs(autodiff), abs(finite_difference), 1e-12
    )


def _pendulum_jit_matches(theta: float, *, dt: float) -> bool:
    angle = Tensor([[theta]]).realize()
    velocity = Tensor([[0.0]]).realize()
    control = Tensor([[0.0]]).realize()
    expected = pendulum_step(angle, velocity, control, damping=0.0, dt=dt)
    jitted = make_jitted_pendulum_step(damping=0.0, dt=dt)
    actual = None
    for _ in range(3):
        actual = jitted(
            angle.clone().realize(),
            velocity.clone().realize(),
            control.clone().realize(),
        )
    assert actual is not None
    return all(
        bool(got.isclose(want, rtol=1e-5, atol=1e-6).all().item())
        for got, want in zip(actual, expected)
    )


def verify_pendulum(
    output: str | Path,
    *,
    steps: int = 600,
    timestep: float = 0.01,
    initial_angle: float = 0.4,
    record: str | Path | None = None,
    width: int = 640,
    height: int = 480,
    fps: int = 60,
) -> VerificationReport:
    """Run the deterministic pendulum demonstration and write its evidence."""
    if steps < 2:
        raise ValueError("steps must be at least two")
    if timestep <= 0 or not isfinite(timestep) or not isfinite(initial_angle):
        raise ValueError("timestep must be positive and inputs must be finite")

    angle = Tensor([[initial_angle]]).realize()
    velocity = Tensor([[0.0]]).realize()
    control = Tensor([[0.0]]).realize()
    rollout_step = make_jitted_pendulum_step(damping=0.0, dt=timestep)
    angles, velocities = [[initial_angle]], [[0.0]]
    Device[Device.DEFAULT].synchronize()
    rollout_started = time.perf_counter()
    for _ in range(steps):
        angle, velocity = rollout_step(angle, velocity, control)
        angles.append([float(angle.item())])
        velocities.append([float(velocity.item())])
    Device[Device.DEFAULT].synchronize()
    rollout_wall_seconds = time.perf_counter() - rollout_started
    jit_bounded, jit_calls = _bounded_jit(rollout_step)

    trajectory = Trajectory(
        scenario="pendulum",
        timestep=timestep,
        qpos=angles,
        qvel=velocities,
        controls=[[0.0] for _ in range(steps)],
        metadata={
            "seed": 0,
            "integrator": "semi_implicit_euler",
            "worlds": 1,
            "coordinate_names": ["angle"],
        },
    )
    energies = [
        0.5 * omega[0] ** 2 + 9.81 * (1.0 - cos(theta[0]))
        for theta, omega in zip(angles, velocities)
    ]
    energy_scale = max(abs(energies[0]), 1e-12)
    max_energy_drift = (
        max(abs(value - energies[0]) for value in energies) / energy_scale
    )
    gradient_error = _gradient_error(initial_angle, dt=timestep)
    return _write_evidence(
        output,
        trajectory,
        checks={
            "finite_state": _finite(angles) and _finite(velocities),
            "energy_drift_below_2_percent": max_energy_drift < 0.02,
            "control_gradient_relative_error_below_1_percent": gradient_error
            < 0.01,
            "eager_matches_tinyjit": _pendulum_jit_matches(
                initial_angle, dt=timestep
            ),
            "bounded_tinyjit_graph": jit_bounded,
        },
        metrics={
            "maximum_relative_energy_drift": max_energy_drift,
            "control_gradient_relative_error": gradient_error,
            "tinyjit_calls": float(jit_calls),
        },
        configuration={
            "seed": 0,
            "steps": steps,
            "timestep": timestep,
            "initial_angle": initial_angle,
            "mass": 1.0,
            "length": 1.0,
            "damping": 0.0,
            "gravity": 9.81,
            "control": 0.0,
            "integrator": "semi_implicit_euler",
            "worlds": 1,
        },
        reference_tensor=angle,
        render_frame=lambda row: render_pendulum(
            row[0], width=width, height=height
        ),
        record=record,
        width=width,
        height=height,
        fps=fps,
        rollout_wall_seconds=rollout_wall_seconds,
    )


def verify_cartpole(
    output: str | Path,
    *,
    steps: int = 160,
    timestep: float = 0.01,
    initial_angle: float = 0.05,
    force_value: float = 0.0,
    record: str | Path | None = None,
    width: int = 640,
    height: int = 480,
    fps: int = 60,
) -> VerificationReport:
    """Verify the analytical cart-pole and generate a side-view recording."""
    if steps < 3:
        raise ValueError("steps must be at least three")
    if timestep <= 0 or not all(
        isfinite(value) for value in (timestep, initial_angle, force_value)
    ):
        raise ValueError("timestep must be positive and inputs must be finite")
    state = (
        Tensor([[0.0]]).realize(),
        Tensor([[0.0]]).realize(),
        Tensor([[initial_angle]]).realize(),
        Tensor([[0.0]]).realize(),
    )
    force = Tensor([[force_value]]).realize()
    rollout_step = make_jitted_cartpole_step(dt=timestep)
    positions = [[0.0, initial_angle]]
    velocities = [[0.0, 0.0]]
    Device[Device.DEFAULT].synchronize()
    rollout_started = time.perf_counter()
    for _ in range(steps):
        state = rollout_step(*state, force)
        positions.append([float(state[0].item()), float(state[2].item())])
        velocities.append([float(state[1].item()), float(state[3].item())])
    Device[Device.DEFAULT].synchronize()
    rollout_wall_seconds = time.perf_counter() - rollout_started
    jit_bounded, jit_calls = _bounded_jit(rollout_step)
    expected = cartpole_step(*state, force, dt=timestep)
    Tensor.realize(*expected)
    actual = rollout_step(*state, force)
    eager_matches = all(
        bool(got.isclose(want, rtol=1e-5, atol=1e-6).all().item())
        for got, want in zip(actual, expected)
    )
    zero = Tensor.zeros(1, 1)
    equilibrium = cartpole_step(zero, zero, zero, zero, zero, dt=timestep)
    equilibrium_stationary = all(
        bool((coordinate == 0).all().item()) for coordinate in equilibrium
    )
    maximum_coordinate = max(abs(value) for row in positions for value in row)
    trajectory = Trajectory(
        scenario="cartpole",
        timestep=timestep,
        qpos=positions,
        qvel=velocities,
        controls=[[force_value] for _ in range(steps)],
        metadata={
            "seed": 0,
            "integrator": "semi_implicit_euler",
            "worlds": 1,
            "coordinate_names": ["cart_position", "pole_angle"],
        },
    )
    return _write_evidence(
        output,
        trajectory,
        checks={
            "finite_state": _finite(positions) and _finite(velocities),
            "upright_equilibrium_is_stationary": equilibrium_stationary,
            "displaced_pole_moves": abs(positions[-1][1] - initial_angle) > 1e-5,
            "coordinates_remain_bounded": maximum_coordinate < 1_000.0,
            "eager_matches_tinyjit": eager_matches,
            "bounded_tinyjit_graph": jit_bounded,
        },
        metrics={
            "maximum_absolute_coordinate": maximum_coordinate,
            "final_cart_position": positions[-1][0],
            "final_pole_angle": positions[-1][1],
            "tinyjit_calls": float(jit_calls),
        },
        configuration={
            "seed": 0,
            "steps": steps,
            "timestep": timestep,
            "initial_angle": initial_angle,
            "force": force_value,
            "cart_mass": 1.0,
            "pole_mass": 0.1,
            "length": 0.5,
            "gravity": 9.81,
            "integrator": "semi_implicit_euler",
            "worlds": 1,
        },
        reference_tensor=state[0],
        render_frame=lambda row: render_cartpole(
            row[0], row[1], width=width, height=height
        ),
        record=record,
        width=width,
        height=height,
        fps=fps,
        rollout_wall_seconds=rollout_wall_seconds,
    )


def verify_bouncing_ball(
    output: str | Path,
    *,
    steps: int = 800,
    timestep: float = 0.002,
    initial_height: float = 0.75,
    radius: float = 0.25,
    record: str | Path | None = None,
    width: int = 640,
    height: int = 480,
    fps: int = 60,
) -> VerificationReport:
    """Verify smooth sphere-plane contact and generate a fixed-camera view."""
    if steps < 3:
        raise ValueError("steps must be at least three")
    if (
        timestep <= 0
        or radius <= 0
        or initial_height <= radius
        or not all(isfinite(value) for value in (timestep, radius, initial_height))
    ):
        raise ValueError("ball dimensions/timestep must be finite and positive")
    position = Tensor([[0.0, 0.0, initial_height]]).realize()
    velocity = Tensor.zeros(1, 3).realize()
    rollout_step = make_jitted_sphere_plane_step(
        radius=radius, timestep=timestep
    )
    positions = [[0.0, 0.0, initial_height]]
    velocities = [[0.0, 0.0, 0.0]]
    Device[Device.DEFAULT].synchronize()
    rollout_started = time.perf_counter()
    for _ in range(steps):
        position, velocity = rollout_step(position, velocity)
        positions.append([float(value) for value in position[0].tolist()])
        velocities.append([float(value) for value in velocity[0].tolist()])
    Device[Device.DEFAULT].synchronize()
    rollout_wall_seconds = time.perf_counter() - rollout_started
    jit_bounded, jit_calls = _bounded_jit(rollout_step)
    expected = sphere_plane_step(
        position, velocity, radius=radius, timestep=timestep
    )
    Tensor.realize(*expected)
    actual = rollout_step(position, velocity)
    eager_matches = all(
        bool(got.isclose(want, rtol=1e-5, atol=1e-6).all().item())
        for got, want in zip(actual, expected)
    )
    minimum_height = min(row[2] for row in positions)
    rebound = minimum_height < radius + 0.02 and any(
        row[2] > 0.05 for row in velocities
    )
    contact_velocities = [
        velocity_row[2]
        for position_row, velocity_row in zip(positions, velocities)
        if position_row[2] <= radius + 0.02
    ]
    impact_speed = abs(min(contact_velocities, default=0.0))
    rebound_speed = max(contact_velocities, default=0.0)
    effective_restitution = (
        rebound_speed / impact_speed if impact_speed > 1e-12 else 0.0
    )
    settled = (
        abs(positions[-1][2] - radius) < 0.01
        and abs(velocities[-1][2]) < 1e-3
    )
    horizontal_drift = max(
        abs(row[axis]) for row in positions for axis in (0, 1)
    )
    trajectory = Trajectory(
        scenario="bouncing_ball",
        timestep=timestep,
        qpos=positions,
        qvel=velocities,
        metadata={
            "seed": 0,
            "integrator": "semi_implicit_euler",
            "worlds": 1,
            "coordinate_names": ["x", "y", "z"],
            "contact": "smooth_sphere_plane",
        },
    )
    return _write_evidence(
        output,
        trajectory,
        checks={
            "finite_state": _finite(positions) and _finite(velocities),
            "contact_prevents_tunneling": minimum_height > 0.15,
            "rebound_detected": rebound,
            "effective_restitution_is_bounded": 0.0
            < effective_restitution
            < 1.2,
            "settles_near_surface": settled,
            "horizontal_position_is_invariant": horizontal_drift < 1e-7,
            "eager_matches_tinyjit": eager_matches,
            "bounded_tinyjit_graph": jit_bounded,
        },
        metrics={
            "minimum_height": minimum_height,
            "maximum_height": max(row[2] for row in positions),
            "horizontal_drift": horizontal_drift,
            "effective_restitution": effective_restitution,
            "final_vertical_speed": abs(velocities[-1][2]),
            "tinyjit_calls": float(jit_calls),
        },
        configuration={
            "seed": 0,
            "steps": steps,
            "timestep": timestep,
            "initial_height": initial_height,
            "radius": radius,
            "mass": 1.0,
            "gravity_z": -9.81,
            "contact_stiffness": 5_000.0,
            "contact_damping": 100.0,
            "integrator": "semi_implicit_euler",
            "worlds": 1,
        },
        reference_tensor=position,
        render_frame=lambda row: render_bouncing_ball(
            row, width=width, height=height, radius=radius
        ),
        record=record,
        width=width,
        height=height,
        fps=fps,
        rollout_wall_seconds=rollout_wall_seconds,
    )


def _default_articulated_mjcf() -> Path:
    return (
        Path(__file__).resolve().parent.parent
        / "examples"
        / "verification"
        / "two_link.xml"
    )


def verify_imported_articulated(
    output: str | Path,
    *,
    source: str | Path | None = None,
    steps: int = 240,
    record: str | Path | None = None,
    width: int = 640,
    height: int = 480,
    fps: int = 60,
) -> VerificationReport:
    """Verify a contact-free two-link model loaded through the MJCF importer."""
    if steps < 3:
        raise ValueError("steps must be at least three")
    source_path = _default_articulated_mjcf() if source is None else Path(source)
    source_bytes = source_path.read_bytes()
    imported = load_mjcf(source_path)
    simulator = Simulator.compile(imported.spec, worlds=1)
    if simulator.model.nq != 2:
        raise ValueError("articulated verification model must contain two coordinates")
    initial_qpos = [0.25, -0.15]
    base = simulator.make_state()
    state = State(
        Tensor([initial_qpos], device=simulator.model.device, dtype=simulator.model.dtype).realize(),
        base.qvel.realize(),
        base.ctrl.realize(),
        base.time.realize(),
    )
    control = simulator.zeros_control().realize()
    positions = [initial_qpos]
    velocities = [[0.0, 0.0]]
    Device[Device.DEFAULT].synchronize()
    rollout_started = time.perf_counter()
    for _ in range(steps):
        state = simulator.inference_step(state, control)
        positions.append([float(value) for value in state.qpos[0].tolist()])
        velocities.append([float(value) for value in state.qvel[0].tolist()])
    Device[Device.DEFAULT].synchronize()
    rollout_wall_seconds = time.perf_counter() - rollout_started
    assert simulator._inference_step is not None
    jit_bounded, jit_calls = _bounded_jit(simulator._inference_step)
    elapsed = float(state.time.item())
    expected = simulator.step(state, control)
    Tensor.realize(expected.qpos, expected.qvel, expected.time)
    actual = simulator.inference_step(state, control)
    eager_matches = bool(
        actual.qpos.isclose(expected.qpos, rtol=1e-5, atol=1e-6).all().item()
        and actual.qvel.isclose(expected.qvel, rtol=1e-5, atol=1e-6).all().item()
    )
    maximum_coordinate = max(abs(value) for row in positions for value in row)
    limited_joints = [
        (simulator.model.joint_qpos[index], joint.limit)
        for index, joint in enumerate(simulator.model.joints)
        if joint.limit is not None
    ]
    maximum_limit_violation = max(
        (
            max(lower - row[index], row[index] - upper, 0.0)
            for row in positions
            for index, (lower, upper) in limited_joints
        ),
        default=0.0,
    )
    trajectory = Trajectory(
        scenario="imported_articulated",
        timestep=imported.spec.timestep,
        qpos=positions,
        qvel=velocities,
        controls=[[] for _ in range(steps)],
        metadata={
            "seed": 0,
            "integrator": "semi_implicit_euler",
            "worlds": 1,
            "coordinate_names": ["shoulder", "elbow"],
            "source_format": "MJCF",
            "mjcf_sha256": sha256(source_bytes).hexdigest(),
            "contact": "none",
        },
    )
    return _write_evidence(
        output,
        trajectory,
        checks={
            "finite_state": _finite(positions) and _finite(velocities),
            "imported_two_link_topology": simulator.model.nbody == 2
            and simulator.model.nq == 2,
            "articulated_state_moves": max(
                abs(final - initial)
                for final, initial in zip(positions[-1], initial_qpos)
            )
            > 1e-5,
            "simulation_time_matches_steps": abs(
                elapsed - steps * imported.spec.timestep
            )
            < 1e-5,
            "coordinates_remain_bounded": maximum_coordinate < 100.0,
            "joint_limits_respected": maximum_limit_violation < 1e-6,
            "eager_matches_tinyjit": eager_matches,
            "bounded_tinyjit_graph": jit_bounded,
        },
        metrics={
            "maximum_absolute_coordinate": maximum_coordinate,
            "final_simulation_time": elapsed,
            "maximum_joint_limit_violation": maximum_limit_violation,
            "tinyjit_calls": float(jit_calls),
        },
        configuration={
            "seed": 0,
            "steps": steps,
            "timestep": imported.spec.timestep,
            "initial_shoulder": initial_qpos[0],
            "initial_elbow": initial_qpos[1],
            "mjcf_sha256": sha256(source_bytes).hexdigest(),
            "model": imported.spec.name,
            "integrator": "semi_implicit_euler",
            "contact": "none",
            "worlds": 1,
        },
        reference_tensor=state.qpos,
        render_frame=lambda row: render_articulated(
            row, width=width, height=height
        ),
        record=record,
        width=width,
        height=height,
        fps=fps,
        rollout_wall_seconds=rollout_wall_seconds,
    )


def _rotate_host(
    quaternion: Sequence[float], vector: Sequence[float]
) -> tuple[float, float, float]:
    w, x, y, z = quaternion
    vx, vy, vz = vector
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (
        vx + w * tx + y * tz - z * ty,
        vy + w * ty + z * tx - x * tz,
        vz + w * tz + x * ty - y * tx,
    )


def verify_free_body(
    output: str | Path,
    *,
    steps: int = 240,
    timestep: float = 0.002,
    record: str | Path | None = None,
    width: int = 640,
    height: int = 480,
    fps: int = 60,
) -> VerificationReport:
    """Record a torque-free anisotropic rigid body with a unit quaternion."""
    if steps < 3:
        raise ValueError("steps must be at least three")
    if timestep <= 0 or not isfinite(timestep):
        raise ValueError("timestep must be finite and positive")
    inertia = (0.2, 0.3, 0.4)
    mass = 2.0
    simulator = Simulator.compile(
        ModelSpec(
            [BodySpec("body", mass=mass, inertia=inertia)],
            [JointSpec("root", 0, "free")],
            gravity=(0.0, 0.0, 0.0),
            timestep=timestep,
            name="free_body_verification",
        ),
        worlds=1,
    )
    base = simulator.make_state()
    initial_qpos = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    initial_qvel = [0.15, -0.08, 0.04, 0.7, -0.4, 0.2]
    state = State(
        Tensor([initial_qpos], dtype=simulator.model.dtype).realize(),
        Tensor([initial_qvel], dtype=simulator.model.dtype).realize(),
        base.ctrl.realize(),
        base.time.realize(),
        base.constraint_impulse,
        base.parameters,
    )
    control = simulator.zeros_control().realize()
    positions = [initial_qpos]
    velocities = [initial_qvel]
    Device[Device.DEFAULT].synchronize()
    rollout_started = time.perf_counter()
    for _ in range(steps):
        state = simulator.inference_step(state, control)
        positions.append([float(value) for value in state.qpos[0].tolist()])
        velocities.append([float(value) for value in state.qvel[0].tolist()])
    Device[Device.DEFAULT].synchronize()
    rollout_wall_seconds = time.perf_counter() - rollout_started
    assert simulator._inference_step is not None
    jit_bounded, jit_calls = _bounded_jit(simulator._inference_step)

    expected = simulator.step(state, control)
    Tensor.realize(expected.qpos, expected.qvel)
    actual = simulator.inference_step(state, control)
    eager_matches = bool(
        actual.qpos.isclose(expected.qpos, rtol=1e-5, atol=1e-6).all().item()
        and actual.qvel.isclose(expected.qvel, rtol=1e-5, atol=1e-6).all().item()
    )
    quaternion_error = max(
        abs(sqrt(sum(value * value for value in row[3:7])) - 1.0)
        for row in positions
    )
    linear_momenta = [
        tuple(mass * value for value in row[:3]) for row in velocities
    ]

    def angular_momentum(
        qpos: Sequence[float], qvel: Sequence[float]
    ) -> tuple[float, float, float]:
        quaternion = qpos[3:7]
        conjugate = (
            quaternion[0],
            -quaternion[1],
            -quaternion[2],
            -quaternion[3],
        )
        local_omega = _rotate_host(conjugate, qvel[3:6])
        return _rotate_host(
            quaternion,
            tuple(
                inertia[index] * local_omega[index] for index in range(3)
            ),
        )

    angular_momenta = [
        angular_momentum(qpos, qvel)
        for qpos, qvel in zip(positions, velocities, strict=True)
    ]
    linear_drift = max(
        abs(value - linear_momenta[0][axis])
        for row in linear_momenta
        for axis, value in enumerate(row)
    )
    angular_drift = max(
        abs(value - angular_momenta[0][axis])
        for row in angular_momenta
        for axis, value in enumerate(row)
    )
    trajectory = Trajectory(
        scenario="free_body",
        timestep=timestep,
        qpos=positions,
        qvel=velocities,
        controls=[[] for _ in range(steps)],
        metadata={
            "seed": 0,
            "integrator": "semi_implicit_euler",
            "worlds": 1,
            "qpos_layout": "xyz,wxyz",
            "qvel_layout": "world_linear,world_angular",
            "contact": "none",
        },
    )
    return _write_evidence(
        output,
        trajectory,
        checks={
            "finite_state": _finite(positions) and _finite(velocities),
            "quaternion_is_normalized": quaternion_error < 1e-5,
            "linear_momentum_is_conserved": linear_drift < 1e-6,
            "angular_momentum_is_conserved": angular_drift < 1e-4,
            "eager_matches_tinyjit": eager_matches,
            "bounded_tinyjit_graph": jit_bounded,
        },
        metrics={
            "maximum_quaternion_norm_error": quaternion_error,
            "maximum_linear_momentum_drift": linear_drift,
            "maximum_angular_momentum_drift": angular_drift,
            "tinyjit_calls": float(jit_calls),
        },
        configuration={
            "seed": 0,
            "steps": steps,
            "timestep": timestep,
            "mass": mass,
            "inertia": ",".join(str(value) for value in inertia),
            "initial_qvel": ",".join(str(value) for value in initial_qvel),
            "integrator": "semi_implicit_euler",
            "contact": "none",
            "worlds": 1,
        },
        reference_tensor=state.qpos,
        render_frame=lambda row: render_free_body(
            row, width=width, height=height
        ),
        record=record,
        width=width,
        height=height,
        fps=fps,
        rollout_wall_seconds=rollout_wall_seconds,
    )


def verify_batched_worlds(
    output: str | Path,
    *,
    steps: int = 240,
    timestep: float = 0.01,
    record: str | Path | None = None,
    width: int = 640,
    height: int = 480,
    fps: int = 60,
) -> VerificationReport:
    """Verify selected independent pendulum worlds in one batched rollout."""
    if steps < 3:
        raise ValueError("steps must be at least three")
    if timestep <= 0 or not isfinite(timestep):
        raise ValueError("timestep must be finite and positive")
    initial_angles = [0.1, 0.1, 0.2, -0.2]
    angle = Tensor([[value] for value in initial_angles]).realize()
    velocity = Tensor.zeros(4, 1).realize()
    control = Tensor.zeros(4, 1).realize()
    rollout_step = make_jitted_pendulum_step(damping=0.0, dt=timestep)
    positions = [initial_angles]
    velocities = [[0.0] * 4]
    Device[Device.DEFAULT].synchronize()
    rollout_started = time.perf_counter()
    for _ in range(steps):
        angle, velocity = rollout_step(angle, velocity, control)
        positions.append([float(row[0]) for row in angle.tolist()])
        velocities.append([float(row[0]) for row in velocity.tolist()])
    Device[Device.DEFAULT].synchronize()
    rollout_wall_seconds = time.perf_counter() - rollout_started
    jit_bounded, jit_calls = _bounded_jit(rollout_step)
    expected = pendulum_step(
        angle, velocity, control, damping=0.0, dt=timestep
    )
    Tensor.realize(*expected)
    actual = rollout_step(angle, velocity, control)
    eager_matches = all(
        bool(got.isclose(want, rtol=1e-5, atol=1e-6).all().item())
        for got, want in zip(actual, expected)
    )
    duplicate_error = max(abs(row[0] - row[1]) for row in positions)
    mirror_error = max(abs(row[2] + row[3]) for row in positions)
    trajectory = Trajectory(
        scenario="batched_worlds",
        timestep=timestep,
        qpos=positions,
        qvel=velocities,
        controls=[[0.0] * 4 for _ in range(steps)],
        metadata={
            "seed": 0,
            "integrator": "semi_implicit_euler",
            "worlds": 4,
            "coordinate_layout": "one pendulum angle per world",
            "selected_worlds": [0, 1, 2, 3],
        },
    )
    return _write_evidence(
        output,
        trajectory,
        checks={
            "finite_state": _finite(positions) and _finite(velocities),
            "duplicate_worlds_remain_identical": duplicate_error < 1e-7,
            "mirrored_worlds_remain_mirrored": mirror_error < 1e-6,
            "eager_matches_tinyjit": eager_matches,
            "bounded_tinyjit_graph": jit_bounded,
        },
        metrics={
            "maximum_duplicate_world_error": duplicate_error,
            "maximum_mirror_world_error": mirror_error,
            "tinyjit_calls": float(jit_calls),
        },
        configuration={
            "seed": 0,
            "steps": steps,
            "timestep": timestep,
            "initial_angles": "0.1,0.1,0.2,-0.2",
            "mass": 1.0,
            "length": 1.0,
            "damping": 0.0,
            "gravity": 9.81,
            "integrator": "semi_implicit_euler",
            "worlds": 4,
        },
        reference_tensor=angle,
        render_frame=lambda row: render_batched_pendulums(
            row, width=width, height=height
        ),
        record=record,
        width=width,
        height=height,
        fps=fps,
        rollout_wall_seconds=rollout_wall_seconds,
    )


_VERIFIERS: dict[str, Callable[..., VerificationReport]] = {
    "pendulum": verify_pendulum,
    "cartpole": verify_cartpole,
    "bouncing-ball": verify_bouncing_ball,
    "imported-articulated": verify_imported_articulated,
    "free-body": verify_free_body,
    "batched-worlds": verify_batched_worlds,
}


def write_release_report(
    output: str | Path,
    reports: Sequence[VerificationReport],
) -> Path:
    """Writes one machine-readable index over all numerical and visual evidence."""

    if not reports:
        raise ValueError("release report requires at least one scenario")
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    path = output_path / "release-report.json"
    source_sha = reports[0].environment["tinysim_source_sha256"]
    project_sha = reports[0].environment["tinysim_project_source_sha256"]
    if any(
        report.environment["tinysim_source_sha256"] != source_sha
        for report in reports
    ):
        raise ValueError("scenario reports come from different TinySim sources")
    if any(
        report.environment["tinysim_project_source_sha256"] != project_sha
        for report in reports
    ):
        raise ValueError("scenario reports come from different project sources")
    artifact_root = output_path.parent

    def read_json(candidate: Path) -> dict[str, object] | None:
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else None
        except (OSError, json.JSONDecodeError):
            return None

    paths = {
        "cpu_validation": artifact_root / "validation" / "cpu-tests.json",
        "performance": artifact_root / "benchmarks" / "cpu-quick.json",
        "mujoco_reference": artifact_root / "reference" / "mujoco.json",
        "capabilities": artifact_root / "capabilities.json",
        "critic_decision": artifact_root / "review" / "critic.json",
        "manager_video_review": (
            artifact_root / "review" / "manager-video-review.json"
        ),
        "golden_clip": artifact_root / "golden" / "pendulum.mp4",
    }
    documents = {
        name: read_json(candidate)
        for name, candidate in paths.items()
        if candidate.suffix == ".json"
    }
    performance = documents["performance"]
    benchmark_source_sha = source_tree_sha256(
        Path(__file__).resolve().parent.parent / "benchmarks"
    )
    validation_source_sha = source_tree_sha256(
        Path(__file__).resolve().parent.parent / "tests"
    )
    cpu_validation = documents["cpu_validation"]
    mujoco = documents["mujoco_reference"]
    critic = documents["critic_decision"]
    manager_review = documents["manager_video_review"]
    golden = paths["golden_clip"]
    golden_sha = (
        sha256(golden.read_bytes()).hexdigest()
        if golden.is_file() and golden.stat().st_size > 0
        else None
    )
    required_scenarios = {
        "pendulum",
        "cartpole",
        "bouncing_ball",
        "imported_articulated",
        "free_body",
        "batched_worlds",
    }
    scenario_names = [report.scenario for report in reports]
    scenario_checks_passed = (
        len(scenario_names) == len(required_scenarios)
        and set(scenario_names) == required_scenarios
        and all(report.passed for report in reports)
    )
    required_workloads = {
        "pendulum",
        "cartpole",
        "quadruped",
        "humanoid",
        "bouncing_contact",
        "locomotion_policy",
    }
    performance_cases = performance.get("cases", []) if performance else []
    performance_protocol = (
        performance.get("protocol", {}) if performance else {}
    )
    accelerator_reports = [
        report
        for report in reports
        if report.scenario == "batched_worlds"
        and report.configuration.get("worlds", 0) > 1
        and report.environment.get("backend") != "CPU"
        and report.passed
        and "video" in report.artifacts
    ]
    accelerator_video = (
        Path(accelerator_reports[0].artifacts["video"])
        if len(accelerator_reports) == 1
        else None
    )
    accelerator_video_sha = (
        sha256(accelerator_video.read_bytes()).hexdigest()
        if accelerator_video is not None
        and accelerator_video.is_file()
        and accelerator_video.stat().st_size > 0
        else None
    )
    gates = {
        "scenario_checks": scenario_checks_passed,
        "cpu_validation": bool(
            cpu_validation
            and cpu_validation.get("passed") is True
            and cpu_validation.get("tinysim_source_sha256") == source_sha
            and cpu_validation.get("project_source_sha256") == project_sha
            and cpu_validation.get("validation_source_sha256")
            == validation_source_sha
            and cpu_validation.get("device") == "CPU"
            and isinstance(cpu_validation.get("tests_run"), int)
            and cpu_validation["tests_run"] > 0
        ),
        "performance": bool(
            performance
            and isinstance(performance.get("environment"), dict)
            and performance["environment"].get("tinysim_source_sha256")
            == source_sha
            and performance["environment"].get("project_source_sha256")
            == project_sha
            and performance["environment"].get("benchmark_source_sha256")
            == benchmark_source_sha
            and performance["environment"].get("device") == "CPU"
            and performance.get("failures") == []
            and isinstance(performance_cases, list)
            and len(performance_cases) == len(required_workloads)
            and {
                case.get("workload")
                for case in performance_cases
                if isinstance(case, dict)
            }
            == required_workloads
            and all(
                isinstance(case, dict) and case.get("worlds") == 1
                for case in performance_cases
            )
            and isinstance(performance_protocol, dict)
            and set(performance_protocol.get("workloads_requested", []))
            == required_workloads
            and performance_protocol.get("batches_requested") == [1]
        ),
        "mujoco_reference": bool(
            mujoco
            and mujoco.get("passed") is True
            and mujoco.get("tinysim_source_sha256") == source_sha
            and mujoco.get("project_source_sha256") == project_sha
            and mujoco.get("validation_source_sha256")
            == validation_source_sha
            and mujoco.get("device") == "CPU"
            and isinstance(mujoco.get("tests_run"), int)
            and mujoco["tests_run"] > 0
            and isinstance(mujoco.get("backend"), dict)
            and mujoco["backend"].get("available") is True
        ),
        "critic_accept": bool(
            critic
            and critic.get("decision") == "ACCEPT"
            and critic.get("tinysim_source_sha256") == source_sha
            and critic.get("project_source_sha256") == project_sha
        ),
        "accelerator_recording": accelerator_video_sha is not None,
        "golden_clip": bool(
            golden_sha
            and manager_review
            and manager_review.get("golden_video_sha256") == golden_sha
        ),
        "manager_video_review": bool(
            manager_review
            and manager_review.get("decision") == "ACCEPT"
            and manager_review.get("signed_by") == "manager"
            and manager_review.get("tinysim_source_sha256") == source_sha
            and manager_review.get("project_source_sha256") == project_sha
            and manager_review.get("golden_video_sha256") == golden_sha
            and manager_review.get("accelerator_video_sha256")
            == accelerator_video_sha
        ),
    }
    evidence: dict[str, object] = {
        "numerical_gradient_visual": {
            "scenario_manifests": [
                report.artifacts["manifest"] for report in reports
            ],
            "gradient_contracts": [
                "tests/test_analytical.py",
                "tests/test_invariants_gradients.py",
                "tests/test_collision_contact.py",
            ],
        },
        "links": {
            name: {"path": str(candidate), "exists": candidate.is_file()}
            for name, candidate in paths.items()
        },
        "accelerator_video": {
            "path": str(accelerator_video) if accelerator_video else None,
            "sha256": accelerator_video_sha,
            "size_bytes": (
                accelerator_video.stat().st_size
                if accelerator_video_sha and accelerator_video
                else 0
            ),
        },
    }
    unmet_gates = [name for name, passed in gates.items() if not passed]
    payload = {
        "schema_version": 1,
        "passed": scenario_checks_passed,
        "release_ready": all(gates.values()),
        "release_gates": gates,
        "unmet_release_gates": unmet_gates,
        "tinysim_source_sha256": source_sha,
        "project_source_sha256": project_sha,
        "scenario_count": len(reports),
        "scenarios": [
            {
                "name": report.scenario,
                "passed": report.passed,
                "checks": report.checks,
                "metrics": report.metrics,
                "trajectory_sha256": report.trajectory_sha256,
                "artifacts": report.artifacts,
                "environment": report.environment,
            }
            for report in reports
        ],
        "evidence": evidence,
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def load_verification_report(path: str | Path) -> VerificationReport:
    """Loads one source-bound manifest without rerunning physics or rendering."""

    manifest = Path(path)
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("manifest root must be an object")
        checks = payload["checks"]
        artifacts = payload["artifacts"]
        metrics = payload["metrics"]
        environment = payload["environment"]
        configuration = payload["configuration"]
        if (
            not isinstance(payload["scenario"], str)
            or not payload["scenario"]
            or not isinstance(checks, dict)
            or not checks
            or not all(
                isinstance(name, str) and isinstance(value, bool)
                for name, value in checks.items()
            )
            or not isinstance(payload["passed"], bool)
            or payload["passed"] != all(checks.values())
            or not all(
                isinstance(value, dict)
                for value in (
                    artifacts,
                    metrics,
                    environment,
                    configuration,
                )
            )
        ):
            raise TypeError("manifest fields are internally inconsistent")
        report = VerificationReport(
            passed=payload["passed"],
            scenario=payload["scenario"],
            checks=dict(checks),
            metrics=dict(metrics),
            artifacts={**dict(artifacts), "manifest": str(manifest)},
            trajectory_sha256=payload["trajectory_sha256"],
            environment=dict(environment),
            configuration=dict(configuration),
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid verification manifest: {manifest}") from error
    project = Path(__file__).resolve().parent.parent
    if (
        report.environment.get("tinysim_source_sha256") != _source_revision()
        or report.environment.get("tinysim_project_source_sha256")
        != project_source_sha256(project)
    ):
        raise ValueError(f"stale verification manifest: {manifest}")
    for artifact in report.artifacts.values():
        candidate = Path(artifact)
        if not candidate.is_file() or candidate.stat().st_size == 0:
            raise ValueError(
                f"missing or empty verification artifact: {candidate}"
            )
    trajectory = Path(report.artifacts["trajectory"])
    if sha256(trajectory.read_bytes()).hexdigest() != report.trajectory_sha256:
        raise ValueError(f"trajectory hash mismatch: {trajectory}")
    return report


def refresh_release_report(output: str | Path) -> Path:
    """Re-indexes existing scenario evidence without modifying encoded videos."""

    output_path = Path(output)
    manifests = sorted(output_path.glob("*/*.manifest.json"))
    reports = [load_verification_report(path) for path in manifests]
    if not reports:
        raise ValueError(f"no scenario manifests found under {output_path}")
    return write_release_report(output_path, reports)


def main() -> None:
    parser = ArgumentParser(
        description="Generate deterministic TinySim numerical and visual evidence."
    )
    parser.add_argument(
        "--scenario", choices=(*_VERIFIERS, "all")
    )
    parser.add_argument("--output", default="artifacts/verify")
    parser.add_argument(
        "--record",
        help="optional single-scenario MP4 destination; requires an encoder",
    )
    parser.add_argument(
        "--accelerator-record",
        help="batched-world MP4 destination for an all-scenario accelerator run",
    )
    parser.add_argument(
        "--refresh-release",
        action="store_true",
        help="re-index existing source-bound manifests without rerunning scenarios",
    )
    parser.add_argument("--steps", type=int)
    args = parser.parse_args()
    if args.refresh_release:
        if (
            args.scenario is not None
            or args.record
            or args.accelerator_record
            or args.steps is not None
        ):
            parser.error(
                "--refresh-release does not run scenarios or encode videos"
            )
        release_report = refresh_release_report(args.output)
        release = json.loads(release_report.read_text(encoding="utf-8"))
        print(json.dumps(release, indent=2, sort_keys=True))
        if release.get("release_ready") is not True:
            raise SystemExit(2)
        return
    scenario = args.scenario or "pendulum"
    if scenario == "all":
        if args.record:
            parser.error("--record requires a single scenario")
        if args.accelerator_record and Device.DEFAULT == "CPU":
            parser.error("--accelerator-record requires a non-CPU backend")
        reports = [
            verifier(
                Path(args.output) / name,
                **(
                    {"record": args.accelerator_record}
                    if name == "batched-worlds" and args.accelerator_record
                    else {}
                ),
                **({} if args.steps is None else {"steps": args.steps}),
            )
            for name, verifier in _VERIFIERS.items()
        ]
        release_report = write_release_report(args.output, reports)
        print(json.dumps([asdict(report) for report in reports], indent=2, sort_keys=True))
        print(f"release_report={release_report}")
        if not all(report.passed for report in reports):
            raise SystemExit(1)
        release = json.loads(release_report.read_text(encoding="utf-8"))
        if release.get("release_ready") is not True:
            raise SystemExit(2)
        return
    if args.accelerator_record:
        parser.error("--accelerator-record requires --scenario all")
    report = _VERIFIERS[scenario](
        args.output,
        record=args.record,
        **({} if args.steps is None else {"steps": args.steps}),
    )
    print(json.dumps(asdict(report), indent=2, sort_keys=True))
    if not report.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
