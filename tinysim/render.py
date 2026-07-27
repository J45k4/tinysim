"""Optional deterministic software rendering and video encoding.

This module consumes host-side trajectories. Nothing in the physics core imports it.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from math import cos, isfinite, sin, sqrt
import os
from pathlib import Path
import shutil
import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .compile import CompiledModel


RGB = tuple[int, int, int]


@dataclass(frozen=True)
class Camera2D:
    """Side-view camera controls in world coordinates."""

    center_x: float = 0.0
    center_z: float = 0.0
    zoom: float = 1.0

    def __post_init__(self) -> None:
        if (
            not all(isfinite(value) for value in (self.center_x, self.center_z, self.zoom))
            or self.zoom <= 0
        ):
            raise ValueError("camera center/zoom must be finite and zoom positive")


Vector3 = tuple[float, float, float]
Quaternion = tuple[float, float, float, float]


def _quat_normalize(quaternion: Quaternion) -> Quaternion:
    length = sqrt(sum(value * value for value in quaternion))
    if length <= 1e-12:
        raise ValueError("render quaternion must be nonzero")
    w, x, y, z = quaternion
    return w / length, x / length, y / length, z / length


def _quat_multiply(left: Quaternion, right: Quaternion) -> Quaternion:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return _quat_normalize(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        )
    )


def _quat_rotate(quaternion: Quaternion, vector: Vector3) -> Vector3:
    w, x, y, z = _quat_normalize(quaternion)
    vx, vy, vz = vector
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (
        vx + w * tx + y * tz - z * ty,
        vy + w * ty + z * tx - x * tz,
        vz + w * tz + x * ty - y * tx,
    )


def _vector_add(left: Vector3, right: Vector3) -> Vector3:
    return (
        left[0] + right[0],
        left[1] + right[1],
        left[2] + right[2],
    )


def _vector3(values: Sequence[float]) -> Vector3:
    if len(values) != 3:
        raise ValueError("render vector must contain three values")
    return float(values[0]), float(values[1]), float(values[2])


def _quaternion(values: Sequence[float]) -> Quaternion:
    if len(values) != 4:
        raise ValueError("render quaternion must contain four values")
    return (
        float(values[0]),
        float(values[1]),
        float(values[2]),
        float(values[3]),
    )


def _axis_angle(axis: Vector3, angle: float) -> Quaternion:
    half = 0.5 * angle
    scale = sin(half)
    return _quat_normalize(
        (cos(half), scale * axis[0], scale * axis[1], scale * axis[2])
    )


def _model_body_transforms(
    model: "CompiledModel", qpos: Sequence[float]
) -> tuple[list[Vector3], list[Quaternion]]:
    if len(qpos) != model.nq:
        raise ValueError(f"model render qpos must contain {model.nq} values")
    positions: list[Vector3] = []
    quaternions: list[Quaternion] = []
    for body_index, parent in enumerate(model.body_parent):
        parent_position = (0.0, 0.0, 0.0) if parent == -1 else positions[parent]
        parent_quaternion = (
            (1.0, 0.0, 0.0, 0.0)
            if parent == -1
            else quaternions[parent]
        )
        joint_index = model.joint_for_body[body_index]
        joint = model.joints[joint_index]
        origin = _vector_add(
            parent_position,
            _quat_rotate(parent_quaternion, joint.pos),
        )
        base = _quat_multiply(parent_quaternion, joint.quat)
        address = model.joint_qpos[joint_index]
        if joint.kind == "hinge":
            position = origin
            quaternion = _quat_multiply(
                base, _axis_angle(joint.axis, float(qpos[address]))
            )
        elif joint.kind == "slide":
            displacement = _quat_rotate(base, joint.axis)
            position = _vector_add(
                origin,
                (
                    float(qpos[address]) * displacement[0],
                    float(qpos[address]) * displacement[1],
                    float(qpos[address]) * displacement[2],
                ),
            )
            quaternion = base
        elif joint.kind == "ball":
            position = origin
            quaternion = _quat_multiply(
                base,
                _quaternion(qpos[address : address + 4]),
            )
        elif joint.kind == "free":
            position = _vector_add(
                origin,
                _quat_rotate(
                    base,
                    _vector3(qpos[address : address + 3]),
                ),
            )
            quaternion = _quat_multiply(
                base,
                _quaternion(qpos[address + 3 : address + 7]),
            )
        else:
            position, quaternion = origin, base
        positions.append(position)
        quaternions.append(quaternion)
    return positions, quaternions


def _pixel(frame: bytearray, width: int, height: int, x: int, y: int, color: RGB) -> None:
    if 0 <= x < width and 0 <= y < height:
        offset = (y * width + x) * 3
        frame[offset : offset + 3] = bytes(color)


def _line(
    frame: bytearray,
    width: int,
    height: int,
    start: tuple[int, int],
    end: tuple[int, int],
    color: RGB,
    thickness: int = 1,
) -> None:
    """Integer Bresenham line with deterministic rasterization."""
    x0, y0 = start
    x1, y1 = end
    dx, sx = abs(x1 - x0), 1 if x0 < x1 else -1
    dy, sy = -abs(y1 - y0), 1 if y0 < y1 else -1
    error = dx + dy
    radius = max(0, thickness // 2)
    while True:
        for oy in range(-radius, radius + 1):
            for ox in range(-radius, radius + 1):
                _pixel(frame, width, height, x0 + ox, y0 + oy, color)
        if x0 == x1 and y0 == y1:
            return
        doubled = 2 * error
        if doubled >= dy:
            error += dy
            x0 += sx
        if doubled <= dx:
            error += dx
            y0 += sy


def _circle(
    frame: bytearray,
    width: int,
    height: int,
    center: tuple[int, int],
    radius: int,
    color: RGB,
) -> None:
    cx, cy = center
    radius_sq = radius * radius
    for y in range(cy - radius, cy + radius + 1):
        for x in range(cx - radius, cx + radius + 1):
            if (x - cx) ** 2 + (y - cy) ** 2 <= radius_sq:
                _pixel(frame, width, height, x, y, color)


def _rectangle(
    frame: bytearray,
    width: int,
    height: int,
    corner: tuple[int, int],
    size: tuple[int, int],
    color: RGB,
) -> None:
    left, top = corner
    rectangle_width, rectangle_height = size
    for y in range(top, top + rectangle_height):
        for x in range(left, left + rectangle_width):
            _pixel(frame, width, height, x, y, color)


def _frame(width: int, height: int) -> bytearray:
    if width < 16 or height < 16:
        raise ValueError("frame dimensions must be at least 16 pixels")
    return bytearray(bytes((245, 247, 250)) * (width * height))


def render_model(
    model: "CompiledModel",
    qpos: Sequence[float],
    *,
    width: int = 640,
    height: int = 480,
    camera: Camera2D = Camera2D(),
) -> bytes:
    """Render the compiled primitive scene from a host-side qpos snapshot."""

    frame = _frame(width, height)
    body_positions, body_quaternions = _model_body_transforms(model, qpos)
    scale = min(width, height) * camera.zoom / 3.0

    def project(position: Vector3) -> tuple[int, int]:
        return (
            width // 2 + round((position[0] - camera.center_x) * scale),
            height // 2 - round((position[2] - camera.center_z) * scale),
        )

    for index, geom in enumerate(model.geoms):
        if geom.body == -1:
            position, quaternion = geom.pos, geom.quat
        else:
            body_position = body_positions[geom.body]
            body_quaternion = body_quaternions[geom.body]
            position = _vector_add(
                body_position,
                _quat_rotate(body_quaternion, geom.pos),
            )
            quaternion = _quat_multiply(body_quaternion, geom.quat)
        color = (
            (220, 65, 55)
            if index % 3 == 0
            else (65, 95, 130)
            if index % 3 == 1
            else (55, 150, 90)
        )
        if geom.kind == "sphere":
            _circle(
                frame,
                width,
                height,
                project(position),
                max(2, round(geom.size[0] * scale)),
                color,
            )
        elif geom.kind == "capsule":
            radius = geom.size[0]
            half_length = geom.size[1]
            axis = _quat_rotate(quaternion, (0.0, 0.0, half_length))
            first = _vector_add(
                position, (-axis[0], -axis[1], -axis[2])
            )
            second = _vector_add(position, axis)
            radius_pixels = max(2, round(radius * scale))
            _line(
                frame,
                width,
                height,
                project(first),
                project(second),
                color,
                thickness=2 * radius_pixels,
            )
            _circle(
                frame, width, height, project(first), radius_pixels, color
            )
            _circle(
                frame, width, height, project(second), radius_pixels, color
            )
        elif geom.kind == "box":
            half_x, half_y, half_z = geom.size
            local_vertices = [
                (x * half_x, y * half_y, z * half_z)
                for x in (-1.0, 1.0)
                for y in (-1.0, 1.0)
                for z in (-1.0, 1.0)
            ]
            vertices = [
                project(
                    _vector_add(
                        position, _quat_rotate(quaternion, vertex)
                    )
                )
                for vertex in local_vertices
            ]
            for left, right in (
                (0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
                (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7),
            ):
                _line(
                    frame,
                    width,
                    height,
                    vertices[left],
                    vertices[right],
                    color,
                    thickness=2,
                )
        else:
            normal = _quat_rotate(quaternion, (0.0, 0.0, 1.0))
            direction_x, direction_z = normal[2], -normal[0]
            length = sqrt(
                direction_x * direction_x + direction_z * direction_z
            )
            if length <= 1e-12:
                direction_x, direction_z, length = 1.0, 0.0, 1.0
            span = max(width, height) / scale
            direction_x *= span / length
            direction_z *= span / length
            _line(
                frame,
                width,
                height,
                project(
                    (
                        position[0] - direction_x,
                        position[1],
                        position[2] - direction_z,
                    )
                ),
                project(
                    (
                        position[0] + direction_x,
                        position[1],
                        position[2] + direction_z,
                    )
                ),
                (145, 150, 158),
                thickness=3,
            )
    return bytes(frame)


def render_model_grid(
    model: "CompiledModel",
    qposes: Sequence[Sequence[float]],
    *,
    width: int = 640,
    height: int = 480,
    columns: int | None = None,
    camera: Camera2D = Camera2D(),
) -> bytes:
    """Render several worlds as tiles in one RGB frame."""

    if not qposes:
        raise ValueError("grid must contain at least one world")
    count = len(qposes)
    if columns is None:
        columns = max(1, int(sqrt(count - 1)) + 1)
    if columns < 1 or columns > count:
        raise ValueError("grid columns must be between one and the world count")
    rows = (count + columns - 1) // columns
    if width // columns < 16 or height // rows < 16:
        raise ValueError("grid tiles must be at least 16 pixels in each dimension")

    frame = _frame(width, height)
    for index, qpos in enumerate(qposes):
        column, row = index % columns, index // columns
        left, right = column * width // columns, (column + 1) * width // columns
        top, bottom = row * height // rows, (row + 1) * height // rows
        tile_width, tile_height = right - left, bottom - top
        tile = render_model(
            model,
            qpos,
            width=tile_width,
            height=tile_height,
            camera=camera,
        )
        for tile_y in range(tile_height):
            source = tile_y * tile_width * 3
            destination = ((top + tile_y) * width + left) * 3
            frame[destination : destination + tile_width * 3] = tile[
                source : source + tile_width * 3
            ]

    for column in range(1, columns):
        x = column * width // columns
        _line(
            frame,
            width,
            height,
            (x, 0),
            (x, height - 1),
            (180, 185, 190),
        )
    for row in range(1, rows):
        y = row * height // rows
        _line(
            frame,
            width,
            height,
            (0, y),
            (width - 1, y),
            (180, 185, 190),
        )
    return bytes(frame)


def render_pendulum(
    angle: float,
    *,
    width: int = 640,
    height: int = 480,
    length_pixels: int | None = None,
) -> bytes:
    """Renders one RGB frame; angle zero points down."""
    from math import cos, sin

    frame = _frame(width, height)
    pivot = (width // 2, height // 4)
    length = length_pixels or min(width, height) // 3
    bob = (
        pivot[0] + round(length * sin(angle)),
        pivot[1] + round(length * cos(angle)),
    )
    _line(frame, width, height, (0, 3 * height // 4), (width - 1, 3 * height // 4), (180, 185, 190))
    _line(frame, width, height, pivot, bob, (30, 45, 65), thickness=5)
    _circle(frame, width, height, pivot, 7, (65, 95, 130))
    _circle(frame, width, height, bob, 14, (220, 65, 55))
    return bytes(frame)


def render_cartpole(
    position: float,
    angle: float,
    *,
    width: int = 640,
    height: int = 480,
) -> bytes:
    """Render one cart-pole frame; angle zero is the upright configuration."""
    from math import cos, sin

    frame = _frame(width, height)
    track_y = 3 * height // 4
    cart_x = width // 2 + round(position * min(width, height) / 4)
    cart_size = (max(14, width // 10), max(8, height // 18))
    cart_top = track_y - cart_size[1] - 3
    pivot = (cart_x, cart_top)
    pole_length = min(width, height) // 3
    tip = (
        pivot[0] + round(pole_length * sin(angle)),
        pivot[1] - round(pole_length * cos(angle)),
    )
    _line(frame, width, height, (0, track_y), (width - 1, track_y), (145, 150, 158), thickness=3)
    _rectangle(
        frame,
        width,
        height,
        (cart_x - cart_size[0] // 2, cart_top),
        cart_size,
        (65, 95, 130),
    )
    _line(frame, width, height, pivot, tip, (30, 45, 65), thickness=5)
    _circle(frame, width, height, pivot, max(3, min(width, height) // 70), (220, 65, 55))
    return bytes(frame)


def render_bouncing_ball(
    position: Sequence[float],
    *,
    width: int = 640,
    height: int = 480,
    radius: float = 0.25,
) -> bytes:
    """Render a sphere above the z=0 plane from a fixed side camera."""
    if len(position) != 3:
        raise ValueError("ball position must contain x, y, and z")
    frame = _frame(width, height)
    ground_y = 4 * height // 5
    scale = min(width, height) / 2.5
    center = (
        width // 2 + round(float(position[0]) * scale),
        ground_y - round(float(position[2]) * scale),
    )
    _line(frame, width, height, (0, ground_y), (width - 1, ground_y), (145, 150, 158), thickness=3)
    _circle(frame, width, height, center, max(3, round(radius * scale)), (220, 65, 55))
    return bytes(frame)


def render_articulated(
    coordinates: Sequence[float],
    *,
    width: int = 640,
    height: int = 480,
) -> bytes:
    """Render a planar serial chain from generalized hinge coordinates."""
    from math import cos, sin

    if not coordinates:
        raise ValueError("articulated coordinates must not be empty")
    frame = _frame(width, height)
    origin = (width // 2, height // 5)
    link_length = max(8, min(width, height) // (len(coordinates) + 2))
    start = origin
    cumulative_angle = 0.0
    _circle(frame, width, height, origin, max(3, min(width, height) // 70), (65, 95, 130))
    for index, coordinate in enumerate(coordinates):
        cumulative_angle += float(coordinate)
        end = (
            start[0] + round(link_length * sin(cumulative_angle)),
            start[1] + round(link_length * cos(cumulative_angle)),
        )
        _line(frame, width, height, start, end, (30, 45, 65), thickness=5)
        _circle(
            frame,
            width,
            height,
            end,
            max(3, min(width, height) // 80),
            (220, 65, 55) if index == len(coordinates) - 1 else (65, 95, 130),
        )
        start = end
    return bytes(frame)


def render_free_body(
    qpos: Sequence[float],
    *,
    width: int = 640,
    height: int = 480,
    camera: Camera2D = Camera2D(),
) -> bytes:
    """Render the orientation axes of one free body from a side camera."""
    if len(qpos) != 7:
        raise ValueError("free-body qpos must contain xyz and a wxyz quaternion")
    from math import sqrt

    x, _, z, w, qx, qy, qz = map(float, qpos)
    norm = sqrt(w * w + qx * qx + qy * qy + qz * qz)
    if norm <= 1e-12:
        raise ValueError("free-body quaternion must be nonzero")
    w, qx, qy, qz = (value / norm for value in (w, qx, qy, qz))

    def rotate_axis(axis: tuple[float, float, float]) -> tuple[float, float, float]:
        vx, vy, vz = axis
        tx = 2.0 * (qy * vz - qz * vy)
        ty = 2.0 * (qz * vx - qx * vz)
        tz = 2.0 * (qx * vy - qy * vx)
        return (
            vx + w * tx + qy * tz - qz * ty,
            vy + w * ty + qz * tx - qx * tz,
            vz + w * tz + qx * ty - qy * tx,
        )

    frame = _frame(width, height)
    scale = min(width, height) * camera.zoom / 3.0

    def project(world_x: float, world_z: float) -> tuple[int, int]:
        return (
            width // 2 + round((world_x - camera.center_x) * scale),
            height // 2 - round((world_z - camera.center_z) * scale),
        )

    center = project(x, z)
    axes = (
        ((1.0, 0.0, 0.0), (210, 55, 55)),
        ((0.0, 1.0, 0.0), (55, 170, 75)),
        ((0.0, 0.0, 1.0), (55, 90, 210)),
    )
    for axis, color in axes:
        direction = rotate_axis(axis)
        endpoint = project(x + 0.55 * direction[0], z + 0.55 * direction[2])
        _line(frame, width, height, center, endpoint, color, thickness=4)
    _circle(frame, width, height, center, max(5, min(width, height) // 45), (30, 45, 65))
    return bytes(frame)


def render_batched_pendulums(
    angles: Sequence[float],
    *,
    width: int = 640,
    height: int = 480,
) -> bytes:
    """Render selected batched worlds as deterministic horizontal tiles."""
    if not angles:
        raise ValueError("angles must not be empty")
    frame = _frame(width, height)
    tile_width = width // len(angles)
    if tile_width < 16:
        raise ValueError("frame is too narrow for the requested world count")
    for index, angle in enumerate(angles):
        actual_width = width - index * tile_width if index == len(angles) - 1 else tile_width
        tile = render_pendulum(float(angle), width=actual_width, height=height)
        for y in range(height):
            source = y * actual_width * 3
            destination = (y * width + index * tile_width) * 3
            frame[destination : destination + actual_width * 3] = tile[source : source + actual_width * 3]
        if index:
            _line(
                frame,
                width,
                height,
                (index * tile_width, 0),
                (index * tile_width, height - 1),
                (180, 185, 190),
            )
    return bytes(frame)


def write_ppm(path: str | Path, rgb: bytes, *, width: int, height: int) -> None:
    if len(rgb) != width * height * 3:
        raise ValueError("RGB payload size does not match dimensions")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(f"P6\n{width} {height}\n255\n".encode() + rgb)


def encode_mp4(
    frames: Iterable[bytes],
    path: str | Path,
    *,
    width: int,
    height: int,
    fps: int = 60,
    ffmpeg: str = "ffmpeg",
    gstreamer: str = "gst-launch-1.0",
) -> None:
    """Streams raw RGB frames to an isolated encoder without buffering them."""
    if fps <= 0:
        raise ValueError("fps must be positive")
    if width <= 0 or height <= 0 or width % 2 or height % 2:
        raise ValueError("MP4 dimensions must be positive even integers")

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if executable := shutil.which(ffmpeg):
        encoder = "FFmpeg"
        command = [
            executable,
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            str(fps),
            "-i",
            "-",
            "-an",
            "-vcodec",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(destination),
        ]
    elif executable := shutil.which(gstreamer):
        encoder = "GStreamer"
        command = [
            executable,
            "-q",
            "fdsrc",
            "fd=0",
            "!",
            "videoparse",
            "format=rgb",
            f"width={width}",
            f"height={height}",
            f"framerate={fps}/1",
            "!",
            "videoconvert",
            "!",
            "openh264enc",
            "!",
            "h264parse",
            "!",
            "mp4mux",
            "!",
            "filesink",
            f"location={destination}",
        ]
    else:
        raise RuntimeError(
            "MP4 export requires FFmpeg or GStreamer with OpenH264; "
            "use PPM frames when neither is installed"
        )
    environment = os.environ.copy()
    if Path(executable).resolve().is_relative_to("/usr"):
        environment.pop("LD_LIBRARY_PATH", None)
        environment.pop("LD_PRELOAD", None)
    probe = subprocess.run(
        [
            executable,
            "-version" if encoder == "FFmpeg" else "--version",
        ],
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=False,
    )
    if probe.returncode:
        detail = probe.stderr.decode(errors="replace").strip()
        raise RuntimeError(
            f"{encoder} failed to start with status {probe.returncode}"
            + (f": {detail}" if detail else "")
        )
    process = subprocess.Popen(
        command,
        env=environment,
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdin is not None
    assert process.stderr is not None
    expected = width * height * 3
    try:
        for frame in frames:
            if len(frame) != expected:
                raise ValueError("RGB frame size does not match dimensions")
            process.stdin.write(frame)
        process.stdin.close()
        detail = process.stderr.read().decode(errors="replace").strip()
        return_code = process.wait()
    except BrokenPipeError as error:
        try:
            process.stdin.close()
        except BrokenPipeError:
            pass
        detail = process.stderr.read().decode(errors="replace").strip()
        return_code = process.wait()
        raise RuntimeError(
            f"{encoder} exited with status {return_code}"
            + (f": {detail}" if detail else "")
        ) from error
    except BaseException:
        process.kill()
        process.wait()
        raise
    if return_code:
        raise RuntimeError(
            f"{encoder} exited with status {return_code}"
            + (f": {detail}" if detail else "")
        )


def pendulum_frames(
    angles: Sequence[float], *, width: int = 640, height: int = 480
) -> Iterable[bytes]:
    for angle in angles:
        yield render_pendulum(angle, width=width, height=height)


def cartpole_frames(
    states: Sequence[Sequence[float]], *, width: int = 640, height: int = 480
) -> Iterable[bytes]:
    for state in states:
        if len(state) != 2:
            raise ValueError("cart-pole render state must contain position and angle")
        yield render_cartpole(state[0], state[1], width=width, height=height)


def bouncing_ball_frames(
    positions: Sequence[Sequence[float]], *, width: int = 640, height: int = 480
) -> Iterable[bytes]:
    for position in positions:
        yield render_bouncing_ball(position, width=width, height=height)


def articulated_frames(
    coordinates: Sequence[Sequence[float]], *, width: int = 640, height: int = 480
) -> Iterable[bytes]:
    for qpos in coordinates:
        yield render_articulated(qpos, width=width, height=height)


def batched_pendulum_frames(
    angles: Sequence[Sequence[float]], *, width: int = 640, height: int = 480
) -> Iterable[bytes]:
    for frame_angles in angles:
        yield render_batched_pendulums(frame_angles, width=width, height=height)
