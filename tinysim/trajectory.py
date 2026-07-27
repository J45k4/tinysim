"""Portable host-side trajectory and verification artifact format."""

from array import array
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field
import json
from math import isfinite
import os
from pathlib import Path
import struct
import sys
import tempfile
from typing import Any


@dataclass(frozen=True)
class Trajectory:
    scenario: str
    timestep: float
    qpos: list[list[float]]
    qvel: list[list[float]]
    controls: list[list[float]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.scenario:
            raise ValueError("scenario must not be empty")
        if not isfinite(self.timestep) or self.timestep <= 0:
            raise ValueError("timestep must be finite and positive")
        if not self.qpos or len(self.qpos) != len(self.qvel):
            raise ValueError("qpos and qvel must contain the same nonzero frame count")
        if self.controls and len(self.controls) not in (len(self.qpos), len(self.qpos) - 1):
            raise ValueError("controls must be empty, per-frame, or per-transition")
        widths = {len(row) for row in self.qpos}
        velocity_widths = {len(row) for row in self.qvel}
        control_widths = {len(row) for row in self.controls}
        if len(widths) != 1 or len(velocity_widths) != 1 or len(control_widths) > 1:
            raise ValueError("trajectory rows must have stable dimensions")
        if not widths or next(iter(widths)) == 0:
            raise ValueError("trajectory coordinates must not be empty")
        if not velocity_widths or next(iter(velocity_widths)) == 0:
            raise ValueError("trajectory velocities must not be empty")
        values = (
            value
            for collection in (self.qpos, self.qvel, self.controls)
            for row in collection
            for value in row
        )
        if not all(isfinite(value) for value in values):
            raise ValueError("trajectory contains a non-finite value")


def save_trajectory(trajectory: Trajectory, path: str | Path) -> None:
    trajectory.validate()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(asdict(trajectory), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_trajectory(path: str | Path) -> Trajectory:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    trajectory = Trajectory(**payload)
    trajectory.validate()
    return trajectory


def iter_playback_indices(
    frame_count: int, *, timestep: float, fps: int
) -> Iterator[int]:
    """Map fixed-step states to video frames without changing playback speed.

    A trajectory with ``N`` states spans ``(N - 1) * timestep`` seconds.
    Nearest-state sampling keeps video duration within half a frame of that
    simulated duration and deliberately performs no tensor work.
    """
    if frame_count < 2:
        raise ValueError("frame_count must include at least two states")
    if not isfinite(timestep) or timestep <= 0:
        raise ValueError("timestep must be finite and positive")
    if fps <= 0:
        raise ValueError("fps must be positive")
    transitions = frame_count - 1
    video_frames = max(1, round(transitions * timestep * fps))
    for frame in range(video_frames):
        yield min(round(frame / (fps * timestep)), transitions)


def playback_indices(frame_count: int, *, timestep: float, fps: int) -> list[int]:
    """Compatibility list wrapper around :func:`iter_playback_indices`."""

    return list(
        iter_playback_indices(frame_count, timestep=timestep, fps=fps)
    )


_STREAM_MAGIC = b"TSTRAJ\x00\x00"
_STREAM_VERSION = 1
_STREAM_HEADER = struct.Struct("<8sHBBIQIIId")
_DTYPE_TO_CODE = {"float32": 1, "float64": 2}
_CODE_TO_DTYPE = {value: key for key, value in _DTYPE_TO_CODE.items()}
_DTYPE_FORMAT = {"float32": "f", "float64": "d"}
_DTYPE_SIZE = {"float32": 4, "float64": 8}
_MAX_METADATA_BYTES = 16 * 1024 * 1024


def _validate_stream_shape(
    *,
    scenario: str,
    timestep: float,
    qpos_width: int,
    qvel_width: int,
    control_width: int,
) -> None:
    if not isinstance(scenario, str) or not scenario:
        raise ValueError("scenario must not be empty")
    if not isfinite(timestep) or timestep <= 0:
        raise ValueError("timestep must be finite and positive")
    if qpos_width < 1 or qvel_width < 1 or control_width < 0:
        raise ValueError(
            "trajectory widths require positive qpos/qvel and nonnegative control"
        )
    if max(qpos_width, qvel_width, control_width) > 0xFFFFFFFF:
        raise ValueError("trajectory width exceeds the format limit")


def _native_bytes(payload: bytes, dtype: str) -> bytes:
    if sys.byteorder == "little":
        return payload
    values = array(_DTYPE_FORMAT[dtype])
    values.frombytes(payload)
    values.byteswap()
    return values.tobytes()


def _typed_view(payload: bytes, dtype: str) -> memoryview:
    native = _native_bytes(payload, dtype)
    return memoryview(native).cast(_DTYPE_FORMAT[dtype])


def _finite(values: memoryview) -> bool:
    return all(isfinite(value) for value in values)


@dataclass(frozen=True)
class TrajectoryFrame:
    qpos: memoryview
    qvel: memoryview
    controls: memoryview


class TrajectoryWriter:
    """Writes fixed-width trajectory frames without retaining prior samples."""

    def __init__(
        self,
        path: str | Path,
        *,
        scenario: str,
        timestep: float,
        qpos_width: int,
        qvel_width: int,
        control_width: int,
        dtype: str = "float32",
        metadata: dict[str, Any] | None = None,
    ):
        _validate_stream_shape(
            scenario=scenario,
            timestep=timestep,
            qpos_width=qpos_width,
            qvel_width=qvel_width,
            control_width=control_width,
        )
        if dtype not in _DTYPE_TO_CODE:
            raise ValueError("trajectory dtype must be float32 or float64")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.scenario = scenario
        self.timestep = timestep
        self.qpos_width = qpos_width
        self.qvel_width = qvel_width
        self.control_width = control_width
        self.dtype = dtype
        self.metadata = dict(metadata or {})
        self.metadata["scenario"] = scenario
        self._metadata_bytes = json.dumps(
            self.metadata,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        if len(self._metadata_bytes) > _MAX_METADATA_BYTES:
            raise ValueError("trajectory metadata is too large")
        temporary = tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=self.path.parent,
            prefix=f".{self.path.name}.",
            suffix=".partial",
            delete=False,
        )
        self._file = temporary
        self._temporary_path = Path(temporary.name)
        self._frames = 0
        self._closed = False
        self._record_values = qpos_width + qvel_width + control_width
        self._record_bytes = self._record_values * _DTYPE_SIZE[dtype]
        try:
            self._write_header(frame_count=0)
            self._file.write(self._metadata_bytes)
        except BaseException:
            self.abort()
            raise

    def _write_header(self, *, frame_count: int) -> None:
        self._file.seek(0)
        self._file.write(
            _STREAM_HEADER.pack(
                _STREAM_MAGIC,
                _STREAM_VERSION,
                _DTYPE_TO_CODE[self.dtype],
                0,
                len(self._metadata_bytes),
                frame_count,
                self.qpos_width,
                self.qvel_width,
                self.control_width,
                self.timestep,
            )
        )

    @property
    def frame_count(self) -> int:
        return self._frames

    def append(self, frame: bytes | bytearray | memoryview) -> None:
        if self._closed:
            raise ValueError("trajectory writer is closed")
        view = memoryview(frame)
        if not view.c_contiguous:
            raise ValueError("trajectory frame must be contiguous")
        raw = view.cast("B")
        if raw.nbytes != self._record_bytes:
            raise ValueError(
                f"trajectory frame has {raw.nbytes} bytes, "
                f"expected {self._record_bytes}"
            )
        if not _finite(raw.cast(_DTYPE_FORMAT[self.dtype])):
            raise ValueError("trajectory contains a non-finite value")
        if sys.byteorder == "little":
            self._file.write(raw)
        else:
            values = array(_DTYPE_FORMAT[self.dtype])
            values.frombytes(raw)
            values.byteswap()
            self._file.write(values.tobytes())
        self._frames += 1

    def close(self) -> None:
        if self._closed:
            return
        if self._frames < 1:
            self.abort()
            raise ValueError("trajectory must contain at least one frame")
        try:
            self._write_header(frame_count=self._frames)
            self._file.flush()
            os.fsync(self._file.fileno())
            self._file.close()
            os.chmod(self._temporary_path, 0o644)
            os.replace(self._temporary_path, self.path)
            self._closed = True
        except BaseException:
            self.abort()
            raise

    def abort(self) -> None:
        if self._closed:
            return
        self._file.close()
        self._temporary_path.unlink(missing_ok=True)
        self._closed = True

    def __enter__(self) -> "TrajectoryWriter":
        return self

    def __exit__(self, error_type, error, traceback) -> None:
        if error_type is None:
            self.close()
        else:
            self.abort()


class TrajectoryReader:
    """Reads fixed-width trajectory frames with bounded host memory."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._file = self.path.open("rb")
        try:
            header = self._file.read(_STREAM_HEADER.size)
            if len(header) != _STREAM_HEADER.size:
                raise ValueError("trajectory header is truncated")
            (
                magic,
                version,
                dtype_code,
                flags,
                metadata_size,
                self.frame_count,
                self.qpos_width,
                self.qvel_width,
                self.control_width,
                self.timestep,
            ) = _STREAM_HEADER.unpack(header)
            if magic != _STREAM_MAGIC:
                raise ValueError("invalid trajectory magic")
            if version != _STREAM_VERSION:
                raise ValueError(f"unsupported trajectory version {version}")
            if flags:
                raise ValueError("unsupported trajectory flags")
            if dtype_code not in _CODE_TO_DTYPE:
                raise ValueError("unsupported trajectory dtype")
            self.dtype = _CODE_TO_DTYPE[dtype_code]
            if metadata_size > _MAX_METADATA_BYTES:
                raise ValueError("trajectory metadata is too large")
            metadata_bytes = self._file.read(metadata_size)
            if len(metadata_bytes) != metadata_size:
                raise ValueError("trajectory metadata is truncated")
            try:
                metadata = json.loads(metadata_bytes)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError("trajectory metadata is invalid") from error
            if not isinstance(metadata, dict):
                raise ValueError("trajectory metadata must be an object")
            self.metadata = metadata
            self.scenario = metadata.get("scenario")
            _validate_stream_shape(
                scenario=self.scenario,
                timestep=self.timestep,
                qpos_width=self.qpos_width,
                qvel_width=self.qvel_width,
                control_width=self.control_width,
            )
            if self.frame_count < 1:
                raise ValueError("trajectory must contain at least one frame")
            self._data_offset = _STREAM_HEADER.size + metadata_size
            self._record_values = (
                self.qpos_width + self.qvel_width + self.control_width
            )
            self._record_bytes = (
                self._record_values * _DTYPE_SIZE[self.dtype]
            )
            expected_size = (
                self._data_offset + self.frame_count * self._record_bytes
            )
            actual_size = os.fstat(self._file.fileno()).st_size
            if actual_size != expected_size:
                raise ValueError(
                    f"trajectory length is {actual_size} bytes, "
                    f"expected {expected_size}"
                )
        except BaseException:
            self._file.close()
            raise

    def _read(self, index: int, *, offset: int, width: int) -> memoryview:
        if not isinstance(index, int) or not 0 <= index < self.frame_count:
            raise IndexError("trajectory frame index is out of range")
        start = (
            self._data_offset
            + index * self._record_bytes
            + offset * _DTYPE_SIZE[self.dtype]
        )
        size = width * _DTYPE_SIZE[self.dtype]
        self._file.seek(start)
        payload = self._file.read(size)
        if len(payload) != size:
            raise ValueError("trajectory frame is truncated")
        values = _typed_view(payload, self.dtype)
        if not _finite(values):
            raise ValueError("trajectory contains a non-finite value")
        return values

    def read_qpos(self, index: int) -> memoryview:
        return self._read(index, offset=0, width=self.qpos_width)

    def read_qvel(self, index: int) -> memoryview:
        return self._read(
            index,
            offset=self.qpos_width,
            width=self.qvel_width,
        )

    def read_controls(self, index: int) -> memoryview:
        return self._read(
            index,
            offset=self.qpos_width + self.qvel_width,
            width=self.control_width,
        )

    def read_frame(self, index: int) -> TrajectoryFrame:
        return TrajectoryFrame(
            qpos=self.read_qpos(index),
            qvel=self.read_qvel(index),
            controls=self.read_controls(index),
        )

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> "TrajectoryReader":
        return self

    def __exit__(self, error_type, error, traceback) -> None:
        self.close()


def export_trajectory_json(
    source: str | Path, destination: str | Path
) -> None:
    """Streams a ``.tstraj`` file to the portable JSON trajectory schema."""

    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".partial",
        delete=False,
    )
    temporary_path = Path(temporary.name)

    def write_rows(
        reader: TrajectoryReader,
        read: Callable[[int], memoryview],
    ) -> None:
        temporary.write("[")
        for index in range(reader.frame_count):
            if index:
                temporary.write(",")
            json.dump(
                list(read(index)),
                temporary,
                allow_nan=False,
                separators=(",", ":"),
            )
        temporary.write("]")

    try:
        with TrajectoryReader(source) as trajectory:
            temporary.write('{"controls":')
            write_rows(trajectory, trajectory.read_controls)
            temporary.write(',"metadata":')
            metadata = {
                key: value
                for key, value in trajectory.metadata.items()
                if key != "scenario"
            }
            json.dump(
                metadata,
                temporary,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            temporary.write(',"qpos":')
            write_rows(trajectory, trajectory.read_qpos)
            temporary.write(',"qvel":')
            write_rows(trajectory, trajectory.read_qvel)
            temporary.write(',"scenario":')
            json.dump(trajectory.scenario, temporary)
            temporary.write(',"timestep":')
            json.dump(trajectory.timestep, temporary, allow_nan=False)
            temporary.write("}\n")
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary.close()
        os.chmod(temporary_path, 0o644)
        os.replace(temporary_path, target)
    except BaseException:
        temporary.close()
        temporary_path.unlink(missing_ok=True)
        raise
