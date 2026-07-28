from io import BytesIO
import math
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from tinysim.render import (
    Camera2D,
    Camera3D,
    OrbitCamera,
    encode_mp4,
    render_articulated,
    render_batched_pendulums,
    render_bouncing_ball,
    render_cartpole,
    render_free_body,
    render_pendulum,
    write_ppm,
)


class TestRender(unittest.TestCase):
    class FakeProcess:
        def __init__(self, *, broken_pipe=False, stderr=b"", status=0):
            self.stdin = self
            self.stderr = BytesIO(stderr)
            self.broken_pipe = broken_pipe
            self.status = status

        def write(self, payload):
            if self.broken_pipe:
                raise BrokenPipeError
            return len(payload)

        def close(self):
            return None

        def wait(self):
            return self.status

        def kill(self):
            return None

    def test_deterministic_frame(self):
        first = render_pendulum(0.25, width=64, height=48)
        second = render_pendulum(0.25, width=64, height=48)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 64 * 48 * 3)
        self.assertNotEqual(len(set(first)), 1)

    def test_ppm_output(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "frame.ppm"
            write_ppm(
                path,
                render_pendulum(0.0, width=32, height=24),
                width=32,
                height=24,
            )
            payload = path.read_bytes()
            self.assertTrue(payload.startswith(b"P6\n32 24\n255\n"))

    def test_scenario_renderers_are_deterministic_and_state_sensitive(self):
        renderings = (
            (render_cartpole, (0.1, 0.2), (0.1, -0.2)),
            (render_bouncing_ball, ([0.0, 0.0, 0.8],), ([0.0, 0.0, 0.4],)),
            (render_articulated, ([0.1, -0.2],), ([0.2, -0.2],)),
            (render_batched_pendulums, ([0.1, 0.1, 0.2, -0.2],), ([0.1, 0.1, 1.0, -0.2],)),
        )
        for renderer, first_args, second_args in renderings:
            with self.subTest(renderer=renderer.__name__):
                first = renderer(*first_args, width=96, height=64)
                self.assertEqual(first, renderer(*first_args, width=96, height=64))
                self.assertNotEqual(first, renderer(*second_args, width=96, height=64))
                self.assertEqual(len(first), 96 * 64 * 3)

    def test_renderer_shape_contracts(self):
        with self.assertRaisesRegex(ValueError, "x, y, and z"):
            render_bouncing_ball([0.0, 1.0])
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            render_articulated([])
        with self.assertRaisesRegex(ValueError, "too narrow"):
            render_batched_pendulums([0.0] * 5, width=64, height=48)
        with self.assertRaisesRegex(ValueError, "xyz"):
            render_free_body([1.0, 0.0], width=64, height=48)

    def test_free_body_camera_pan_and_zoom_are_deterministic(self):
        qpos = [0.2, 0.0, -0.1, 0.98, 0.1, 0.05, 0.0]
        default = render_free_body(qpos, width=96, height=64)
        camera = Camera2D(center_x=0.2, center_z=-0.1, zoom=1.4)
        controlled = render_free_body(
            qpos, width=96, height=64, camera=camera
        )
        self.assertEqual(
            controlled,
            render_free_body(qpos, width=96, height=64, camera=camera),
        )
        self.assertNotEqual(default, controlled)
        with self.assertRaisesRegex(ValueError, "zoom positive"):
            Camera2D(zoom=0.0)

    def test_orbit_camera_advances_azimuth_over_the_recording(self):
        orbit = OrbitCamera(
            center_z=0.6,
            elevation=0.25,
            start_azimuth=0.4,
            turns=0.5,
            zoom=1.2,
        )
        first = orbit(0, 5)
        middle = orbit(2, 5)
        last = orbit(4, 5)
        self.assertIsInstance(first, Camera3D)
        self.assertAlmostEqual(first.azimuth, 0.4)
        self.assertAlmostEqual(middle.azimuth, 0.4 + 0.5 * math.pi)
        self.assertAlmostEqual(last.azimuth, 0.4 + math.pi)
        self.assertEqual(first.center_z, 0.6)
        with self.assertRaisesRegex(ValueError, "index"):
            orbit(5, 5)
        with self.assertRaisesRegex(ValueError, "zoom positive"):
            Camera3D(zoom=0.0)

    def test_mp4_requires_even_dimensions(self):
        with self.assertRaisesRegex(ValueError, "positive even"):
            encode_mp4([], "ignored.mp4", width=63, height=48)

    def test_system_encoder_ignores_inherited_loader_overrides(self):
        process = self.FakeProcess()
        probe = type("Probe", (), {"returncode": 0, "stderr": b""})()
        with (
            patch.dict(
                os.environ,
                {
                    "LD_LIBRARY_PATH": "/incompatible",
                    "LD_PRELOAD": "/incompatible.so",
                },
            ),
            patch(
                "tinysim.render.shutil.which",
                side_effect=lambda name: (
                    None
                    if name == "ffmpeg"
                    else "/usr/bin/gst-launch-1.0"
                ),
            ),
            patch("tinysim.render.subprocess.run", return_value=probe),
            patch(
                "tinysim.render.subprocess.Popen",
                return_value=process,
            ) as popen,
        ):
            encode_mp4(
                [bytes(64 * 48 * 3)],
                "ignored.mp4",
                width=64,
                height=48,
            )
        environment = popen.call_args.kwargs["env"]
        self.assertNotIn("LD_LIBRARY_PATH", environment)
        self.assertNotIn("LD_PRELOAD", environment)

    def test_encoder_broken_pipe_reports_subprocess_error(self):
        process = self.FakeProcess(
            broken_pipe=True,
            stderr=b"symbol lookup error",
            status=127,
        )
        probe = type("Probe", (), {"returncode": 0, "stderr": b""})()
        with (
            patch(
                "tinysim.render.shutil.which",
                side_effect=lambda name: (
                    None
                    if name == "ffmpeg"
                    else "/usr/bin/gst-launch-1.0"
                ),
            ),
            patch("tinysim.render.subprocess.run", return_value=probe),
            patch(
                "tinysim.render.subprocess.Popen",
                return_value=process,
            ),
        ):
            with self.assertRaisesRegex(
                RuntimeError, "symbol lookup error"
            ):
                encode_mp4(
                    [bytes(64 * 48 * 3)],
                    "ignored.mp4",
                    width=64,
                    height=48,
                )


if __name__ == "__main__":
    unittest.main()
