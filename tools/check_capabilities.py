"""Report optional TinySim release capabilities without changing the system."""

import json
import shutil
import subprocess

from tinygrad import Device, Tensor


def main() -> None:
    accelerator_candidates = ("NV", "CUDA", "AMD", "METAL")
    accelerators: dict[str, dict[str, str | bool]] = {}
    for name in accelerator_candidates:
        try:
            tensor = Tensor([1.0], device=name).realize()
            accelerators[name] = {
                "available": True,
                "device": str(tensor.device),
            }
        except Exception as error:
            accelerators[name] = {
                "available": False,
                "reason": f"{type(error).__name__}: {error}",
            }
    print(
        json.dumps(
            {
                "selected_backend": Device.DEFAULT,
                "accelerators": accelerators,
                "ffmpeg": {
                    "available": shutil.which("ffmpeg") is not None,
                    "path": shutil.which("ffmpeg") or "",
                },
                "gstreamer_openh264": {
                    "available": all(
                        shutil.which(executable) is not None
                        for executable in ("gst-launch-1.0", "gst-inspect-1.0")
                    )
                    and all(
                        subprocess.run(
                            ["gst-inspect-1.0", plugin],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            check=False,
                        ).returncode
                        == 0
                        for plugin in ("videoparse", "openh264enc", "h264parse", "mp4mux")
                    ),
                    "path": shutil.which("gst-launch-1.0") or "",
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
