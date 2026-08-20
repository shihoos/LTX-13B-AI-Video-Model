from __future__ import annotations

import socket
import subprocess
import sys
import time
from pathlib import Path


PROJECT = Path(
    "/kaggle/working/LTX-13B-AI-Video-Model"
)

PREFLIGHT = (
    PROJECT
    / "kaggle"
    / "preflight_modern.py"
)

BOOTSTRAP = (
    PROJECT
    / "kaggle"
    / "bootstrap.py"
)

TUNNEL = (
    PROJECT
    / "kaggle"
    / "start_comfyui_tunnel.py"
)


def main():

    print(
        "=" * 80
    )

    print(
        "LTX-13B MODERN ONE-CELL STARTUP"
    )

    print(
        "=" * 80
    )

    subprocess.run(
        [
            sys.executable,
            str(PREFLIGHT),
        ],
        check=True,
    )

    subprocess.run(
        [
            sys.executable,
            str(BOOTSTRAP),
        ],
        check=True,
    )

    # Start ComfyUI + Cloudflare tunnel in the background.
    process = subprocess.Popen(
        [
            sys.executable,
            str(TUNNEL),
        ],
        cwd=PROJECT,
        start_new_session=True,
    )

    # Wait until ComfyUI is listening on port 8188.
    deadline = (
        time.time()
        + 180
    )

    while time.time() < deadline:

        if process.poll() is not None:

            raise RuntimeError(
                "ComfyUI/tunnel process exited "
                f"with code {process.returncode}."
            )

        try:

            with socket.create_connection(
                (
                    "127.0.0.1",
                    8188,
                ),
                timeout=2,
            ):

                print()
                print(
                    "✅ ComfyUI is running on port 8188."
                )
                print(
                    "✅ Background runtime started."
                )
                print(
                    "✅ This Kaggle cell can now finish."
                )

                return

        except OSError:

            time.sleep(
                2
            )

    raise TimeoutError(
        "ComfyUI did not become ready "
        "within 180 seconds."
    )


if __name__ == "__main__":
    main()
