from __future__ import annotations

import subprocess
import sys
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

START_COMFYUI = (
    PROJECT
    / "kaggle"
    / "start_comfyui.py"
)


def run_checked(
    script: Path,
) -> None:

    subprocess.run(
        [
            sys.executable,
            str(script),
        ],
        cwd=PROJECT,
        check=True,
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

    print()
    print(
        "STEP 1 — REPOSITORY / GPU PREFLIGHT"
    )

    run_checked(
        PREFLIGHT
    )

    print()
    print(
        "STEP 2 — EXACT RUNTIME BOOTSTRAP"
    )

    run_checked(
        BOOTSTRAP
    )

    print()
    print(
        "STEP 3 — START LOCAL COMFYUI BACKEND"
    )

    run_checked(
        START_COMFYUI
    )

    print()
    print(
        "=" * 80
    )

    print(
        "✅ LTX-13B BACKEND READY"
    )

    print(
        "=" * 80
    )

    print(
        "ComfyUI API: http://127.0.0.1:8188"
    )

    print(
        "Cloudflare: disabled"
    )

    print(
        "Browser: not required"
    )

    print(
        "ComfyUI continues running in the background."
    )

    print(
        "You can now run the generation cell."
    )


if __name__ == "__main__":
    main()
