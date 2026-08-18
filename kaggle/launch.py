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

    subprocess.run(
        [
            sys.executable,
            str(TUNNEL),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
