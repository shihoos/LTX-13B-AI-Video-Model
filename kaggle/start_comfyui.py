"""
Compatibility entry point.

The official startup path is:

    kaggle/launch.py

This wrapper exists so older commands that call
kaggle/start_comfyui.py do not bypass the modern
preflight/bootstrap path.
"""

from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)

LAUNCH = (
    PROJECT_ROOT
    / "kaggle"
    / "launch.py"
)


def main():

    if not LAUNCH.exists():

        raise FileNotFoundError(
            f"Modern launcher not found:\n"
            f"{LAUNCH}"
        )

    subprocess.run(
        [
            sys.executable,
            str(LAUNCH),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
