"""
Start ComfyUI for the LTX-13B AI Video Model project.

This script is intentionally lightweight.
Environment installation and custom-node setup are handled by bootstrap.py.
"""

from pathlib import Path
import os
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMFYUI_DIR = Path(
    os.getenv("COMFYUI_DIR", PROJECT_ROOT / "ComfyUI")
)

HOST = os.getenv("COMFYUI_HOST", "0.0.0.0")
PORT = os.getenv("COMFYUI_PORT", "8188")


def main():
    main_py = COMFYUI_DIR / "main.py"

    if not main_py.exists():
        print(f"ERROR: ComfyUI was not found at: {COMFYUI_DIR}")
        print("Run bootstrap.py first.")
        sys.exit(1)

    command = [
        sys.executable,
        "main.py",
        "--listen",
        HOST,
        "--port",
        PORT,
    ]

    print("=" * 60)
    print("Starting ComfyUI")
    print(f"Directory : {COMFYUI_DIR}")
    print(f"Host      : {HOST}")
    print(f"Port      : {PORT}")
    print("=" * 60)

    subprocess.run(
        command,
        cwd=COMFYUI_DIR,
        check=True,
    )


if __name__ == "__main__":
    main()
