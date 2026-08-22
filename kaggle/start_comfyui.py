from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COMFY = ROOT / "ComfyUI"


def main():
    command = [
        sys.executable,
        str(COMFY / "main.py"),
        "--listen",
        "0.0.0.0",
        "--port",
        "8188",
    ]

    print("Starting MiniMax H3 ComfyUI:")
    print(" ".join(command))

    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
