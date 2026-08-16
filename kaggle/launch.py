from pathlib import Path
import socket
import subprocess
import sys
import time


PROJECT = Path(
    "/kaggle/working/LTX-13B-AI-Video-Model"
)

BOOTSTRAP = (
    PROJECT / "kaggle" / "bootstrap.py"
)

TUNNEL = (
    PROJECT / "kaggle"
    / "start_comfyui_tunnel.py"
)

COMFY = PROJECT / "ComfyUI"

COMFY_MAIN = COMFY / "main.py"

PORT = 8188


# ============================================================
# HELPERS
# ============================================================

def run(command):
    print(
        "\n$",
        " ".join(map(str, command)),
    )

    subprocess.run(
        command,
        check=True,
    )


def port_is_open(
    host="127.0.0.1",
    port=8188,
):
    try:

        with socket.create_connection(
            (host, port),
            timeout=1,
        ):
            return True

    except OSError:

        return False


# ============================================================
# GPU CHECK
# ============================================================

def check_gpu():

    import torch

    print("=" * 60)
    print("GPU ENVIRONMENT")
    print("=" * 60)

    print("Torch:", torch.__version__)
    print("CUDA:", torch.version.cuda)
    print(
        "CUDA available:",
        torch.cuda.is_available(),
    )

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA is unavailable."
        )

    count = torch.cuda.device_count()

    print("GPU count:", count)

    if count < 2:

        raise RuntimeError(
            "This project requires two T4 GPUs."
        )

    for i in range(count):

        print(
            f"GPU {i}:",
            torch.cuda.get_device_name(i),
        )


# ============================================================
# REPOSITORY
# ============================================================

def ensure_repository():

    if PROJECT.exists():

        print(
            "Repository already exists:"
        )

        print(PROJECT)

        return

    print(
        "Repository missing."
    )

    PROJECT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    run(
        [
            "git",
            "clone",
            "https://github.com/"
            "shihoos/LTX-13B-AI-Video-Model",
            str(PROJECT),
        ]
    )


# ============================================================
# COMFYUI
# ============================================================

def ensure_comfyui():

    if COMFY_MAIN.exists():

        print(
            "✅ ComfyUI installation found."
        )

        return

    print(
        "⚠️ ComfyUI installation missing."
    )

    print(
        "Running bootstrap..."
    )

    run(
        [
            sys.executable,
            str(BOOTSTRAP),
        ]
    )

    if not COMFY_MAIN.exists():

        raise RuntimeError(
            "Bootstrap completed but "
            "ComfyUI/main.py was not found."
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print(
        "LTX-13B AI VIDEO SYSTEM"
    )
    print(
        "ONE-CELL STARTUP"
    )
    print("=" * 60)

    # 1. GPU
    check_gpu()

    # 2. Repository
    ensure_repository()

    # 3. ComfyUI
    ensure_comfyui()

    # 4. If ComfyUI is already running,
    #    do not launch another copy.
    if port_is_open():

        print(
            "✅ ComfyUI is already running "
            "on port 8188."
        )

        print(
            "Open your existing tunnel URL."
        )

        return

    # 5. Start tunnel script
    print()
    print(
        "Starting ComfyUI + Cloudflare tunnel..."
    )

    run(
        [
            sys.executable,
            "-u",
            str(TUNNEL),
        ]
    )


if __name__ == "__main__":
    main()
