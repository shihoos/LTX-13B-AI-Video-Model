from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


PROJECT = Path(
    "/kaggle/working/LTX-13B-AI-Video-Model"
)

BOOTSTRAP = (
    PROJECT
    / "kaggle"
    / "bootstrap.py"
)

PREFLIGHT = (
    PROJECT
    / "kaggle"
    / "preflight_modern.py"
)

START_COMFYUI = (
    PROJECT
    / "kaggle"
    / "start_comfyui.py"
)

COMFYUI_HOST = (
    "127.0.0.1"
)

PRIMARY_COMFYUI_PORT = (
    8188
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


def get_gpu_count() -> int:

    try:

        import torch

    except ImportError as error:

        raise RuntimeError(
            "PyTorch is not available after "
            "runtime bootstrap."
        ) from error

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA is not available. "
            "Kaggle GPU must be enabled."
        )

    gpu_count = (
        torch.cuda.device_count()
    )

    if gpu_count < 1:

        raise RuntimeError(
            "No CUDA GPUs were detected."
        )

    return gpu_count


def start_comfyui_worker(
    gpu_id: int,
) -> None:

    port = (
        PRIMARY_COMFYUI_PORT
        + gpu_id
    )

    environment = os.environ.copy()

    environment[
        "COMFYUI_HOST"
    ] = COMFYUI_HOST


    environment[
        "COMFYUI_GPU_ID"
    ] = str(gpu_id)

    print()
    print(
        "=" * 80
    )

    print(
        f"STARTING COMFYUI WORKER "
        f"GPU {gpu_id}"
    )

    print(
        "=" * 80
    )

    print(
        f"GPU:  {gpu_id}"
    )

    print(
        f"Host: {COMFYUI_HOST}"
    )

    print(
        f"Port: {port}"
    )

    subprocess.run(
        [
            sys.executable,
            str(START_COMFYUI),
        ],
        cwd=PROJECT,
        env=environment,
        check=True,
    )


def main():

    print(
        "=" * 80
    )

    print(
        "LTX-13B MODERN MULTI-GPU STARTUP"
    )

    print(
        "=" * 80
    )

    print()

    print(
        "STEP 1 — REPOSITORY / GPU PREFLIGHT"
    )

    run_checked(
        BOOTSTRAP
    )

    print()

    print(
        "STEP 2 — EXACT RUNTIME BOOTSTRAP"
    )

    run_checked(
        PREFLIGHT
    )

    print()

    print(
        "STEP 3 — DETECT CUDA GPUs"
    )

    gpu_count = (
        get_gpu_count()
    )

    print(
        f"Detected {gpu_count} CUDA GPU(s)."
    )

    print()

    print(
        "STEP 4 — START COMFYUI WORKERS"
    )

    for gpu_id in range(
        gpu_count
    ):

        start_comfyui_worker(
            gpu_id
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

    print()

    for gpu_id in range(
        gpu_count
    ):

        port = (
            PRIMARY_COMFYUI_PORT
            + gpu_id
        )

        print(
            f"GPU {gpu_id}: "
            f"http://{COMFYUI_HOST}:{port}"
        )

    print()


    print(
        "All ComfyUI workers continue "
        "running in the background."
    )

    print(
        "You can now run the generation cell."
    )


if __name__ == "__main__":
    main()
