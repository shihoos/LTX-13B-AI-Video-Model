from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path


# ============================================================================
# PROJECT
# ============================================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)

COMFYUI_DIR = (
    PROJECT_ROOT
    / "ComfyUI"
)


# ============================================================================
# COMFYUI WORKER
# ============================================================================

HOST = os.getenv(
    "COMFYUI_HOST",
    "127.0.0.1",
)

GPU_ID_TEXT = os.getenv(
    "COMFYUI_GPU_ID",
    "0",
)

try:

    GPU_ID = int(
        GPU_ID_TEXT
    )

except ValueError as error:

    raise ValueError(
        "COMFYUI_GPU_ID must be an integer:\n"
        f"Received: {GPU_ID_TEXT}"
    ) from error

if GPU_ID < 0:

    raise ValueError(
        "COMFYUI_GPU_ID must be >= 0."
    )


# ============================================================================
# CANONICAL PORT
# ============================================================================
#
# GPU 0 -> 8188
# GPU 1 -> 8189
# GPU 2 -> 8190
#
# launch.py starts one worker per GPU.
# This file is responsible for one worker only.
# ============================================================================

CANONICAL_COMFYUI_PORT = 8188

PORT = (
    CANONICAL_COMFYUI_PORT
    + GPU_ID
)


# ============================================================================
# RUNTIME FILES
# ============================================================================

LOG_DIR = (
    PROJECT_ROOT
    / ".runtime_kaggle"
)

LOG_FILE = (
    LOG_DIR
    / f"comfyui_gpu_{GPU_ID}_port_{PORT}.log"
)

PID_FILE = (
    LOG_DIR
    / f"comfyui_gpu_{GPU_ID}_port_{PORT}.pid"
)


# ============================================================================
# PORT CHECK
# ============================================================================

def port_ready() -> bool:

    try:

        with socket.create_connection(
            (
                HOST,
                PORT,
            ),
            timeout=2,
        ):

            return True

    except OSError:

        return False


# ============================================================================
# WAIT FOR COMFYUI
# ============================================================================

def wait_for_comfyui(
    process: subprocess.Popen,
    timeout: int = 180,
) -> None:

    deadline = (
        time.monotonic()
        + timeout
    )

    while (
        time.monotonic()
        < deadline
    ):

        if process.poll() is not None:

            raise RuntimeError(
                "ComfyUI exited before becoming ready.\n"
                f"GPU: {GPU_ID}\n"
                f"Host: {HOST}\n"
                f"Port: {PORT}\n"
                f"Exit code: {process.returncode}\n"
                f"Log: {LOG_FILE}"
            )

        if port_ready():

            print(
                f"✅ ComfyUI is ready on "
                f"{HOST}:{PORT}"
            )

            return

        time.sleep(
            2
        )

    raise TimeoutError(
        "ComfyUI did not become ready within "
        f"{timeout} seconds.\n"
        f"GPU: {GPU_ID}\n"
        f"Host: {HOST}\n"
        f"Port: {PORT}\n"
        f"Log: {LOG_FILE}"
    )


# ============================================================================
# MAIN
# ============================================================================

def main():

    main_py = (
        COMFYUI_DIR
        / "main.py"
    )

    if not main_py.is_file():

        raise FileNotFoundError(
            "ComfyUI main.py was not found:\n"
            f"{main_py}"
        )

    # ------------------------------------------------------------------------
    # Do not start another worker if this worker's endpoint is already alive.
    # ------------------------------------------------------------------------

    if port_ready():

        print(
            f"✅ ComfyUI is already running "
            f"on {HOST}:{PORT}"
        )

        print(
            f"GPU: {GPU_ID}"
        )

        return

    # ------------------------------------------------------------------------
    # ComfyUI command
    # ------------------------------------------------------------------------

    command = [
        sys.executable,
        str(main_py),
        "--listen",
        HOST,
        "--port",
        str(PORT),
        "--cuda-device",
        str(GPU_ID),
    ]

    print(
        "=" * 80
    )

    print(
        "LTX-13B LOCAL COMFYUI BACKEND"
    )

    print(
        "=" * 80
    )

    print(
        f"GPU: {GPU_ID}"
    )

    print(
        f"Host: {HOST}"
    )

    print(
        f"Port: {PORT}"
    )

    print(
        "Mode: backend/API only"
    )

    print(
        "Public tunnel: disabled"
    )

    print(
        "=" * 80
    )

    # ------------------------------------------------------------------------
    # Runtime directory
    # ------------------------------------------------------------------------

    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------------------
    # Log
    # ------------------------------------------------------------------------

    log_handle = LOG_FILE.open(
        "a",
        encoding="utf-8",
    )

    # ------------------------------------------------------------------------
    # Start ComfyUI
    # ------------------------------------------------------------------------

    process = subprocess.Popen(
        command,
        cwd=COMFYUI_DIR,
        stdin=subprocess.DEVNULL,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )

    log_handle.close()

    # ------------------------------------------------------------------------
    # PID
    # ------------------------------------------------------------------------

    PID_FILE.write_text(
        str(process.pid),
        encoding="utf-8",
    )

    print(
        f"Background ComfyUI PID: "
        f"{process.pid}"
    )

    print(
        f"ComfyUI log: {LOG_FILE}"
    )

    print(
        f"ComfyUI PID file: {PID_FILE}"
    )

    # ------------------------------------------------------------------------
    # Wait for readiness
    # ------------------------------------------------------------------------

    try:

        wait_for_comfyui(
            process
        )

    except Exception:

        if process.poll() is None:

            process.terminate()

        PID_FILE.unlink(
            missing_ok=True
        )

        raise

    # ------------------------------------------------------------------------
    # Ready
    # ------------------------------------------------------------------------

    print()

    print(
        "✅ LOCAL COMFYUI BACKEND READY"
    )

    print(
        f"GPU: {GPU_ID}"
    )

    print(
        f"API: http://{HOST}:{PORT}"
    )

    print(
        "The startup process can now exit."
    )

    print(
        "ComfyUI continues running in the background."
    )


if __name__ == "__main__":
    main()
