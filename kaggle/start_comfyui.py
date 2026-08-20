from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)

COMFYUI_DIR = (
    PROJECT_ROOT
    / "ComfyUI"
)

HOST = os.getenv(
    "COMFYUI_HOST",
    "127.0.0.1",
)

PORT = int(
    os.getenv(
        "COMFYUI_PORT",
        "8188",
    )
)


def port_ready() -> bool:

    try:

        with socket.create_connection(
            (
                "127.0.0.1",
                PORT,
            ),
            timeout=2,
        ):

            return True

    except OSError:

        return False


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
                f"Exit code: {process.returncode}"
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
        f"{timeout} seconds."
    )


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

    if port_ready():

        print(
            f"✅ ComfyUI is already running "
            f"on {HOST}:{PORT}"
        )

        return

    command = [
        sys.executable,
        str(main_py),
        "--listen",
        HOST,
        "--port",
        str(PORT),
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

    log_dir = (
        PROJECT_ROOT
        / ".runtime_kaggle"
    )

    log_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_file = (
        log_dir
        / "comfyui.log"
    )

    log_handle = log_file.open(
        "a",
        encoding="utf-8",
    )

    process = subprocess.Popen(
        command,
        cwd=COMFYUI_DIR,
        stdin=subprocess.DEVNULL,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )

    pid_file = (
        log_dir
        / "comfyui.pid"
    )

    pid_file.write_text(
        str(process.pid),
        encoding="utf-8",
    )

    print(
        f"Background ComfyUI PID: "
        f"{process.pid}"
    )

    print(
        f"ComfyUI log: {log_file}"
    )

    try:

        wait_for_comfyui(
            process
        )

    except Exception:

        if process.poll() is None:

            process.terminate()

        raise

    print()
    print(
        "✅ LOCAL COMFYUI BACKEND READY"
    )
    print(
        f"API: http://127.0.0.1:{PORT}"
    )
    print(
        "The startup process can now exit."
    )
    print(
        "ComfyUI continues running in the background."
    )


if __name__ == "__main__":
    main()
