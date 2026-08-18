from __future__ import annotations

import os
import queue
import re
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path


PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)

COMFYUI_DIR = Path(
    os.getenv(
        "COMFYUI_DIR",
        str(
            PROJECT_ROOT
            / "ComfyUI"
        ),
    )
)

HOST = os.getenv(
    "COMFYUI_HOST",
    "0.0.0.0",
)

PORT = int(
    os.getenv(
        "COMFYUI_PORT",
        "8188",
    )
)

CLOUDFLARED_PATH = (
    PROJECT_ROOT
    / "cloudflared"
)

CLOUDFLARED_URL = (
    "https://github.com/cloudflare/"
    "cloudflared/releases/latest/"
    "download/cloudflared-linux-amd64"
)

URL_PATTERN = re.compile(
    r"https://[a-zA-Z0-9-]+\.trycloudflare\.com"
)


def stream(
    process,
    prefix,
    output_queue=None,
):

    if process.stdout is None:
        return

    for line in iter(
        process.stdout.readline,
        "",
    ):

        if not line:
            break

        line = line.rstrip()

        if output_queue is not None:

            output_queue.put(
                line
            )

        print(
            f"{prefix} {line}",
            flush=True,
        )


def ensure_cloudflared():

    if not CLOUDFLARED_PATH.exists():

        print(
            "Downloading cloudflared..."
        )

        urllib.request.urlretrieve(
            CLOUDFLARED_URL,
            CLOUDFLARED_PATH,
        )

    os.chmod(
        CLOUDFLARED_PATH,
        0o755,
    )


def start_comfyui():

    if not (
        COMFYUI_DIR
        / "main.py"
    ).exists():

        raise FileNotFoundError(
            f"ComfyUI not found:\n"
            f"{COMFYUI_DIR}"
        )

    # IMPORTANT:
    # Do not pass --front-end-version @latest.
    # The exact frontend package has already been pinned
    # by the modern v0.33.1 requirements.
    command = [
        sys.executable,
        "main.py",
        "--listen",
        HOST,
        "--port",
        str(PORT),
    ]

    process = subprocess.Popen(
        command,
        cwd=COMFYUI_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    threading.Thread(
        target=stream,
        args=(
            process,
            "[ComfyUI]",
        ),
        daemon=True,
    ).start()

    return process


def port_open():

    try:

        with socket.create_connection(
            (
                "127.0.0.1",
                PORT,
            ),
            timeout=1,
        ):

            return True

    except OSError:

        return False


def wait_for_comfyui(
    process,
    timeout=180,
):

    start = time.time()

    while (
        time.time()
        - start
        < timeout
    ):

        if process.poll() is not None:

            raise RuntimeError(
                "ComfyUI exited with code "
                f"{process.returncode}"
            )

        if port_open():

            print(
                f"✅ ComfyUI ready on port "
                f"{PORT}"
            )

            return

        time.sleep(
            1
        )

    raise TimeoutError(
        "ComfyUI did not become ready "
        f"within {timeout}s."
    )


def start_tunnel():

    process = subprocess.Popen(
        [
            str(
                CLOUDFLARED_PATH
            ),
            "tunnel",
            "--url",
            f"http://127.0.0.1:{PORT}",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    output_queue = (
        queue.Queue()
    )

    threading.Thread(
        target=stream,
        args=(
            process,
            "[Tunnel]",
            output_queue,
        ),
        daemon=True,
    ).start()

    return (
        process,
        output_queue,
    )


def wait_for_url(
    process,
    output_queue,
    timeout=60,
):

    start = time.time()

    while (
        time.time()
        - start
        < timeout
    ):

        if process.poll() is not None:

            raise RuntimeError(
                "Cloudflare tunnel exited "
                "before providing a URL."
            )

        try:

            line = (
                output_queue.get(
                    timeout=1
                )
            )

        except queue.Empty:

            continue

        match = URL_PATTERN.search(
            line
        )

        if match:

            return match.group(0)

    raise TimeoutError(
        "Cloudflare tunnel URL not found."
    )


def terminate(
    process,
):

    if (
        process is None
        or process.poll()
        is not None
    ):

        return

    process.terminate()

    try:

        process.wait(
            timeout=10
        )

    except subprocess.TimeoutExpired:

        process.kill()


def main():

    comfy = None
    tunnel = None

    try:

        ensure_cloudflared()

        comfy = (
            start_comfyui()
        )

        wait_for_comfyui(
            comfy
        )

        (
            tunnel,
            output_queue,
        ) = start_tunnel()

        public_url = (
            wait_for_url(
                tunnel,
                output_queue,
            )
        )

        print(
            "=" * 80
        )

        print(
            "✅ LTX-13B MODERN COMFYUI READY"
        )

        print(
            "=" * 80
        )

        print(
            "Local:",
            f"http://127.0.0.1:{PORT}",
        )

        print(
            "Public:",
            public_url,
        )

        print(
            "Keep this Kaggle cell running."
        )

        while True:

            if (
                comfy.poll()
                is not None
            ):

                raise RuntimeError(
                    "ComfyUI stopped unexpectedly."
                )

            if (
                tunnel.poll()
                is not None
            ):

                raise RuntimeError(
                    "Cloudflare tunnel stopped unexpectedly."
                )

            time.sleep(
                5
            )

    finally:

        terminate(
            tunnel
        )

        terminate(
            comfy
        )


if __name__ == "__main__":
    main()
