"""
Start ComfyUI on Kaggle and expose it through a temporary
Cloudflare Quick Tunnel.

Flow:

    ComfyUI
       ↓
    localhost:8188
       ↓
    cloudflared
       ↓
    https://xxxx.trycloudflare.com
"""

from pathlib import Path
import os
import queue
import re
import socket
import subprocess
import sys
import threading
import time
import urllib.request


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

COMFYUI_DIR = Path(
    os.getenv(
        "COMFYUI_DIR",
        str(PROJECT_ROOT / "ComfyUI"),
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
    PROJECT_ROOT / "cloudflared"
)

CLOUDFLARED_URL = (
    "https://github.com/cloudflare/"
    "cloudflared/releases/latest/download/"
    "cloudflared-linux-amd64"
)


# ============================================================
# CLOUDFLARE URL DETECTION
# ============================================================

TUNNEL_URL_PATTERN = re.compile(
    r"https://[a-zA-Z0-9-]+\.trycloudflare\.com"
)


# ============================================================
# PROCESS OUTPUT
# ============================================================

def stream_output(
    process,
    prefix,
    output_queue=None,
):
    """
    Continuously read subprocess output so the child process
    does not block because its stdout pipe becomes full.
    """

    if process.stdout is None:
        return

    try:
        for line in iter(
            process.stdout.readline,
            "",
        ):
            if not line:
                break

            line = line.rstrip()

            if output_queue is not None:
                output_queue.put(line)

            print(
                f"{prefix} {line}",
                flush=True,
            )

    except Exception as exc:
        print(
            f"{prefix} output error: {exc}",
            flush=True,
        )


# ============================================================
# CLOUDFLARED
# ============================================================

def ensure_cloudflared():
    """
    Download cloudflared if necessary and always restore its
    executable permission.
    """

    if not CLOUDFLARED_PATH.exists():

        print(
            "Downloading cloudflared..."
        )

        urllib.request.urlretrieve(
            CLOUDFLARED_URL,
            CLOUDFLARED_PATH,
        )

        print(
            "✅ cloudflared downloaded"
        )

    # Kaggle may preserve the file but not its executable bit.
    os.chmod(
        CLOUDFLARED_PATH,
        0o755,
    )

    if not os.access(
        CLOUDFLARED_PATH,
        os.X_OK,
    ):
        raise RuntimeError(
            "cloudflared exists but is not executable:\n"
            f"{CLOUDFLARED_PATH}"
        )

    print(
        "✅ cloudflared executable:"
    )
    print(
        f"   {CLOUDFLARED_PATH}"
    )


# ============================================================
# COMFYUI
# ============================================================

def verify_comfyui():
    """
    Verify that the ComfyUI installation exists.
    """

    main_py = (
        COMFYUI_DIR / "main.py"
    )

    if not main_py.exists():

        raise FileNotFoundError(
            "ComfyUI was not found at:\n"
            f"{COMFYUI_DIR}\n\n"
            "Run the project bootstrap first."
        )


def start_comfyui():
    """
    Start ComfyUI in the background.
    """

    verify_comfyui()

    command = [
        sys.executable,
        "main.py",
        "--listen",
        HOST,
        "--port",
        str(PORT),
    ]

    print()
    print("=" * 70)
    print("STARTING COMFYUI")
    print("=" * 70)

    print(
        "Command:",
        " ".join(command),
    )

    process = subprocess.Popen(
        command,
        cwd=COMFYUI_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    thread = threading.Thread(
        target=stream_output,
        args=(
            process,
            "[ComfyUI]",
        ),
        daemon=True,
    )

    thread.start()

    return process


# ============================================================
# PORT CHECK
# ============================================================

def port_is_open(
    host="127.0.0.1",
    port=8188,
):
    """
    Check whether a TCP server is accepting connections.
    """

    try:

        with socket.create_connection(
            (host, port),
            timeout=1,
        ):
            return True

    except OSError:

        return False


# ============================================================
# WAIT FOR COMFYUI
# ============================================================

def wait_for_comfyui(
    process,
    timeout=180,
):
    """
    Wait until ComfyUI is actually listening on its TCP port.

    This is more reliable than depending on a specific startup
    log message.
    """

    print()
    print(
        "Waiting for ComfyUI to become ready..."
    )

    start_time = time.time()

    while (
        time.time() - start_time
        < timeout
    ):

        return_code = process.poll()

        if return_code is not None:

            raise RuntimeError(
                "ComfyUI exited before becoming ready.\n"
                f"Exit code: {return_code}"
            )

        if port_is_open(
            "127.0.0.1",
            PORT,
        ):

            print()
            print(
                "✅ ComfyUI is listening on "
                f"port {PORT}"
            )

            return

        time.sleep(1)

    raise TimeoutError(
        "ComfyUI did not start within "
        f"{timeout} seconds."
    )


# ============================================================
# CLOUDFLARE TUNNEL
# ============================================================

def start_tunnel():
    """
    Start a Cloudflare Quick Tunnel pointing at ComfyUI.
    """

    command = [
        str(CLOUDFLARED_PATH),
        "tunnel",
        "--url",
        f"http://127.0.0.1:{PORT}",
    ]

    print()
    print("=" * 70)
    print("STARTING CLOUDFLARE QUICK TUNNEL")
    print("=" * 70)

    print(
        "Command:",
        " ".join(command),
    )

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    output_queue = queue.Queue()

    thread = threading.Thread(
        target=stream_output,
        args=(
            process,
            "[Tunnel]",
            output_queue,
        ),
        daemon=True,
    )

    thread.start()

    return process, output_queue


# ============================================================
# WAIT FOR TUNNEL URL
# ============================================================

def wait_for_tunnel_url(
    tunnel_process,
    output_queue,
    timeout=60,
):
    """
    Wait for cloudflared to print its public URL.
    """

    print()
    print(
        "Waiting for Cloudflare tunnel URL..."
    )

    start_time = time.time()

    while (
        time.time() - start_time
        < timeout
    ):

        if tunnel_process.poll() is not None:

            raise RuntimeError(
                "cloudflared exited before "
                "providing a tunnel URL.\n"
                f"Exit code: "
                f"{tunnel_process.returncode}"
            )

        try:

            line = output_queue.get(
                timeout=1
            )

        except queue.Empty:

            continue

        match = TUNNEL_URL_PATTERN.search(
            line
        )

        if match:

            return match.group(0)

    raise TimeoutError(
        "Cloudflare did not provide a "
        "trycloudflare.com URL within "
        f"{timeout} seconds."
    )


# ============================================================
# PROCESS CLEANUP
# ============================================================

def terminate_process(
    process,
    name,
):
    """
    Gracefully terminate a subprocess and force-kill it if
    necessary.
    """

    if process is None:
        return

    if process.poll() is None:

        print(
            f"Stopping {name}..."
        )

        process.terminate()

        try:

            process.wait(
                timeout=10
            )

        except subprocess.TimeoutExpired:

            print(
                f"{name} did not stop cleanly. "
                "Killing it..."
            )

            process.kill()

            try:
                process.wait(
                    timeout=5
                )
            except subprocess.TimeoutExpired:
                pass


# ============================================================
# MAIN
# ============================================================

def main():

    comfy_process = None
    tunnel_process = None

    try:

        print()
        print("=" * 70)
        print(
            "LTX-13B COMFYUI + CLOUDFLARE"
        )
        print("=" * 70)

        # ----------------------------------------------------
        # 1. Ensure cloudflared exists
        # ----------------------------------------------------

        ensure_cloudflared()

        # ----------------------------------------------------
        # 2. Start ComfyUI
        # ----------------------------------------------------

        comfy_process = start_comfyui()

        # ----------------------------------------------------
        # 3. Wait until ComfyUI actually accepts connections
        # ----------------------------------------------------

        wait_for_comfyui(
            comfy_process
        )

        # ----------------------------------------------------
        # 4. Start Cloudflare only AFTER ComfyUI is ready
        # ----------------------------------------------------

        (
            tunnel_process,
            output_queue,
        ) = start_tunnel()

        # ----------------------------------------------------
        # 5. Wait for public tunnel URL
        # ----------------------------------------------------

        url = wait_for_tunnel_url(
            tunnel_process,
            output_queue,
        )

        # ----------------------------------------------------
        # 6. Display URL
        # ----------------------------------------------------

        print()
        print("=" * 70)
        print("✅ COMFYUI IS READY")
        print("=" * 70)
        print()
        print(
            "Open this URL in your normal Chrome/Edge:"
        )
        print()
        print(url)
        print()
        print(
            "The URL remains available while "
            "this Kaggle cell is running."
        )
        print()
        print("=" * 70)
        print(
            "Keeping ComfyUI + Cloudflare alive..."
        )
        print(
            "Stop the cell to shut them down."
        )
        print("=" * 70)

        # ----------------------------------------------------
        # 7. Keep both processes alive
        # ----------------------------------------------------

        while True:

            if comfy_process.poll() is not None:

                raise RuntimeError(
                    "ComfyUI stopped unexpectedly."
                )

            if tunnel_process.poll() is not None:

                raise RuntimeError(
                    "Cloudflare tunnel stopped unexpectedly."
                )

            time.sleep(5)

    except KeyboardInterrupt:

        print(
            "\nStopping LTX services..."
        )

    except Exception as exc:

        print()
        print("=" * 70)
        print("❌ LTX STARTUP FAILED")
        print("=" * 70)
        print()
        print(exc)
        print()

        raise

    finally:

        terminate_process(
            tunnel_process,
            "Cloudflare tunnel",
        )

        terminate_process(
            comfy_process,
            "ComfyUI",
        )

        print(
            "✅ Services stopped."
        )


if __name__ == "__main__":
    main()
