"""
Start ComfyUI on Kaggle AND expose it to your browser.

Running ComfyUI alone (start_comfyui.py) only makes it listen on
Kaggle's own internal localhost:8188 -- that address means
"Kaggle's computer" and is not reachable from your browser at
home. This script additionally starts a temporary Cloudflare
tunnel, which gives you a public https://....trycloudflare.com
URL that forwards to Kaggle's ComfyUI for as long as this cell
keeps running.

Run this in a Kaggle cell. It will keep running (that's
expected -- it's the server). Watch the output for a line like:

    ComfyUI is ready at: https://random-words-here.trycloudflare.com

Open that URL in your own browser -- that IS the ComfyUI node
interface, running on Kaggle's GPU.
"""

from pathlib import Path
import os
import re
import stat
import subprocess
import sys
import time
import urllib.request

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMFYUI_DIR = Path(
    os.getenv("COMFYUI_DIR", PROJECT_ROOT / "ComfyUI")
)

HOST = os.getenv("COMFYUI_HOST", "0.0.0.0")
PORT = os.getenv("COMFYUI_PORT", "8188")

CLOUDFLARED_PATH = PROJECT_ROOT / "cloudflared"
CLOUDFLARED_URL = (
    "https://github.com/cloudflare/cloudflared/releases/"
    "latest/download/cloudflared-linux-amd64"
)

# The tunnel URL Cloudflare prints always matches this shape.
TUNNEL_URL_PATTERN = re.compile(
    r"https://[a-zA-Z0-9-]+\.trycloudflare\.com"
)


def ensure_cloudflared():
    """Download the cloudflared binary if it isn't already here."""

    if CLOUDFLARED_PATH.exists():
        return

    print("Downloading cloudflared (one-time)...")

    urllib.request.urlretrieve(
        CLOUDFLARED_URL,
        CLOUDFLARED_PATH,
    )

    CLOUDFLARED_PATH.chmod(
        CLOUDFLARED_PATH.stat().st_mode | stat.S_IEXEC
    )

    print("✅ cloudflared ready")


def start_comfyui():
    """Launch ComfyUI as a background process (non-blocking)."""

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

    print("Starting ComfyUI in the background...")

    process = subprocess.Popen(
        command,
        cwd=COMFYUI_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    return process


def wait_for_comfyui_ready(process, timeout=180):
    """Block until ComfyUI's startup log says it's serving."""

    start = time.time()

    for line in process.stdout:
        print("[ComfyUI]", line.rstrip())

        if "To see the GUI go to" in line or "Starting server" in line:
            return True

        if time.time() - start > timeout:
            print("⚠️ Timed out waiting for ComfyUI to start.")
            return False

    return False


def start_tunnel():
    """Launch the Cloudflare quick tunnel pointed at ComfyUI."""

    print("Starting Cloudflare tunnel...")

    command = [
        str(CLOUDFLARED_PATH),
        "tunnel",
        "--url",
        f"http://localhost:{PORT}",
    ]

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    return process


def print_tunnel_url(tunnel_process, timeout=60):
    """Watch cloudflared's own log until it prints the public URL."""

    start = time.time()

    for line in tunnel_process.stdout:

        match = TUNNEL_URL_PATTERN.search(line)

        if match:
            url = match.group(0)

            print()
            print("=" * 70)
            print("✅ ComfyUI is ready at:")
            print(f"   {url}")
            print("=" * 70)
            print("Open that link in your own browser (not Kaggle's).")
            print("It stays valid only while this cell keeps running.")
            print()

            return url

        if time.time() - start > timeout:
            print("⚠️ Timed out waiting for the tunnel URL.")
            return None

    return None


def main():
    ensure_cloudflared()

    comfy_process = start_comfyui()
    comfy_ready = wait_for_comfyui_ready(comfy_process)

    if not comfy_ready:
        print(
            "ComfyUI did not confirm startup in time -- check the "
            "logs above for errors before continuing."
        )

    tunnel_process = start_tunnel()
    print_tunnel_url(tunnel_process)

    print("Keeping this cell alive so both processes keep running...")
    print("(Stop the cell to shut everything down.)")

    # Keep the cell blocked so ComfyUI + the tunnel stay alive.
    # Interleave both processes' remaining output for visibility.
    try:
        while True:
            comfy_line = comfy_process.stdout.readline()
            if comfy_line:
                print("[ComfyUI]", comfy_line.rstrip())

            tunnel_line = tunnel_process.stdout.readline()
            if tunnel_line:
                print("[tunnel]", tunnel_line.rstrip())

            if (
                comfy_process.poll() is not None
                and tunnel_process.poll() is not None
            ):
                break

    except KeyboardInterrupt:
        print("Stopping...")
        comfy_process.terminate()
        tunnel_process.terminate()


if __name__ == "__main__":
    main()
