from __future__ import annotations

import hashlib
import json
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


# ============================================================================
# PROJECT
# ============================================================================

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

LOCK_FILE = (
    PROJECT_ROOT
    / "kaggle"
    / "compatibility_lock.yaml"
)

URL_PATTERN = re.compile(
    r"https://[a-zA-Z0-9-]+\.trycloudflare\.com"
)


# ============================================================================
# HELPERS
# ============================================================================

def fail(
    message: str,
) -> None:

    raise RuntimeError(
        message
    )


def load_lock() -> dict:

    try:

        import yaml

    except ImportError as error:

        fail(
            "PyYAML is required to read "
            "compatibility_lock.yaml.\n"
            f"{error}"
        )

    if not LOCK_FILE.is_file():

        fail(
            "Compatibility lock not found:\n"
            f"{LOCK_FILE}"
        )

    try:

        data = yaml.safe_load(
            LOCK_FILE.read_text(
                encoding="utf-8"
            )
        )

    except Exception as error:

        fail(
            "Could not parse "
            "compatibility_lock.yaml:\n"
            f"{error}"
        )

    if not isinstance(
        data,
        dict,
    ):

        fail(
            "compatibility_lock.yaml "
            "must contain a mapping."
        )

    return data


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


# ============================================================================
# CLOUDFLARED
# ============================================================================

def get_cloudflared_config(
    lock: dict,
) -> dict:

    comfy = lock.get(
        "comfyui"
    )

    if not isinstance(
        comfy,
        dict,
    ):

        fail(
            "compatibility_lock.yaml "
            "has no valid comfyui section."
        )

    config = comfy.get(
        "cloudflared"
    )

    if not isinstance(
        config,
        dict,
    ):

        fail(
            "compatibility_lock.yaml "
            "has no valid comfyui.cloudflared section."
        )

    required = (
        "version",
        "asset",
        "url",
        "sha256",
    )

    for field in required:

        value = config.get(
            field
        )

        if not isinstance(
            value,
            str,
        ) or not value.strip():

            fail(
                "Missing cloudflared lock field:\n"
                f"comfyui.cloudflared.{field}"
            )

    if config["asset"] != (
        "cloudflared-linux-amd64"
    ):

        fail(
            "Unsupported cloudflared asset:\n"
            f"{config['asset']}\n"
            "Expected: cloudflared-linux-amd64"
        )

    if len(
        config["sha256"]
    ) != 64:

        fail(
            "cloudflared SHA-256 must "
            "contain 64 hexadecimal characters."
        )

    try:

        int(
            config["sha256"],
            16,
        )

    except ValueError as error:

        fail(
            "cloudflared SHA-256 is not valid "
            "hexadecimal."
        )

    return config


def sha256_file(
    path: Path,
) -> str:

    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as file:

        for chunk in iter(
            lambda: file.read(
                1024 * 1024
            ),
            b"",
        ):

            digest.update(
                chunk
            )

    return digest.hexdigest()


def ensure_cloudflared(
    lock: dict,
) -> None:

    config = get_cloudflared_config(
        lock
    )

    expected_sha256 = (
        config["sha256"]
        .lower()
    )

    # ------------------------------------------------------------------------
    # Existing binary
    # ------------------------------------------------------------------------

    if CLOUDFLARED_PATH.is_file():

        actual = sha256_file(
            CLOUDFLARED_PATH
        ).lower()

        if actual == expected_sha256:

            os.chmod(
                CLOUDFLARED_PATH,
                0o755,
            )

            print(
                "✅ cloudflared verified"
            )

            print(
                f"   version: {config['version']}"
            )

            print(
                f"   sha256:  {actual}"
            )

            return

        print(
            "⚠️ Existing cloudflared checksum "
            "does not match the locked binary."
        )

        print(
            f"   expected: {expected_sha256}"
        )

        print(
            f"   actual:   {actual}"
        )

        CLOUDFLARED_PATH.unlink()

    # ------------------------------------------------------------------------
    # Download exact locked release
    # ------------------------------------------------------------------------

    print(
        "Downloading locked cloudflared..."
    )

    print(
        f"Version: {config['version']}"
    )

    print(
        f"URL: {config['url']}"
    )

    temporary_path = (
        PROJECT_ROOT
        / "cloudflared.download"
    )

    if temporary_path.exists():

        temporary_path.unlink()

    try:

        urllib.request.urlretrieve(
            config["url"],
            temporary_path,
        )

    except Exception as error:

        if temporary_path.exists():

            temporary_path.unlink()

        fail(
            "Failed to download cloudflared:\n"
            f"{error}"
        )

    actual = sha256_file(
        temporary_path
    ).lower()

    if actual != expected_sha256:

        temporary_path.unlink()

        fail(
            "cloudflared SHA-256 verification failed.\n"
            f"Expected: {expected_sha256}\n"
            f"Actual:   {actual}"
        )

    temporary_path.replace(
        CLOUDFLARED_PATH
    )

    os.chmod(
        CLOUDFLARED_PATH,
        0o755,
    )

    # Final verification after rename.
    final_hash = sha256_file(
        CLOUDFLARED_PATH
    ).lower()

    if final_hash != expected_sha256:

        CLOUDFLARED_PATH.unlink()

        fail(
            "cloudflared final verification failed."
        )

    print(
        "✅ cloudflared downloaded and verified"
    )

    print(
        f"   version: {config['version']}"
    )

    print(
        f"   sha256:  {final_hash}"
    )


# ============================================================================
# COMFYUI
# ============================================================================

def start_comfyui():

    if not (
        COMFYUI_DIR
        / "main.py"
    ).exists():

        raise FileNotFoundError(
            f"ComfyUI not found:\n"
            f"{COMFYUI_DIR}"
        )

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


# ============================================================================
# TUNNEL
# ============================================================================

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


# ============================================================================
# CLEANUP
# ============================================================================

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


# ============================================================================
# MAIN
# ============================================================================

def main():

    comfy = None
    tunnel = None

    try:

        lock = load_lock()

        ensure_cloudflared(
            lock
        )

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
            "Cloudflared:",
            "locked + SHA-256 verified",
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
