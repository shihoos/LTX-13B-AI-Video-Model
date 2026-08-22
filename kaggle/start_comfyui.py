from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path


ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

COMFY = ROOT / "ComfyUI"
LOGS = ROOT / "data" / "comfy_logs"


def start(
    gpu_id: int,
    port: int,
):

    LOGS.mkdir(
        parents=True,
        exist_ok=True,
    )

    logfile = (
        LOGS
        / f"gpu_{gpu_id}.log"
    )

    env = os.environ.copy()

    env[
        "CUDA_VISIBLE_DEVICES"
    ] = str(gpu_id)

    env[
        "PYTORCH_CUDA_ALLOC_CONF"
    ] = (
        "expandable_segments:True"
    )

    handle = logfile.open(
        "a",
        encoding="utf-8",
    )

    process = subprocess.Popen(
        [
            sys.executable,
            "main.py",
            "--listen",
            "127.0.0.1",
            "--port",
            str(port),
            "--lowvram",
            "--cpu-vae",
        ],
        cwd=str(COMFY),
        env=env,
        stdout=handle,
        stderr=subprocess.STDOUT,
        text=True,
    )

    return process, handle


def main():

    workers = [
        (0, 8188),
        (1, 8189),
    ]

    processes = [
        start(
            gpu_id,
            port,
        )
        for gpu_id, port
        in workers
    ]

    print(
        "MiniMax H3 workers started:"
    )

    for index, (
        process,
        _handle,
    ) in enumerate(
        processes
    ):

        print(
            f"GPU {index}: "
            f"PID={process.pid} "
            f"PORT={8188 + index}"
        )

    try:

        while True:

            if not any(
                process.poll()
                is None
                for process, _handle
                in processes
            ):
                break

            time.sleep(2)

    except KeyboardInterrupt:

        for process, handle in processes:

            if process.poll() is None:
                process.send_signal(
                    signal.SIGTERM
                )

            handle.close()


if __name__ == "__main__":
    main()
