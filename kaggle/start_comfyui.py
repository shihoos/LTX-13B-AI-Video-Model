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

COMFY = (
    ROOT
    / "ComfyUI"
)

LOG_ROOT = (
    ROOT
    / "data"
    / "comfy_logs"
)


def start_worker(
    gpu_id: int,
    port: int,
):

    LOG_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    logfile = (
        LOG_ROOT
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

    command = [
        sys.executable,
        "main.py",
        "--listen",
        "127.0.0.1",
        "--port",
        str(port),
        "--lowvram",
        "--cpu-vae",
    ]

    print(
        f"Starting H3 GPU worker "
        f"{gpu_id} on port {port}"
    )

    handle = logfile.open(
        "a",
        encoding="utf-8",
    )

    process = subprocess.Popen(
        command,
        cwd=str(COMFY),
        env=env,
        stdout=handle,
        stderr=subprocess.STDOUT,
        text=True,
    )

    return (
        process,
        handle,
    )


def main():

    workers = [
        (
            0,
            8188,
        ),
        (
            1,
            8189,
        ),
    ]

    processes = []

    for gpu_id, port in workers:

        processes.append(
            start_worker(
                gpu_id,
                port,
            )
        )

    print(
        "\nWorkers:"
    )

    for index, (
        process,
        handle,
    ) in enumerate(processes):

        print(
            f"GPU {index}: "
            f"PID {process.pid}"
        )

    try:

        while True:

            alive = False

            for process, _handle in processes:

                if process.poll() is None:
                    alive = True

            if not alive:
                break

            time.sleep(3)

    except KeyboardInterrupt:

        print(
            "\nStopping H3 workers..."
        )

        for process, handle in processes:

            if process.poll() is None:

                process.send_signal(
                    signal.SIGTERM
                )

            handle.close()


if __name__ == "__main__":
    main()
