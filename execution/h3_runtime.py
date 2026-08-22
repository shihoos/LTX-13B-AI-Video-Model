from __future__ import annotations

import gc
import os
import subprocess
import time
from pathlib import Path


class H3Runtime:

    @staticmethod
    def clear_cuda():
        try:
            import torch

            gc.collect()

            if not torch.cuda.is_available():
                return

            for device_id in range(
                torch.cuda.device_count()
            ):
                with torch.cuda.device(
                    device_id
                ):
                    torch.cuda.empty_cache()

                    try:
                        torch.cuda.ipc_collect()
                    except Exception:
                        pass

        except Exception:
            pass

    @staticmethod
    def worker_environment(
        gpu_id: int,
    ) -> dict[str, str]:

        env = os.environ.copy()

        env[
            "CUDA_VISIBLE_DEVICES"
        ] = str(gpu_id)

        env[
            "PYTORCH_CUDA_ALLOC_CONF"
        ] = (
            "expandable_segments:True"
        )

        env[
            "PYTHONUNBUFFERED"
        ] = "1"

        return env

    @staticmethod
    def launch_worker(
        comfy_root: Path,
        gpu_id: int,
        port: int,
        log_path: Path,
    ):

        log_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        handle = log_path.open(
            "a",
            encoding="utf-8",
        )

        process = subprocess.Popen(
            [
                "python",
                "main.py",
                "--listen",
                "127.0.0.1",
                "--port",
                str(port),
                "--lowvram",
                "--cpu-vae",
            ],
            cwd=str(comfy_root),
            env=H3Runtime.worker_environment(
                gpu_id
            ),
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )

        return process, handle

    @staticmethod
    def wait_http(
        url: str,
        timeout: int = 300,
    ):

        import urllib.request

        started = time.time()

        while (
            time.time() - started
            < timeout
        ):

            try:
                with urllib.request.urlopen(
                    url,
                    timeout=5,
                ) as response:

                    if response.status == 200:
                        return

            except Exception:
                pass

            time.sleep(2)

        raise TimeoutError(
            f"ComfyUI did not start: {url}"
        )
