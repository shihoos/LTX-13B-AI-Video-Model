from __future__ import annotations

import gc
import os
import subprocess
import time
from pathlib import Path
from typing import Optional

import torch


class H3Runtime:

    def __init__(
        self,
        project_root: Path,
    ):
        self.project_root = Path(
            project_root
        )

    @staticmethod
    def cuda_devices() -> list[int]:
        if not torch.cuda.is_available():
            return []

        return list(
            range(
                torch.cuda.device_count()
            )
        )

    @staticmethod
    def clear_cuda() -> None:
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

    @staticmethod
    def release_qwen(
        model,
        tokenizer=None,
    ) -> None:

        if model is not None:
            del model

        if tokenizer is not None:
            del tokenizer

        H3Runtime.clear_cuda()

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
    def worker_port(
        base_port: int,
        gpu_id: int,
    ) -> int:

        return (
            int(base_port)
            + int(gpu_id)
        )

    @staticmethod
    def launch_comfyui(
        comfy_root: Path,
        gpu_id: int,
        port: int,
        logfile: Path,
    ):

        logfile.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        command = [
            "python",
            "main.py",
            "--listen",
            "127.0.0.1",
            "--port",
            str(port),
            "--lowvram",
            "--cpu-vae",
        ]

        log_handle = logfile.open(
            "a",
            encoding="utf-8",
        )

        process = subprocess.Popen(
            command,
            cwd=str(comfy_root),
            env=(
                H3Runtime
                .worker_environment(
                    gpu_id
                )
            ),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )

        return process, log_handle

    @staticmethod
    def wait_for_http(
        url: str,
        timeout: int = 300,
    ) -> None:

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
            f"ComfyUI did not become ready: {url}"
        )

    @staticmethod
    def assert_required_nodes(
        object_info: dict,
    ) -> None:

        required = {
            "H3ModelLoaderAny",
            "H3ClipLoaderAny",
            "MiniMaxH3ReferenceToVideo",
            "H3FreeTextEncoder",
            "H3ReferenceAudio",
            "H3MultishotMemorySampler",
            "VAEDecode",
            "VAEDecodeAudio",
            "SamplerCustomAdvanced",
            "CreateVideo",
            "SaveVideo",
        }

        missing = (
            required
            - set(object_info)
        )

        if missing:
            raise RuntimeError(
                "H3 runtime missing nodes:\n"
                + "\n".join(
                    sorted(missing)
                )
            )
