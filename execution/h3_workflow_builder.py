from __future__ import annotations

from pathlib import Path
from typing import Any


class H3WorkflowBuilder:

    MODEL = (
        "minimax_h3_ref2va_pruned-Q4_K_M.gguf"
    )

    CLIP = (
        "qwen3vl_32b_minimax_h3-Q4_K_M.gguf"
    )

    VIDEO_VAE = (
        "minimax_h3_video_vae_fp16.safetensors"
    )

    AUDIO_VAE = (
        "minimax_h3_audio_vae_fp32.safetensors"
    )

    def __init__(
        self,
        project_root: Path,
        object_info: dict[str, Any],
    ):
        self.project_root = Path(
            project_root
        )
        self.object_info = object_info

    def _require(
        self,
        node_name: str,
    ) -> None:

        if node_name not in self.object_info:
            raise RuntimeError(
                f"Required H3 node missing: {node_name}"
            )

    def validate_runtime(
        self,
    ) -> None:

        required = (
            "H3ModelLoaderAny",
            "H3ClipLoaderAny",
            "MiniMaxH3ReferenceToVideo",
            "H3FreeTextEncoder",
            "VAELoader",
            "VAEDecode",
            "VAEDecodeAudio",
            "RandomNoise",
            "BasicGuider",
            "KSamplerSelect",
            "BasicScheduler",
            "SamplerCustomAdvanced",
            "CreateVideo",
            "SaveVideo",
        )

        for node in required:
            self._require(node)

    @staticmethod
    def _load_image(
        filename: str,
    ) -> dict:

        return {
            "class_type": "LoadImage",
            "inputs": {
                "image": filename,
            },
        }

    @staticmethod
    def _load_audio(
        filename: str,
    ) -> dict:

        return {
            "class_type": "LoadAudio",
            "inputs": {
                "audio": filename,
            },
        }

    @staticmethod
    def _load_video(
        filename: str,
    ) -> dict:

        return {
            "class_type": "VHS_LoadVideo",
            "inputs": {
                "video": filename,
                "force_rate": 24,
                "custom_width": 0,
                "custom_height": 0,
                "frame_load_cap": 0,
                "skip_first_frames": 0,
                "select_every_nth": 1,
            },
        }

    def build_native_ref2va(
        self,
        *,
        prompt: str,
        image_files: list[str],
        video_files: list[str],
        audio_files: list[str],
        width: int,
        height: int,
        frames: int,
        steps: int,
        seed: int,
        output_prefix: str,
        ref_image_size: str = "match",
    ) -> dict:

        self.validate_runtime()

        image_files = image_files[:9]
        video_files = video_files[:3]
        audio_files = audio_files[:3]

        nodes: dict[str, dict] = {}

        nodes["1"] = {
            "class_type": "H3ModelLoaderAny",
            "inputs": {
                "model_name": self.MODEL,
            },
        }

        nodes["2"] = {
            "class_type": "H3ClipLoaderAny",
            "inputs": {
                "model_name": self.CLIP,
                "type": "minimax",
            },
        }

        nodes["3"] = {
            "class_type": "VAELoader",
            "inputs": {
                "vae_name": self.VIDEO_VAE,
            },
        }

        nodes["4"] = {
            "class_type": "VAELoader",
            "inputs": {
                "vae_name": self.AUDIO_VAE,
            },
        }

        # Reference image nodes.
        for index, filename in enumerate(
            image_files
        ):
            node_id = str(
                10 + index
            )

            nodes[node_id] = (
                self._load_image(
                    filename
                )
            )

        # Reference video nodes.
        for index, filename in enumerate(
            video_files
        ):
            node_id = str(
                30 + index
            )

            nodes[node_id] = (
                self._load_video(
                    filename
                )
            )

        # Reference audio nodes.
        for index, filename in enumerate(
            audio_files
        ):
            node_id = str(
                50 + index
            )

            nodes[node_id] = (
                self._load_audio(
                    filename
                )
            )

        ref_inputs: dict[str, Any] = {
            "clip": [
                "2",
                0,
            ],
            "vae": [
                "3",
                0,
            ],
            "audio_vae": [
                "4",
                0,
            ],
            "prompt": prompt,
            "width": int(width),
            "height": int(height),
            "length": int(frames),
            "ref_image_size": ref_image_size,
        }

        # H3 uses 0-based API slots and 1-based prompt ordinals.
        for index in range(
            len(image_files)
        ):
            ref_inputs[
                f"ref_images.ref_image_{index}"
            ] = [
                str(10 + index),
                0,
            ]

        for index in range(
            len(video_files)
        ):
            ref_inputs[
                "ref_videos."
                f"ref_video_{index}"
            ] = [
                str(30 + index),
                0,
            ]

            ref_inputs[
                "ref_video_audios."
                f"ref_video_audio_{index}"
            ] = [
                str(30 + index),
                2,
            ]

        for index in range(
            len(audio_files)
        ):
            ref_inputs[
                "ref_audios."
                f"ref_audio_{index}"
            ] = [
                str(50 + index),
                0,
            ]

        nodes["60"] = {
            "class_type": (
                "MiniMaxH3ReferenceToVideo"
            ),
            "inputs": ref_inputs,
        }

        # Free Qwen3-VL after conditioning.
        nodes["61"] = {
            "class_type": "H3FreeTextEncoder",
            "inputs": {
                "conditioning": [
                    "60",
                    0,
                ],
                "clip": [
                    "2",
                    0,
                ],
            },
        }

        nodes["62"] = {
            "class_type": "RandomNoise",
            "inputs": {
                "noise_seed": int(seed),
            },
        }

        nodes["63"] = {
            "class_type": "KSamplerSelect",
            "inputs": {
                "sampler_name": "res_multistep",
            },
        }

        nodes["64"] = {
            "class_type": "BasicScheduler",
            "inputs": {
                "model": [
                    "1",
                    0,
                ],
                "scheduler": "beta",
                "steps": int(steps),
                "denoise": 1.0,
            },
        }

        nodes["65"] = {
            "class_type": "BasicGuider",
            "inputs": {
                "model": [
                    "1",
                    0,
                ],
                "conditioning": [
                    "61",
                    0,
                ],
            },
        }

        nodes["66"] = {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": [
                    "62",
                    0,
                ],
                "guider": [
                    "65",
                    0,
                ],
                "sampler": [
                    "63",
                    0,
                ],
                "sigmas": [
                    "64",
                    0,
                ],
                "latent_image": [
                    "60",
                    1,
                ],
            },
        }

        nodes["67"] = {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": [
                    "66",
                    0,
                ],
                "vae": [
                    "3",
                    0,
                ],
            },
        }

        nodes["68"] = {
            "class_type": "VAEDecodeAudio",
            "inputs": {
                "samples": [
                    "66",
                    0,
                ],
                "vae": [
                    "4",
                    0,
                ],
            },
        }

        nodes["69"] = {
            "class_type": "CreateVideo",
            "inputs": {
                "images": [
                    "67",
                    0,
                ],
                "audio": [
                    "68",
                    0,
                ],
                "frame_rate": 24,
                "loop_count": 1,
            },
        }

        nodes["70"] = {
            "class_type": "SaveVideo",
            "inputs": {
                "video": [
                    "69",
                    0,
                ],
                "filename_prefix": output_prefix,
            },
        }

        return nodes
