from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any


class H3WorkflowBuilder:
    """
    Builds an H3-Multishot Ref2VA API graph from the current ComfyUI /object_info.

    The graph intentionally avoids hard-coding custom-node widget names where
    ComfyUI exposes them dynamically. The H3-Multishot pack provides:
      H3ModelLoaderAny
      H3ClipLoaderAny
      H3MultishotMemorySampler
      H3ReferenceAudio

    Reference inputs are linked only when actually provided.
    """

    def __init__(
        self,
        project_root: Path,
        object_info: dict[str, Any],
    ):
        self.project_root = Path(project_root)
        self.object_info = object_info

    def _node_info(self, class_type: str) -> dict:
        info = self.object_info.get(class_type)
        if not isinstance(info, dict):
            raise RuntimeError(
                f"Required ComfyUI node is unavailable: {class_type}"
            )
        return info

    def _make_model_loader(self, node_id: str, filename: str) -> dict:
        self._node_info("H3ModelLoaderAny")
        return {
            "class_type": "H3ModelLoaderAny",
            "inputs": self._best_model_input(
                "H3ModelLoaderAny",
                filename,
                type_value="minimax",
            ),
        }

    def _best_model_input(
        self,
        class_type: str,
        filename: str,
        type_value: str | None = None,
    ) -> dict:
        info = self._node_info(class_type)
        required = info.get("input", {}).get("required", {})

        result: dict[str, Any] = {}

        # H3ModelLoaderAny/H3ClipLoaderAny currently use a model filename
        # combo and H3's type selector. Handle common name variants.
        model_key = None
        for key in ("model_name", "model", "checkpoint", "ckpt_name"):
            if key in required:
                model_key = key
                break

        if model_key:
            result[model_key] = filename
        else:
            # Fall back to the first non-type required widget.
            for key in required:
                if key not in ("type", "device", "weight_dtype"):
                    result[key] = filename
                    break

        if type_value is not None and "type" in required:
            result["type"] = type_value

        return result

    def _add_image_loader(
        self,
        nodes: dict,
        node_id: str,
        filename: str,
    ):
        nodes[node_id] = {
            "class_type": "LoadImage",
            "inputs": {"image": filename},
        }
        return [node_id, 0]

    @staticmethod
    def _add_batch(
        nodes: dict,
        node_id: str,
        left,
        right,
    ):
        nodes[node_id] = {
            "class_type": "ImageBatch",
            "inputs": {
                "image1": left,
                "image2": right,
            },
        }
        return [node_id, 0]

    def build(
        self,
        *,
        script: str,
        image_files: list[str],
        voice_audio: str | None,
        reference_video: str | None,
        width: int,
        height: int,
        frames_per_shot: int,
        steps: int,
        output_prefix: str,
    ) -> dict:
        for required_node in (
            "H3ModelLoaderAny",
            "H3ClipLoaderAny",
            "H3MultishotMemorySampler",
            "H3ReferenceAudio",
            "VAELoader",
            "CreateVideo",
            "SaveVideo",
        ):
            self._node_info(required_node)

        nodes: dict[str, dict] = {}

        # ------------------------------------------------------------
        # Model / encoder / VAEs.
        # ------------------------------------------------------------
        nodes["1"] = self._make_model_loader(
            "1",
            "minimax_h3_ref2va_pruned-Q4_K_M.gguf",
        )

        nodes["2"] = {
            "class_type": "H3ClipLoaderAny",
            "inputs": self._best_model_input(
                "H3ClipLoaderAny",
                "qwen3vl_32b_minimax_h3-Q4_K_M.gguf",
                type_value="minimax",
            ),
        }

        nodes["3"] = {
            "class_type": "VAELoader",
            "inputs": {
                "vae_name": "minimax_h3_video_vae_fp16.safetensors",
            },
        }

        nodes["4"] = {
            "class_type": "VAELoader",
            "inputs": {
                "vae_name": "minimax_h3_audio_vae_fp32.safetensors",
            },
        }

        # ------------------------------------------------------------
        # References.
        # ------------------------------------------------------------
        reference_link = None
        if image_files:
            image_files = image_files[:9]
            first = self._add_image_loader(nodes, "10", image_files[0])
            reference_link = first

            next_id = 11
            for index, filename in enumerate(image_files[1:], start=1):
                current = self._add_image_loader(
                    nodes,
                    str(next_id),
                    filename,
                )
                reference_link = self._add_batch(
                    nodes,
                    str(next_id + 20),
                    reference_link,
                    current,
                )
                next_id += 1

        voice_link = None
        if voice_audio:
            nodes["50"] = {
                "class_type": "LoadAudio",
                "inputs": {"audio": voice_audio},
            }
            nodes["51"] = {
                "class_type": "H3ReferenceAudio",
                "inputs": {"audio": ["50", 0]},
            }
            voice_link = ["51", 0]

        video_link = None
        video_audio_link = None
        if reference_video:
            self._node_info("VHS_LoadVideo")
            nodes["60"] = {
                "class_type": "VHS_LoadVideo",
                "inputs": {
                    "video": reference_video,
                    "force_rate": 2,
                    "custom_width": 0,
                    "custom_height": 0,
                    "frame_load_cap": 48,
                    "skip_first_frames": 0,
                    "select_every_nth": 1,
                },
            }
            video_link = ["60", 0]
            video_audio_link = ["60", 2]

        # ------------------------------------------------------------
        # H3 Memory Sampler.
        # ------------------------------------------------------------
        sampler_inputs: dict[str, Any] = {
            "model": ["1", 0],
            "clip": ["2", 0],
            "video_vae": ["3", 0],
            "audio_vae": ["4", 0],
            "script": script,
            "shot_count": 0,
            "width": int(width),
            "height": int(height),
            "frames_per_shot": int(frames_per_shot),
            "steps": int(steps),
        }

        if reference_link:
            sampler_inputs["reference_images"] = reference_link

        if voice_link:
            sampler_inputs["voice_ref"] = voice_link

        if video_link:
            sampler_inputs["reference_video"] = video_link

        if video_audio_link:
            sampler_inputs["reference_video_audio"] = video_audio_link

        nodes["70"] = {
            "class_type": "H3MultishotMemorySampler",
            "inputs": sampler_inputs,
        }

        nodes["80"] = {
            "class_type": "CreateVideo",
            "inputs": {
                "images": ["70", 0],
                "audio": ["70", 1],
                "frame_rate": 24,
                "loop_count": 1,
            },
        }

        nodes["81"] = {
            "class_type": "SaveVideo",
            "inputs": {
                "video": ["80", 0],
                "filename_prefix": output_prefix,
            },
        }

        return nodes
