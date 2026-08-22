from __future__ import annotations

import shutil
from pathlib import Path

from execution.h3_workflow_builder import (
    H3WorkflowBuilder,
)


class ShotExecutor:

    def __init__(
        self,
        comfy_client,
        project_root: Path,
        comfy_input_dir: Path,
    ):
        self.client = comfy_client

        self.project_root = Path(
            project_root
        )

        self.comfy_input_dir = Path(
            comfy_input_dir
        )

        self.comfy_input_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.object_info = (
            self.client
            .get_object_info()
        )

        self.builder = (
            H3WorkflowBuilder(
                project_root=(
                    self.project_root
                ),
                object_info=(
                    self.object_info
                ),
            )
        )

    def _copy_input(
        self,
        source: str,
        prefix: str,
    ) -> str:

        source_path = Path(
            source
        )

        if not source_path.is_file():
            raise FileNotFoundError(
                source_path
            )

        safe_name = "".join(
            char
            if (
                char.isalnum()
                or char in "._-"
            )
            else "_"
            for char
            in source_path.name
        )

        destination = (
            self.comfy_input_dir
            / f"{prefix}_{safe_name}"
        )

        shutil.copy2(
            source_path,
            destination,
        )

        return destination.name

    @staticmethod
    def _prompt(
        shot: dict,
    ) -> str:

        parts = []

        parts.extend(
            shot.get(
                "reference_bindings",
                [],
            )
        )

        parts.extend(
            shot.get(
                "identity_locks",
                [],
            )
        )

        parts.append(
            "TARGET SHOT:"
        )

        parts.append(
            shot.get(
                "visual_prompt",
                "",
            )
        )

        parts.append(
            "ACTION:"
        )

        parts.append(
            shot.get(
                "action",
                "",
            )
        )

        parts.append(
            "CAMERA:"
        )

        parts.append(
            shot.get(
                "camera_shot",
                "",
            )
        )

        parts.append(
            shot.get(
                "camera_movement",
                "",
            )
        )

        parts.append(
            "LIGHTING:"
        )

        parts.append(
            shot.get(
                "lighting",
                "",
            )
        )

        parts.append(
            "MOOD:"
        )

        parts.append(
            shot.get(
                "mood",
                "",
            )
        )

        parts.append(
            "CONTINUITY:"
        )

        parts.append(
            shot.get(
                "continuity_notes",
                "",
            )
        )

        speech = str(
            shot.get(
                "speech_text",
                "",
            )
            or ""
        ).strip()

        if speech:
            parts.extend(
                [
                    "DIALOGUE:",
                    speech,
                ]
            )

        negative = str(
            shot.get(
                "negative_prompt",
                "",
            )
            or ""
        ).strip()

        if negative:
            parts.extend(
                [
                    "NEGATIVE:",
                    negative,
                ]
            )

        return "\n".join(
            str(part)
            for part in parts
            if str(part).strip()
        )

    def _native_workflow(
        self,
        shot: dict,
    ) -> dict:

        image_files = [
            self._copy_input(
                path,
                "ref_image",
            )
            for path in (
                shot.get(
                    "reference_images",
                    [],
                )[:9]
            )
        ]

        video_files = [
            self._copy_input(
                path,
                "ref_video",
            )
            for path in (
                shot.get(
                    "reference_videos",
                    [],
                )[:3]
            )
        ]

        audio_paths = list(
            shot.get(
                "reference_audio_paths",
                [],
            )
        )

        voice = shot.get(
            "reference_audio"
        )

        if (
            voice
            and voice not in audio_paths
        ):
            audio_paths.insert(
                0,
                voice,
            )

        audio_files = [
            self._copy_input(
                path,
                "ref_audio",
            )
            for path in audio_paths[:3]
        ]

        seed = shot.get(
            "seed"
        )

        if seed is None:
            seed = 0

        return (
            self.builder
            .build_native_ref2va(
                prompt=self._prompt(
                    shot
                ),
                image_files=image_files,
                video_files=video_files,
                audio_files=audio_files,
                width=int(
                    shot.get(
                        "width",
                        960,
                    )
                ),
                height=int(
                    shot.get(
                        "height",
                        544,
                    )
                ),
                frames=int(
                    shot.get(
                        "frames_per_shot",
                        124,
                    )
                ),
                steps=int(
                    shot.get(
                        "steps",
                        14,
                    )
                ),
                seed=int(seed),
                output_prefix=(
                    "h3/native/"
                    + str(
                        shot[
                            "shot_id"
                        ]
                    )
                ),
            )
        )

    def _memory_workflow(
        self,
        shot: dict,
    ) -> dict:

        workflow = {
            "1": {
                "class_type": (
                    "H3ModelLoaderAny"
                ),
                "inputs": {
                    "model_name": (
                        "minimax_h3_ref2va_"
                        "pruned-Q4_K_M.gguf"
                    )
                },
            },
            "2": {
                "class_type": (
                    "H3ClipLoaderAny"
                ),
                "inputs": {
                    "model_name": (
                        "qwen3vl_32b_minimax_h3_"
                        "Q4_K_M.gguf"
                    ),
                    "type": "minimax",
                },
            },
            "3": {
                "class_type": "VAELoader",
                "inputs": {
                    "vae_name": (
                        "minimax_h3_video_vae_"
                        "fp16.safetensors"
                    )
                },
            },
            "4": {
                "class_type": "VAELoader",
                "inputs": {
                    "vae_name": (
                        "minimax_h3_audio_vae_"
                        "fp32.safetensors"
                    )
                },
            },
            "70": {
                "class_type": (
                    "H3MultishotMemorySampler"
                ),
                "inputs": {
                    "model": [
                        "1",
                        0,
                    ],
                    "clip": [
                        "2",
                        0,
                    ],
                    "video_vae": [
                        "3",
                        0,
                    ],
                    "audio_vae": [
                        "4",
                        0,
                    ],
                    "script": (
                        self._prompt(
                            shot
                        )
                    ),
                    "shot_count": 1,
                    "width": int(
                        shot.get(
                            "width",
                            960,
                        )
                    ),
                    "height": int(
                        shot.get(
                            "height",
                            544,
                        )
                    ),
                    "frames_per_shot": int(
                        shot.get(
                            "frames_per_shot",
                            124,
                        )
                    ),
                    "steps": int(
                        shot.get(
                            "steps",
                            14,
                        )
                    ),
                },
            },
            "80": {
                "class_type": "CreateVideo",
                "inputs": {
                    "images": [
                        "70",
                        0,
                    ],
                    "audio": [
                        "70",
                        1,
                    ],
                    "frame_rate": 24,
                    "loop_count": 1,
                },
            },
            "81": {
                "class_type": "SaveVideo",
                "inputs": {
                    "video": [
                        "80",
                        0,
                    ],
                    "filename_prefix": (
                        "h3/memory/"
                        + str(
                            shot["shot_id"]
                        )
                    ),
                },
            },
        }

        return workflow

    def execute(
        self,
        shot: dict,
        output_dir: Path,
        native_ref2va: bool = True,
    ) -> Path:

        if native_ref2va:
            workflow = (
                self._native_workflow(
                    shot
                )
            )
        else:
            workflow = (
                self._memory_workflow(
                    shot
                )
            )

        prompt_id = (
            self.client.queue_prompt(
                workflow
            )
        )

        history = (
            self.client.wait_for_prompt(
                prompt_id,
                timeout=7200,
            )
        )

        outputs = (
            self.client.find_video_outputs(
                history
            )
        )

        if not outputs:
            raise RuntimeError(
                f"No H3 video returned for "
                f"{shot['shot_id']}"
            )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        destination = (
            output_dir
            / f"{shot['shot_id']}.mp4"
        )

        result = outputs[-1]

        self.client.download_file(
            filename=result[
                "filename"
            ],
            subfolder=result[
                "subfolder"
            ],
            file_type=result[
                "type"
            ],
            destination=destination,
        )

        return destination
