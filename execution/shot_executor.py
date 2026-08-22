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
        self.project_root = Path(project_root)
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
                project_root=self.project_root,
                object_info=self.object_info,
            )
        )

    def _copy(
        self,
        source: str,
        prefix: str,
    ) -> str:

        source_path = Path(source)

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
            for char in source_path.name
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
    def build_prompt(
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

        parts.extend(
            [
                "SHOT:",
                str(
                    shot.get(
                        "visual_prompt",
                        "",
                    )
                ),
                "ACTION:",
                str(
                    shot.get(
                        "action",
                        "",
                    )
                ),
                "CAMERA:",
                str(
                    shot.get(
                        "camera_shot",
                        "",
                    )
                ),
                str(
                    shot.get(
                        "camera_movement",
                        "",
                    )
                ),
                "LIGHTING:",
                str(
                    shot.get(
                        "lighting",
                        "",
                    )
                ),
                "MOOD:",
                str(
                    shot.get(
                        "mood",
                        "",
                    )
                ),
                "CONTINUITY:",
                str(
                    shot.get(
                        "continuity_notes",
                        "",
                    )
                ),
            ]
        )

        dialogue = str(
            shot.get(
                "speech_text",
                "",
            )
            or ""
        ).strip()

        if dialogue:
            parts.extend(
                [
                    "DIALOGUE:",
                    dialogue,
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
            value
            for value in parts
            if value.strip()
        )

    def execute_native_ref2va(
        self,
        shot: dict,
        output_dir: Path,
    ) -> Path:

        image_files = [
            self._copy(
                path,
                "reference_image",
            )
            for path in shot.get(
                "reference_images",
                [],
            )[:9]
        ]

        video_files = [
            self._copy(
                path,
                "reference_video",
            )
            for path in shot.get(
                "reference_videos",
                [],
            )[:3]
        ]

        audio_paths = list(
            shot.get(
                "reference_audio_paths",
                [],
            )
        )

        primary_voice = shot.get(
            "reference_audio"
        )

        if (
            primary_voice
            and primary_voice not in audio_paths
        ):
            audio_paths.insert(
                0,
                primary_voice,
            )

        audio_files = [
            self._copy(
                path,
                "reference_audio",
            )
            for path in audio_paths[:3]
        ]

        seed = shot.get(
            "seed"
        )

        if seed is None:
            seed = 0

        workflow = (
            self.builder
            .build_native_ref2va(
                prompt=self.build_prompt(
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
                    "h3/ref2va/"
                    + str(
                        shot["shot_id"]
                    )
                ),
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
                f"H3 Ref2VA produced no video: "
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
            filename=result["filename"],
            subfolder=result["subfolder"],
            file_type=result["type"],
            destination=destination,
        )

        return destination
