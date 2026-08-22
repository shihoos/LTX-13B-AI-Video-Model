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
            self.client.get_object_info()
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

        source_path = Path(
            source
        )

        if not source_path.is_file():
            raise FileNotFoundError(
                f"Reference file does not exist: "
                f"{source_path}"
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

        for binding in shot.get(
            "reference_bindings",
            [],
        ):
            if str(binding).strip():
                parts.append(
                    str(binding)
                )

        for lock in shot.get(
            "identity_locks",
            [],
        ):
            if str(lock).strip():
                parts.append(
                    str(lock)
                )

        visual = str(
            shot.get(
                "visual_prompt",
                "",
            )
            or ""
        ).strip()

        if visual:
            parts.append(
                visual
            )

        action = str(
            shot.get(
                "action",
                "",
            )
            or ""
        ).strip()

        if action:
            parts.append(
                "ACTION: "
                + action
            )

        camera = " ".join(
            value
            for value in [
                str(
                    shot.get(
                        "camera_shot",
                        "",
                    )
                    or ""
                ).strip(),
                str(
                    shot.get(
                        "camera_movement",
                        "",
                    )
                    or ""
                ).strip(),
            ]
            if value
        )

        if camera:
            parts.append(
                "CAMERA: "
                + camera
            )

        lighting = str(
            shot.get(
                "lighting",
                "",
            )
            or ""
        ).strip()

        if lighting:
            parts.append(
                "LIGHTING: "
                + lighting
            )

        mood = str(
            shot.get(
                "mood",
                "",
            )
            or ""
        ).strip()

        if mood:
            parts.append(
                "MOOD: "
                + mood
            )

        continuity = str(
            shot.get(
                "continuity_notes",
                "",
            )
            or ""
        ).strip()

        if continuity:
            parts.append(
                "CONTINUITY: "
                + continuity
            )

        dialogue = str(
            shot.get(
                "speech_text",
                "",
            )
            or ""
        ).strip()

        if dialogue:
            parts.append(
                "DIALOGUE: "
                + dialogue
            )

        negative = str(
            shot.get(
                "negative_prompt",
                "",
            )
            or ""
        ).strip()

        if negative:
            parts.append(
                "NEGATIVE: "
                + negative
            )

        return "\n".join(
            parts
        )

    def _copy_references(
        self,
        shot: dict,
    ) -> tuple[list[str], list[str], list[str]]:

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

        primary = shot.get(
            "reference_audio"
        )

        if (
            primary
            and primary not in audio_paths
        ):
            audio_paths.insert(
                0,
                primary,
            )

        audio_files = [
            self._copy(
                path,
                "reference_audio",
            )
            for path in audio_paths[:3]
        ]

        return (
            image_files,
            video_files,
            audio_files,
        )

    @staticmethod
    def _seed(
        shot: dict,
    ) -> int:

        value = shot.get(
            "seed"
        )

        if value is None:
            return 0

        return int(value)

    def _download_result(
        self,
        history: dict,
        destination: Path,
        label: str,
    ) -> Path:

        outputs = (
            self.client.find_video_outputs(
                history
            )
        )

        if not outputs:
            raise RuntimeError(
                f"H3 {label} produced no video."
            )

        result = outputs[-1]

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        return self.client.download_file(
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

    def execute_native_ref2va(
        self,
        shot: dict,
        output_dir: Path,
    ) -> Path:

        (
            image_files,
            video_files,
            audio_files,
        ) = self._copy_references(
            shot
        )

        workflow = (
            self.builder
            .build_native_ref2va(
                client=self.client,
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
                seed=self._seed(
                    shot
                ),
                output_prefix=(
                    "h3/ref2va/"
                    + str(
                        shot[
                            "shot_id"
                        ]
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

        destination = (
            output_dir
            / f"{shot['shot_id']}.mp4"
        )

        return self._download_result(
            history,
            destination,
            "Ref2VA",
        )

    def execute_hardmode_chained(
        self,
        shots: list[dict],
        output_dir: Path,
    ) -> Path:

        if len(shots) < 2:
            raise ValueError(
                "Hard Mode Chained requires "
                "at least two shots."
            )

        first_shot = shots[0]

        (
            image_files,
            video_files,
            audio_files,
        ) = self._copy_references(
            first_shot
        )

        workflow = (
            self.builder
            .build_hardmode_chained(
                client=self.client,
                shots=shots,
                image_files=image_files,
                video_files=video_files,
                audio_files=audio_files,
                width=int(
                    first_shot.get(
                        "width",
                        960,
                    )
                ),
                height=int(
                    first_shot.get(
                        "height",
                        544,
                    )
                ),
                frames=int(
                    first_shot.get(
                        "frames_per_shot",
                        124,
                    )
                ),
                steps=int(
                    first_shot.get(
                        "steps",
                        14,
                    )
                ),
                seed=self._seed(
                    first_shot
                ),
                output_prefix=(
                    "h3/chained/"
                    + str(
                        first_shot[
                            "scene_id"
                        ]
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
                timeout=14400,
            )
        )

        destination = (
            output_dir
            / f"{first_shot['scene_id']}_master.mp4"
        )

        return self._download_result(
            history,
            destination,
            "Hard Mode Chained",
        )
