from __future__ import annotations

import json
from pathlib import Path

from pipeline.identity_continuity import (
    IdentityContinuity,
)
from pipeline.reference_manager import (
    ReferenceManager,
)
from planner.config import (
    H3_FRAMES_PER_SHOT,
    H3_HEIGHT,
    H3_STEPS,
    H3_WIDTH,
    H3_FPS,
    QWEN_SHOT_PLAN_TEMPERATURE,
)
from planner.qwen_loader import QwenStoryModel
from schemas.parser import extract_json
from schemas.shot import Shot


class ShotPlanner:

    def __init__(
        self,
        model=None,
    ):
        self.model = (
            model
            if model is not None
            else QwenStoryModel()
        )

        self.project_root = (
            Path(__file__)
            .resolve()
            .parents[1]
        )

        self.references = (
            ReferenceManager(
                self.project_root
            )
        )

    def _read_prompt(self) -> str:
        return (
            self.project_root
            / "prompts"
            / "qwen"
            / "shot_plan.txt"
        ).read_text(
            encoding="utf-8"
        )

    @staticmethod
    def _names(
        values,
    ) -> list[str]:
        if not isinstance(
            values,
            list,
        ):
            return []

        return [
            str(value).strip()
            for value in values
            if str(value).strip()
        ]

    def create_shot_plan(
        self,
        story: str,
        characters: list,
        scene,
        continuity_context: str = "",
        shot_start_index: int = 1,
    ) -> list[Shot]:

        character_data = [
            character.to_dict()
            if hasattr(
                character,
                "to_dict",
            )
            else character
            for character in characters
        ]

        scene_data = (
            scene.to_dict()
            if hasattr(
                scene,
                "to_dict",
            )
            else scene
        )

        prompt = self._read_prompt().format(
            story=story,
            characters=json.dumps(
                character_data,
                indent=2,
                ensure_ascii=False,
            ),
            scene=json.dumps(
                scene_data,
                indent=2,
                ensure_ascii=False,
            ),
            continuity_context=(
                continuity_context
            ),
        )

        response = self.model.generate(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a cinematic shot "
                        "planning system for "
                        "MiniMax H3 Ref2VA. "
                        "Return JSON only."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=(
                QWEN_SHOT_PLAN_TEMPERATURE
            ),
        )

        data = extract_json(
            response
        )

        by_name = {
            character.name.lower():
                character
            for character in characters
        }

        shots = []

        for item in data.get(
            "shots",
            [],
        ):
            shot_characters = (
                self._names(
                    item.get(
                        "characters",
                        [],
                    )
                )
            )

            speaking_characters = (
                self._names(
                    item.get(
                        "speaking_characters",
                        [],
                    )
                )
            )

            selected_characters = [
                by_name[name.lower()]
                for name in shot_characters
                if name.lower()
                in by_name
            ]

            image_paths = []
            video_paths = []
            audio_paths = []
            character_image_bindings = {}

            for character in (
                selected_characters
            ):
                images = (
                    character
                    .normalized_reference_paths()
                )

                videos = (
                    character
                    .normalized_video_paths()
                )

                audios = (
                    character
                    .normalized_audio_paths()
                )

                for path in images:
                    if (
                        path
                        not in image_paths
                        and len(image_paths) < 9
                    ):
                        image_paths.append(
                            path
                        )

                for path in videos:
                    if (
                        path
                        not in video_paths
                        and len(video_paths) < 3
                    ):
                        video_paths.append(
                            path
                        )

                for path in audios:
                    if (
                        path
                        not in audio_paths
                        and len(audio_paths) < 3
                    ):
                        audio_paths.append(
                            path
                        )

                character_image_bindings[
                    character.name
                ] = images

            identity_locks = (
                IdentityContinuity.build_locks(
                    characters=selected_characters,
                    shot_characters=shot_characters,
                )
            )

            reference_bindings = (
                IdentityContinuity
                .build_reference_bindings(
                    image_paths=image_paths,
                    character_bindings=(
                        character_image_bindings
                    ),
                )
            )

            visual_prompt, negative_prompt = (
                IdentityContinuity.merge(
                    visual_prompt=str(
                        item.get(
                            "visual_prompt",
                            "",
                        )
                    ),
                    locks=identity_locks,
                    bindings=reference_bindings,
                    negative_prompt=str(
                        item.get(
                            "negative_prompt",
                            "",
                        )
                    ),
                )
            )

            audio_by_character = {
                character.name:
                    character
                    .normalized_audio_paths()
                for character
                in selected_characters
            }

            video_by_character = {
                character.name:
                    character
                    .normalized_video_paths()
                for character
                in selected_characters
            }

            primary_voice = None

            for speaker in (
                speaking_characters
            ):
                character = by_name.get(
                    speaker.lower()
                )

                if character is None:
                    continue

                audio = (
                    character
                    .normalized_audio_paths()
                )

                if audio:
                    primary_voice = audio[0]
                    break

            shot = Shot(
                shot_id=(
                    f"shot_"
                    f"{shot_start_index + len(shots):03d}"
                ),
                scene_id=scene_data.get(
                    "scene_id",
                    "",
                ),
                order=len(shots) + 1,
                duration_seconds=float(
                    item.get(
                        "duration_seconds",
                        5.0,
                    )
                ),
                characters=shot_characters,
                location=str(
                    item.get(
                        "location",
                        scene_data.get(
                            "location",
                            "",
                        ),
                    )
                ),
                action=str(
                    item.get(
                        "action",
                        "",
                    )
                ),
                camera_shot=str(
                    item.get(
                        "camera_shot",
                        "",
                    )
                ),
                camera_movement=str(
                    item.get(
                        "camera_movement",
                        "",
                    )
                ),
                lighting=str(
                    item.get(
                        "lighting",
                        "",
                    )
                ),
                mood=str(
                    item.get(
                        "mood",
                        "",
                    )
                ),
                visual_prompt=visual_prompt,
                negative_prompt=negative_prompt,
                continuity_notes=str(
                    item.get(
                        "continuity_notes",
                        "",
                    )
                ),
                seed=item.get(
                    "seed"
                ),
                reference_images=image_paths,
                reference_videos=video_paths,
                reference_audio=primary_voice,
                reference_audio_paths=audio_paths,
                reference_audio_by_character=(
                    audio_by_character
                ),
                reference_video_by_character=(
                    video_by_character
                ),
                speaking_characters=(
                    speaking_characters
                ),
                speech_text=str(
                    item.get(
                        "speech_text",
                        "",
                    )
                ),
                reference_bindings=(
                    reference_bindings
                ),
                identity_locks=(
                    identity_locks
                ),
                width=H3_WIDTH,
                height=H3_HEIGHT,
                fps=H3_FPS,
                frames_per_shot=(
                    H3_FRAMES_PER_SHOT
                ),
                steps=H3_STEPS,
            )

            shots.append(
                shot
            )

        return shots

    def unload(self):
        self.model.unload()
