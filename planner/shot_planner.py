from __future__ import annotations

import json
from pathlib import Path

from pipeline.identity_continuity import (
    IdentityContinuity,
)

from planner.config import (
    H3_FPS,
    H3_FRAMES_PER_SHOT,
    H3_HEIGHT,
    H3_STEPS,
    H3_WIDTH,
    QWEN_SHOT_PLAN_TEMPERATURE,
)

from planner.qwen_loader import (
    QwenStoryModel,
)

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

    def _read_prompt(self):
        return (
            self.project_root
            / "prompts"
            / "qwen"
            / "shot_plan.txt"
        ).read_text(
            encoding="utf-8"
        )

    @staticmethod
    def _names(values):

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
        story,
        characters,
        scene,
        continuity_context="",
        shot_start_index=1,
    ):

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

        prompt = (
            self._read_prompt()
            .replace(
                "{story}",
                story,
            )
            .replace(
                "{characters}",
                json.dumps(
                    character_data,
                    indent=2,
                    ensure_ascii=False,
                ),
            )
            .replace(
                "{scene}",
                json.dumps(
                    scene_data,
                    indent=2,
                    ensure_ascii=False,
                ),
            )
            .replace(
                "{continuity_context}",
                continuity_context,
            )
        )

        response = self.model.generate(
            [
                {
                    "role": "system",
                    "content": (
                        "You are the shot director "
                        "for MiniMax H3 Ref2VA. "
                        "Return valid JSON only."
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

            selected = [
                by_name[name.lower()]
                for name in shot_characters
                if name.lower()
                in by_name
            ]

            image_paths = []
            video_paths = []
            audio_paths = []

            image_bindings = {}

            for character in selected:

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
                        path not in image_paths
                        and len(image_paths) < 9
                    ):
                        image_paths.append(
                            path
                        )

                for path in videos:
                    if (
                        path not in video_paths
                        and len(video_paths) < 3
                    ):
                        video_paths.append(
                            path
                        )

                for path in audios:
                    if (
                        path not in audio_paths
                        and len(audio_paths) < 3
                    ):
                        audio_paths.append(
                            path
                        )

                image_bindings[
                    character.name
                ] = images

            locks = (
                IdentityContinuity.build_locks(
                    selected,
                    shot_characters,
                )
            )

            bindings = (
                IdentityContinuity
                .build_reference_bindings(
                    image_paths,
                    image_bindings,
                )
            )

            visual_prompt = str(
                item.get(
                    "visual_prompt",
                    "",
                )
            )

            negative_prompt = str(
                item.get(
                    "negative_prompt",
                    "",
                )
            )

            (
                visual_prompt,
                negative_prompt,
            ) = IdentityContinuity.merge(
                visual_prompt=visual_prompt,
                locks=locks,
                bindings=bindings,
                negative_prompt=negative_prompt,
            )

            primary_audio = (
                audio_paths[0]
                if audio_paths
                else None
            )

            duration = float(
                item.get(
                    "duration_seconds",
                    5.0,
                )
            )

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

                duration_seconds=duration,

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

                retention_analysis=str(
                    item.get(
                        "retention_analysis",
                        "",
                    )
                ),

                detailed_description=str(
                    item.get(
                        "detailed_description",
                        "",
                    )
                ),

                overall_soundscape=str(
                    item.get(
                        "overall_soundscape",
                        "",
                    )
                ),

                non_diegetic_music=str(
                    item.get(
                        "non_diegetic_music",
                        "",
                    )
                ),

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

                reference_audio=primary_audio,
                reference_audio_paths=audio_paths,

                reference_audio_by_character={
                    character.name:
                        character
                        .normalized_audio_paths()
                    for character in selected
                },

                reference_video_by_character={
                    character.name:
                        character
                        .normalized_video_paths()
                    for character in selected
                },

                speaking_characters=(
                    speaking_characters
                ),

                speech_text=str(
                    item.get(
                        "speech_text",
                        "",
                    )
                ),

                reference_bindings=bindings,
                identity_locks=locks,

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
