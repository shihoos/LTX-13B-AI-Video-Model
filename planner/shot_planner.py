import json

from pathlib import Path

from planner.config import (
    QWEN_SHOT_PLAN_TEMPERATURE,
)

from planner.qwen_loader import (
    QwenStoryModel,
)

from schemas.shot import (
    Shot,
)

from schemas.parser import (
    extract_json,
)


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

    def _read_prompt(self) -> str:

        project_root = (
            Path(__file__)
            .resolve()
            .parents[1]
        )

        prompt_path = (
            project_root
            / "prompts"
            / "qwen"
            / "shot_plan.txt"
        )

        return prompt_path.read_text(
            encoding="utf-8"
        )

    def _replace(
        self,
        template: str,
        **values,
    ) -> str:

        prompt = template

        for key, value in values.items():

            prompt = prompt.replace(
                "{" + key + "}",
                str(value),
            )

        return prompt

    def _character_reference_map(
        self,
        characters: list,
    ) -> dict:

        references = {}

        for character in characters:

            if hasattr(
                character,
                "name",
            ):

                name = character.name

                reference_path = (
                    character.reference_path
                )

            elif isinstance(
                character,
                dict,
            ):

                name = character.get(
                    "name",
                    "",
                )

                reference_path = (
                    character.get(
                        "reference_path"
                    )
                )

            else:

                continue

            if (
                name
                and reference_path
            ):

                references[
                    name.lower()
                ] = str(
                    reference_path
                )

        return references

    def _reference_images_for_shot(
        self,
        shot_characters: list,
        reference_map: dict,
    ) -> list:

        images = []

        for name in shot_characters:

            if not isinstance(
                name,
                str,
            ):
                continue

            path = reference_map.get(
                name.lower()
            )

            if (
                path
                and path not in images
            ):

                images.append(
                    path
                )

        return images

    def create_shot_plan(
        self,
        story: str,
        characters: list,
        scene,
        continuity_context: str = "",
        shot_start_index: int = 1,
    ) -> list:

        template = (
            self._read_prompt()
        )

        character_data = []

        for character in characters:

            if hasattr(
                character,
                "to_dict",
            ):

                character_data.append(
                    character.to_dict()
                )

            elif isinstance(
                character,
                dict,
            ):

                character_data.append(
                    character
                )

        if hasattr(
            scene,
            "to_dict",
        ):

            scene_data = (
                scene.to_dict()
            )

        else:

            scene_data = scene

        prompt = self._replace(
            template,
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

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a cinematic shot "
                    "planning system. Return only "
                    "valid JSON."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        response = self.model.generate(
            messages,
            temperature=(
                QWEN_SHOT_PLAN_TEMPERATURE
            ),
        )

        data = extract_json(
            response
        )

        reference_map = (
            self._character_reference_map(
                characters
            )
        )

        shots = []

        for item in data.get(
            "shots",
            [],
        ):

            shot_characters = (
                item.get(
                    "characters",
                    [],
                )
            )

            shot = Shot(
                shot_id=(
                    f"shot_"
                    f"{shot_start_index + len(shots):03d}"
                ),

                scene_id=(
                    scene_data.get(
                        "scene_id",
                        "",
                    )
                ),

                order=(
                    len(shots) + 1
                ),

                duration_seconds=float(
                    item.get(
                        "duration_seconds",
                        5.0,
                    )
                ),

                characters=(
                    shot_characters
                ),

                location=item.get(
                    "location",
                    scene_data.get(
                        "location",
                        "",
                    ),
                ),

                action=item.get(
                    "action",
                    "",
                ),

                camera_shot=item.get(
                    "camera_shot",
                    "",
                ),

                camera_movement=item.get(
                    "camera_movement",
                    "",
                ),

                lighting=item.get(
                    "lighting",
                    "",
                ),

                mood=item.get(
                    "mood",
                    "",
                ),

                visual_prompt=item.get(
                    "visual_prompt",
                    "",
                ),

                negative_prompt=item.get(
                    "negative_prompt",
                    "",
                ),

                previous_shot=None,

                next_shot=None,

                continuity_notes=item.get(
                    "continuity_notes",
                    "",
                ),

                seed=item.get(
                    "seed"
                ),

                reference_images=(
                    self
                    ._reference_images_for_shot(
                        shot_characters,
                        reference_map,
                    )
                ),
            )

            shots.append(
                shot
            )

        return shots

    def unload(self):

        self.model.unload()
