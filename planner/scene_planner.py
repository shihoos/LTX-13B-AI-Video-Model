import json

from pathlib import Path

from planner.qwen_loader import (
    QwenStoryModel,
)

from schemas.scene import (
    Scene,
)

from schemas.parser import (
    extract_json,
)


class ScenePlanner:

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
            / "scene_plan.txt"
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

    def create_scene_plan(
        self,
        story: str,
        characters: list,
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

        prompt = self._replace(
            template,
            story=story,
            characters=json.dumps(
                character_data,
                indent=2,
                ensure_ascii=False,
            ),
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a cinematic scene "
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
            messages
        )

        data = extract_json(
            response
        )

        scenes = []

        for item in data.get(
            "scenes",
            [],
        ):

            scenes.append(
                Scene(
                    scene_id=(
                        f"scene_"
                        f"{len(scenes) + 1:03d}"
                    ),

                    order=(
                        len(scenes) + 1
                    ),

                    location=item.get(
                        "location",
                        "",
                    ),

                    time_of_day=item.get(
                        "time_of_day",
                        "",
                    ),

                    description=item.get(
                        "description",
                        "",
                    ),

                    characters=item.get(
                        "characters",
                        [],
                    ),

                    story_summary=item.get(
                        "story_summary",
                        "",
                    ),

                    continuity_notes=item.get(
                        "continuity_notes",
                        "",
                    ),

                    shot_ids=[],
                )
            )

        return scenes

    def unload(self):

        self.model.unload()
