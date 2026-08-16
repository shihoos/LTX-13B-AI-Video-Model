import json

from pathlib import Path

from planner.qwen_loader import (
    QwenStoryModel,
)


class ScenePlanner:

    def __init__(self):

        self.model = QwenStoryModel()

    def create_scene_plan(
        self,
        story: str,
        characters: list,
    ) -> str:

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

        template = prompt_path.read_text(
            encoding="utf-8"
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

            else:

                character_data.append(
                    {
                        "name": str(
                            character
                        )
                    }
                )

        prompt = template.format(
            story=story,
            characters=json.dumps(
                character_data,
                indent=2,
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

        return self.model.generate(
            messages
        )

    def unload(self):

        self.model.unload()
