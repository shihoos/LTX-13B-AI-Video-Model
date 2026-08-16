import json

from planner.qwen_loader import (
    QwenStoryModel,
)

from pipeline.reference_manager import (
    ReferenceManager,
)


class CharacterPlanner:

    def __init__(self):

        self.model = QwenStoryModel()

        self.references = (
            ReferenceManager()
        )

    def create_character_plan(
        self,
        story: str,
        character_names: list,
    ) -> str:

        from pathlib import Path

        project_root = (
            Path(__file__)
            .resolve()
            .parents[1]
        )

        prompt_path = (
            project_root
            / "prompts"
            / "qwen"
            / "character_plan.txt"
        )

        template = prompt_path.read_text(
            encoding="utf-8"
        )

        reference_data = {}

        for name in character_names:

            reference_data[name] = (
                self.references
                .get_character_source(name)
            )

        prompt = template.format(
            story=story,
            references=json.dumps(
                reference_data,
                indent=2,
            ),
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "Return a structured character "
                    "production plan."
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
