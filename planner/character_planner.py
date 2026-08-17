import json

from pathlib import Path

from planner.qwen_loader import (
    QwenStoryModel,
)

from pipeline.reference_manager import (
    ReferenceManager,
)

from schemas.character import (
    Character,
)

from schemas.parser import (
    extract_json,
)


class CharacterPlanner:

    def __init__(
        self,
        model=None,
    ):

        self.model = (
            model
            if model is not None
            else QwenStoryModel()
        )

        self.references = (
            ReferenceManager()
        )

    def create_character_plan(
        self,
        story: str,
        character_names: list,
    ) -> list:

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
                    "You are a cinematic character "
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

        characters = []

        for item in data.get(
            "characters",
            [],
        ):

            name = item.get(
                "name",
                "",
            )

            reference = reference_data.get(
                name
            )

            if reference is None:

                reference = (
                    self.references
                    .get_character_source(
                        name
                    )
                )

            characters.append(
                Character(
                    character_id=item.get(
                        "character_id"
                    ) or (
                        f"character_"
                        f"{len(characters) + 1:03d}"
                    ),

                    name=name,

                    role=item.get(
                        "role",
                        "",
                    ),

                    description=item.get(
                        "description",
                        "",
                    ),

                    personality=item.get(
                        "personality",
                        "",
                    ),

                    appearance=item.get(
                        "appearance",
                        {},
                    ),

                    clothing=item.get(
                        "clothing",
                        {},
                    ),

                    distinctive_features=item.get(
                        "distinctive_features",
                        [],
                    ),

                    reference_mode=reference.get(
                        "mode",
                        "auto",
                    ),

                    reference_path=reference.get(
                        "path"
                    ),

                    continuity_rules=item.get(
                        "continuity_rules",
                        [],
                    ),
                )
            )

        return characters

    def unload(self):

        self.model.unload()
