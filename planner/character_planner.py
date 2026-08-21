import json

from pathlib import Path

from execution.comfy_client import (
    ComfyClient,
)

from execution.reference_image_generator import (
    ReferenceImageGenerator,
)

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
        reference_generator=None,
    ):

        self.model = (
            model
            if model is not None
            else QwenStoryModel()
        )

        self.references = (
            ReferenceManager()
        )

        if reference_generator is None:

            reference_generator = (
                ReferenceImageGenerator(
                    client=ComfyClient(
                        base_url=(
                            "http://127.0.0.1:8188"
                        )
                    )
                )
            )

        self.reference_generator = (
            reference_generator
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
            / "character_plan.txt"
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

    def create_character_plan(
        self,
        story: str,
        character_names: list,
    ) -> list:

        template = (
            self._read_prompt()
        )

        reference_data = {}

        for name in character_names:

            reference_data[name] = (
                self.references
                .get_character_source(
                    name
                )
            )

        prompt = self._replace(
            template,
            story=story,
            references=json.dumps(
                reference_data,
                indent=2,
                ensure_ascii=False,
            ),
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a cinematic "
                    "character planning system. "
                    "Return only valid JSON."
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

        references_by_name = {
            name.lower(): reference
            for name, reference
            in reference_data.items()
        }

        for item in data.get(
            "characters",
            [],
        ):

            name = str(
                item.get(
                    "name",
                    "",
                )
            ).strip()

            if not name:

                continue

            reference = (
                references_by_name.get(
                    name.lower()
                )
            )

            if reference is None:

                reference = (
                    self.references
                    .get_character_source(
                        name
                    )
                )

            description = item.get(
                "description",
                "",
            )

            personality = item.get(
                "personality",
                "",
            )

            appearance = item.get(
                "appearance",
                {},
            )

            clothing = item.get(
                "clothing",
                {},
            )

            distinctive_features = (
                item.get(
                    "distinctive_features",
                    [],
                )
            )

            if (
                reference.get("mode")
                == "missing"
            ):

                generated_path = (
                    self.reference_generator
                    .generate(
                        character_name=name,
                        description=(
                            description
                        ),
                        personality=(
                            personality
                        ),
                        appearance=(
                            appearance
                        ),
                        clothing=(
                            clothing
                        ),
                        distinctive_features=(
                            distinctive_features
                        ),
                    )
                )

                reference = {
                    "mode": "generated",
                    "path": str(
                        generated_path
                    ),
                    "message": (
                        "Generated and stored "
                        "a persistent character "
                        "reference."
                    ),
                }

            characters.append(
                Character(
                    character_id=(
                        f"character_"
                        f"{len(characters) + 1:03d}"
                    ),

                    name=name,

                    role=item.get(
                        "role",
                        "",
                    ),

                    description=(
                        description
                    ),

                    personality=(
                        personality
                    ),

                    appearance=(
                        appearance
                    ),

                    clothing=(
                        clothing
                    ),

                    distinctive_features=(
                        distinctive_features
                    ),

                    reference_mode=(
                        reference.get(
                            "mode",
                            "missing",
                        )
                    ),

                    reference_path=(
                        reference.get(
                            "path"
                        )
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
