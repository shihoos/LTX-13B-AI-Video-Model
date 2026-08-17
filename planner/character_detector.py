import json

from planner.qwen_loader import (
    QwenStoryModel,
)

from schemas.parser import (
    extract_json,
)


class CharacterDetector:

    """
    Detects the important named characters
    from a completed story.

    This removes the need for the user to manually
    provide character_names.
    """

    def __init__(
        self,
        model=None,
    ):

        self.model = (
            model
            if model is not None
            else QwenStoryModel()
        )

    def detect(
        self,
        story: str,
    ) -> list:

        messages = [
            {
                "role": "system",
                "content": (
                    "You analyze cinematic stories. "
                    "Return only valid JSON."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Identify the important named "
                    "characters that require a "
                    "consistent visual identity.\n\n"

                    "Do not include unnamed crowds, "
                    "background people, generic roles, "
                    "or temporary extras unless they "
                    "are important recurring characters.\n\n"

                    "Return exactly this structure:\n\n"

                    "{\n"
                    '  "characters": [\n'
                    '    "Character Name"\n'
                    "  ]\n"
                    "}\n\n"

                    "Story:\n"
                    f"{story}"
                ),
            },
        ]

        response = self.model.generate(
            messages=messages,
            temperature=0.2,
        )

        data = extract_json(
            response
        )

        names = (
            data.get(
                "characters",
                [],
            )
        )

        cleaned_names = []
        seen = set()

        for name in names:

            if not isinstance(
                name,
                str,
            ):
                continue

            name = name.strip()

            if not name:
                continue

            key = name.lower()

            if key in seen:
                continue

            seen.add(key)

            cleaned_names.append(
                name
            )

        return cleaned_names

    def unload(self):

        self.model.unload()
