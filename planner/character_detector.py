import re

from pipeline.reference_manager import (
    ReferenceManager,
)

from planner.qwen_loader import (
    QwenStoryModel,
)

from schemas.parser import (
    extract_json,
)


class CharacterDetector:

    """
    Detect important named characters from the story.

    Explicit character names that match a provided
    character asset are preserved exactly.
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

        self.references = (
            ReferenceManager()
        )

    def _explicit_asset_names(
        self,
        story: str,
    ) -> list[str]:

        names = []

        for name in (
            self.references.character_asset_names()
        ):

            pattern = (
                r"(?<!\w)"
                + re.escape(name)
                + r"(?!\w)"
            )

            if re.search(
                pattern,
                story,
                flags=re.IGNORECASE,
            ):

                names.append(
                    name
                )

        return names

    def detect(
        self,
        story: str,
        original_request: str = "",
    ) -> list[str]:

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

                    "Preserve character names exactly "
                    "as written in the story.\n\n"

                    "Do not rename, translate, expand, "
                    "or replace explicit character names.\n\n"

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

        names = data.get(
            "characters",
            [],
        )

        cleaned_names = []
        seen = set()

        explicit_names = (
            self._explicit_asset_names(
                story
            )
        )

        # Explicit asset-backed names always win.
        for name in explicit_names:

            key = name.lower()

            if key in seen:
                continue

            seen.add(
                key
            )

            cleaned_names.append(
                name
            )

        # Add Qwen-detected characters, but do not
        # add a second name when an explicit asset-backed
        # character is already present.
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

            # If an explicit asset name is present in the
            # story, do not add a Qwen-generated alternative
            # that could represent the same character.
            if explicit_names:

                normalized_name = (
                    re.sub(
                        r"[^a-z0-9]+",
                        " ",
                        name.lower(),
                    ).strip()
                )

                if any(
                    normalized_name
                    == re.sub(
                        r"[^a-z0-9]+",
                        " ",
                        explicit_name.lower(),
                    ).strip()
                    for explicit_name
                    in explicit_names
                ):

                    continue

            seen.add(
                key
            )

            cleaned_names.append(
                name
            )

        return cleaned_names

    def unload(self):

        self.model.unload()
