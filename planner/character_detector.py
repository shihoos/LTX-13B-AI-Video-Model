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
    Detect important named characters.

    Character names that already have a matching asset in
    assets/characters/ are treated as authoritative when they
    appear in the original user request or generated story.
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
        text: str,
    ) -> list[str]:

        if not text:

            return []

        names = []

        for asset_name in (
            self.references.character_asset_names()
        ):

            pattern = (
                r"(?<!\w)"
                + re.escape(asset_name)
                + r"(?!\w)"
            )

            if re.search(
                pattern,
                text,
                flags=re.IGNORECASE,
            ):

                names.append(
                    asset_name
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
                    "as written.\n\n"

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

        if not isinstance(
            names,
            list,
        ):

            names = []

        # ------------------------------------------------------------
        # Asset-backed names mentioned by the user or story are
        # authoritative. This is what preserves names such as "sz"
        # even if Qwen rewrites the story using another name.
        # ------------------------------------------------------------

        explicit_names = []

        for text in (
            original_request,
            story,
        ):

            for name in (
                self._explicit_asset_names(
                    text
                )
            ):

                key = name.lower()

                if key not in {
                    item.lower()
                    for item
                    in explicit_names
                }:

                    explicit_names.append(
                        name
                    )

        result = []
        seen = set()

        # ------------------------------------------------------------
        # Add authoritative asset-backed names first.
        # ------------------------------------------------------------

        for name in explicit_names:

            key = name.lower()

            if key in seen:
                continue

            seen.add(
                key
            )

            result.append(
                name
            )

        # ------------------------------------------------------------
        # Add Qwen-detected names that are not already represented
        # by an authoritative asset-backed name.
        # ------------------------------------------------------------

        explicit_normalized = {
            re.sub(
                r"[^a-z0-9]+",
                " ",
                name.lower(),
            ).strip()
            for name
            in explicit_names
        }

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

            normalized_name = re.sub(
                r"[^a-z0-9]+",
                " ",
                name.lower(),
            ).strip()

            if (
                normalized_name
                in explicit_normalized
            ):

                continue

            seen.add(
                key
            )

            result.append(
                name
            )

        return result

    def unload(self):

        self.model.unload()
