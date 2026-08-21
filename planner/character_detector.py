import re

from pipeline.reference_manager import (
    ReferenceManager,
)

from planner.qwen_loader import (
    QwenStoryModel,
)

from planner.config import (
    QWEN_CHARACTER_DETECTION_TEMPERATURE,
)

from schemas.parser import (
    extract_json,
)


class CharacterDetector:

    """
    Detect important named characters.

    Explicit names corresponding to provided character assets
    are authoritative when they appear in the original request
    or generated story.
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
            temperature=(
                QWEN_CHARACTER_DETECTION_TEMPERATURE
            ),
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

                if name.lower() not in {
                    item.lower()
                    for item
                    in explicit_names
                }:

                    explicit_names.append(
                        name
                    )

        result = []

        seen = set()

        for name in explicit_names:

            key = name.lower()

            if key not in seen:

                seen.add(
                    key
                )

                result.append(
                    name
                )

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

            normalized = re.sub(
                r"[^a-z0-9]+",
                " ",
                name.lower(),
            ).strip()

            if (
                normalized
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
