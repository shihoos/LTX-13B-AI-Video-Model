from __future__ import annotations

import re
from pathlib import Path

from pipeline.reference_manager import (
    ReferenceManager,
)
from planner.config import (
    QWEN_CHARACTER_DETECTION_TEMPERATURE,
)
from planner.qwen_loader import (
    QwenStoryModel,
)
from schemas.parser import (
    extract_json,
)


class CharacterDetector:

    def __init__(
        self,
        model=None,
    ):
        self.model = (
            model
            if model is not None
            else QwenStoryModel()
        )

        self.project_root = (
            Path(__file__)
            .resolve()
            .parents[1]
        )

        self.references = (
            ReferenceManager(
                self.project_root
            )
        )

    @staticmethod
    def _normalise(
        value: str,
    ) -> str:
        return re.sub(
            r"[^a-z0-9]+",
            " ",
            value.lower(),
        ).strip()

    def _explicit_asset_names(
        self,
        text: str,
    ) -> list[str]:
        if not text:
            return []

        result = []

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
                result.append(
                    asset_name
                )

        return result

    def detect(
        self,
        story: str,
        original_request: str = "",
    ) -> list[str]:

        if not story or not story.strip():
            return []

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
                    "Identify the important named characters "
                    "that require a consistent visual identity.\n\n"
                    "Preserve explicit character names exactly.\n"
                    "Do not rename, translate or replace names.\n"
                    "Do not include generic crowds or temporary extras.\n\n"
                    "Return exactly:\n"
                    "{\n"
                    '  "characters": ["Character Name"]\n'
                    "}\n\n"
                    "STORY:\n"
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
            for name in self._explicit_asset_names(
                text
            ):
                if name.lower() not in {
                    item.lower()
                    for item in explicit_names
                }:
                    explicit_names.append(
                        name
                    )

        result = []
        seen = set()

        for name in explicit_names:
            key = name.lower()

            if key not in seen:
                seen.add(key)
                result.append(name)

        explicit_normalised = {
            self._normalise(name)
            for name in explicit_names
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

            normalised = self._normalise(
                name
            )

            if normalised in explicit_normalised:
                continue

            seen.add(key)
            result.append(name)

        return result

    def unload(self):
        self.model.unload()
