from __future__ import annotations

import json
from pathlib import Path

from pipeline.reference_manager import ReferenceManager
from planner.config import (
    QWEN_CHARACTER_PLAN_TEMPERATURE,
)
from planner.qwen_loader import QwenStoryModel
from schemas.character import Character
from schemas.parser import extract_json


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

    def _read_prompt(self) -> str:
        return (
            self.project_root
            / "prompts"
            / "qwen"
            / "character_plan.txt"
        ).read_text(
            encoding="utf-8"
        )

    def create_character_plan(
        self,
        story: str,
        character_names: list,
    ) -> list:
        reference_data = {
            name: (
                self.references
                .get_character_source(
                    name
                )
            )
            for name in character_names
        }

        prompt = self._read_prompt().format(
            story=story,
            references=json.dumps(
                reference_data,
                indent=2,
                ensure_ascii=False,
            ),
        )

        response = self.model.generate(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a production-grade "
                        "cinematic character planner. "
                        "Return valid JSON only."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=(
                QWEN_CHARACTER_PLAN_TEMPERATURE
            ),
        )

        data = extract_json(
            response
        )

        characters = []

        reference_by_name = {
            name.lower(): source
            for name, source
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

            source = reference_by_name.get(
                name.lower()
            )

            if source is None:
                source = (
                    self.references
                    .get_character_source(
                        name
                    )
                )

            character = Character(
                character_id=(
                    str(
                        item.get(
                            "character_id",
                            "",
                        )
                    ).strip()
                    or (
                        f"character_"
                        f"{len(characters) + 1:03d}"
                    )
                ),
                name=name,
                role=str(
                    item.get(
                        "role",
                        "",
                    )
                ),
                description=str(
                    item.get(
                        "description",
                        "",
                    )
                ),
                personality=str(
                    item.get(
                        "personality",
                        "",
                    )
                ),
                appearance=(
                    item.get(
                        "appearance",
                        {},
                    )
                    or {}
                ),
                clothing=(
                    item.get(
                        "clothing",
                        {},
                    )
                    or {}
                ),
                distinctive_features=(
                    item.get(
                        "distinctive_features",
                        [],
                    )
                    or []
                ),
                character_state=(
                    item.get(
                        "character_state",
                        {},
                    )
                    or {}
                ),
                continuity_rules=(
                    item.get(
                        "continuity_rules",
                        [],
                    )
                    or []
                ),
                reference_mode=(
                    source.get(
                        "mode",
                        "missing",
                    )
                ),
                reference_paths=list(
                    source.get(
                        "reference_paths",
                        [],
                    )
                ),
                reference_video_paths=list(
                    source.get(
                        "reference_video_paths",
                        [],
                    )
                ),
                reference_audio_paths=list(
                    source.get(
                        "reference_audio_paths",
                        [],
                    )
                ),
                reference_path=source.get(
                    "path"
                ),
                reference_video_path=source.get(
                    "reference_video_path"
                ),
                reference_audio_path=source.get(
                    "reference_audio_path"
                ),
            )

            character.build_identity_profile()
            characters.append(
                character
            )

        self.references.resolve_characters(
            characters
        )

        return characters

    def unload(self):
        self.model.unload()
