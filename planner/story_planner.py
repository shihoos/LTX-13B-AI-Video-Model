from __future__ import annotations

from pathlib import Path

from planner.config import (
    AI_STORY_MODE,
    CREATE_STORY_PROMPT_PATH,
    EXPAND_USER_STORY_MODE,
    PRESERVE_STORY_PROMPT_PATH,
    PRESERVE_USER_STORY_MODE,
    QWEN_PRESERVE_STORY_TEMPERATURE,
    QWEN_STORY_TEMPERATURE,
)
from planner.qwen_loader import QwenStoryModel


class StoryPlanner:
    """
    Creates or transforms the story that becomes the source of truth
    for the rest of the production planning pipeline.

    Important:
    The story itself is plain text.

    CharacterPlanner, ScenePlanner and ShotPlanner consume this text.
    Do NOT try to force the story through extract_json().
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

    @staticmethod
    def _read(
        path: Path,
    ) -> str:
        return path.read_text(
            encoding="utf-8"
        )

    def _generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
    ) -> str:
        response = self.model.generate(
            [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            temperature=temperature,
        )

        result = (
            response
            or ""
        ).strip()

        if not result:
            raise RuntimeError(
                "Qwen story planner returned an empty story."
            )

        return result

    def plan(
        self,
        mode: str,
        user_input: str,
    ) -> str:
        if not user_input or not user_input.strip():
            raise ValueError(
                "Story input cannot be empty."
            )

        if mode == AI_STORY_MODE:
            template = self._read(
                CREATE_STORY_PROMPT_PATH
            )

            prompt = template.replace(
                "{user_request}",
                user_input.strip(),
            )

            return self._generate(
                system_prompt=(
                    "You are the story director for a cinematic "
                    "AI video production system. "
                    "Create a coherent production-ready story. "
                    "Return plain story text only. "
                    "Do not use markdown fences."
                ),
                user_prompt=prompt,
                temperature=QWEN_STORY_TEMPERATURE,
            )

        if mode == PRESERVE_USER_STORY_MODE:
            template = self._read(
                PRESERVE_STORY_PROMPT_PATH
            )

            prompt = template.replace(
                "{user_story}",
                user_input.strip(),
            )

            return self._generate(
                system_prompt=(
                    "You are a story-preservation director. "
                    "Preserve the user's narrative faithfully. "
                    "Return the preserved/structured story as plain "
                    "text without markdown fences."
                ),
                user_prompt=prompt,
                temperature=QWEN_PRESERVE_STORY_TEMPERATURE,
            )

        if mode == EXPAND_USER_STORY_MODE:
            prompt = (
                "Expand the user's story into a complete cinematic "
                "production story.\n\n"
                "Preserve all existing named characters, major events, "
                "relationships and intended ending.\n"
                "Do not replace the original story with a different story.\n"
                "Add only details that improve cinematic storytelling, "
                "scene transitions, character motivation and visual continuity.\n\n"
                "USER STORY:\n"
                f"{user_input.strip()}"
            )

            return self._generate(
                system_prompt=(
                    "You are a cinematic story expansion director. "
                    "Expand the supplied story while preserving its "
                    "core narrative and explicit character names. "
                    "Return plain story text only."
                ),
                user_prompt=prompt,
                temperature=QWEN_STORY_TEMPERATURE,
            )

        raise ValueError(
            f"Unsupported story mode: {mode}"
        )

    def unload(self):
        self.model.unload()
