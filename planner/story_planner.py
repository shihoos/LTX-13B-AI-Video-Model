from __future__ import annotations

from planner.config import (
    AI_STORY_MODE,
    CREATE_STORY_PROMPT_PATH,
    EXPAND_STORY_PROMPT_PATH,
    PRESERVE_STORY_PROMPT_PATH,
    PRESERVE_USER_STORY_MODE,
    EXPAND_USER_STORY_MODE,
    QWEN_PRESERVE_STORY_TEMPERATURE,
    QWEN_STORY_TEMPERATURE,
)
from planner.qwen_loader import QwenStoryModel


class StoryPlanner:

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
    def _read(path):
        return path.read_text(
            encoding="utf-8"
        )

    def _generate(
        self,
        system_prompt,
        user_prompt,
        temperature,
    ):

        result = (
            self.model.generate(
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
            .strip()
        )

        if not result:
            raise RuntimeError(
                "Qwen story planner returned empty output."
            )

        return result

    def plan(
        self,
        mode: str,
        user_input: str,
    ) -> str:

        if not user_input.strip():
            raise ValueError(
                "Story input cannot be empty."
            )

        if mode == AI_STORY_MODE:

            prompt = self._read(
                CREATE_STORY_PROMPT_PATH
            ).replace(
                "{user_request}",
                user_input.strip(),
            )

            return self._generate(
                (
                    "You are the cinematic story "
                    "director. Create a coherent "
                    "production-ready story."
                ),
                prompt,
                QWEN_STORY_TEMPERATURE,
            )

        if mode == PRESERVE_USER_STORY_MODE:

            prompt = self._read(
                PRESERVE_STORY_PROMPT_PATH
            ).replace(
                "{user_story}",
                user_input.strip(),
            )

            return self._generate(
                (
                    "You are a strict story "
                    "preservation director. "
                    "Never change the user's "
                    "narrative."
                ),
                prompt,
                QWEN_PRESERVE_STORY_TEMPERATURE,
            )

        if mode == EXPAND_USER_STORY_MODE:

            prompt = self._read(
                EXPAND_STORY_PROMPT_PATH
            ).replace(
                "{user_story}",
                user_input.strip(),
            )

            return self._generate(
                (
                    "You are a cinematic story "
                    "expansion director. Expand "
                    "without replacing the story."
                ),
                prompt,
                QWEN_STORY_TEMPERATURE,
            )

        raise ValueError(
            f"Unsupported story mode: {mode}"
        )

    def unload(self):
        self.model.unload()
