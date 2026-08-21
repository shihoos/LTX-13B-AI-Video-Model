from pathlib import Path

from planner.config import (
    AI_STORY_MODE,
    EXPAND_USER_STORY_MODE,
    PRESERVE_USER_STORY_MODE,
    QWEN_PRESERVE_STORY_TEMPERATURE,
    QWEN_STORY_TEMPERATURE,
)

from planner.qwen_loader import (
    QwenStoryModel,
)


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

    def _read_prompt(
        self,
        filename: str,
    ) -> str:

        project_root = (
            Path(__file__)
            .resolve()
            .parents[1]
        )

        prompt_path = (
            project_root
            / "prompts"
            / "qwen"
            / filename
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

    def create_story(
        self,
        user_request: str,
    ) -> str:

        system_prompt = (
            self._read_prompt(
                "system.txt"
            )
        )

        template = (
            self._read_prompt(
                "create_story.txt"
            )
        )

        prompt = self._replace(
            template,
            user_request=user_request,
        )

        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        return self.model.generate(
            messages,
            temperature=(
                QWEN_STORY_TEMPERATURE
            ),
        )

    def preserve_story(
        self,
        user_story: str,
    ) -> str:

        system_prompt = (
            self._read_prompt(
                "system.txt"
            )
        )

        template = (
            self._read_prompt(
                "preserve_story.txt"
            )
        )

        prompt = self._replace(
            template,
            user_story=user_story,
        )

        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        return self.model.generate(
            messages,
            temperature=(
                QWEN_PRESERVE_STORY_TEMPERATURE
            ),
        )

    def expand_story(
        self,
        user_idea: str,
    ) -> str:

        request = (
            "Expand the following idea into "
            "a complete cinematic story while "
            "preserving its core concept:\n\n"
            f"{user_idea}"
        )

        return self.create_story(
            request
        )

    def plan(
        self,
        mode: str,
        user_input: str,
    ) -> str:

        if mode == AI_STORY_MODE:

            return self.create_story(
                user_input
            )

        if (
            mode
            == PRESERVE_USER_STORY_MODE
        ):

            return self.preserve_story(
                user_input
            )

        if (
            mode
            == EXPAND_USER_STORY_MODE
        ):

            return self.expand_story(
                user_input
            )

        raise ValueError(
            f"Unsupported story mode: {mode}"
        )

    def unload(self):

        self.model.unload()
