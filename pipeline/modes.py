from enum import Enum


class StoryMode(str, Enum):

    AI_STORY = "ai_story"

    PRESERVE_USER_STORY = (
        "preserve_user_story"
    )

    EXPAND_USER_STORY = (
        "expand_user_story"
    )


class ReferenceMode(str, Enum):

    AUTO = "auto"

    PROVIDED = "provided"

    TEXT_ONLY = "text_only"
