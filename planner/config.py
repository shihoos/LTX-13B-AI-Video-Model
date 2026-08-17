import os

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# QWEN MODEL
# ============================================================

QWEN_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"

# Optional exact path.
#
# In Kaggle, you can set this to the directory containing:
#
# config.json
# tokenizer.json
# tokenizer_config.json
# model weights / safetensors files
#
# Example:
#
# /kaggle/input/qwen3-4b-instruct-2507
#
# If not provided, qwen_loader.py will automatically search
# inside /kaggle/input for a valid Qwen model directory.

QWEN_LOCAL_PATH_ENV = (
    os.environ.get(
        "QWEN_LOCAL_PATH",
        "",
    ).strip()
)

QWEN_LOCAL_PATH = (
    Path(QWEN_LOCAL_PATH_ENV)
    if QWEN_LOCAL_PATH_ENV
    else None
)

KAGGLE_INPUT_DIR = Path(
    "/kaggle/input"
)

QWEN_MAX_NEW_TOKENS = 4096
QWEN_TEMPERATURE = 0.7
QWEN_TOP_P = 0.8


# ============================================================
# PROMPTS
# ============================================================

QWEN_PROMPTS_DIR = (
    PROJECT_ROOT
    / "prompts"
    / "qwen"
)

SYSTEM_PROMPT_PATH = (
    QWEN_PROMPTS_DIR
    / "system.txt"
)

CREATE_STORY_PROMPT_PATH = (
    QWEN_PROMPTS_DIR
    / "create_story.txt"
)

PRESERVE_STORY_PROMPT_PATH = (
    QWEN_PROMPTS_DIR
    / "preserve_story.txt"
)

CHARACTER_PLAN_PROMPT_PATH = (
    QWEN_PROMPTS_DIR
    / "character_plan.txt"
)

SCENE_PLAN_PROMPT_PATH = (
    QWEN_PROMPTS_DIR
    / "scene_plan.txt"
)

SHOT_PLAN_PROMPT_PATH = (
    QWEN_PROMPTS_DIR
    / "shot_plan.txt"
)


# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

DATA_DIR = (
    PROJECT_ROOT
    / "data"
)

STORIES_DIR = (
    DATA_DIR
    / "stories"
)

CHARACTERS_DIR = (
    DATA_DIR
    / "characters"
)

SCENES_DIR = (
    DATA_DIR
    / "scenes"
)

SHOTS_DIR = (
    DATA_DIR
    / "shots"
)


# ============================================================
# STORY MODES
# ============================================================

AI_STORY_MODE = "ai_story"

PRESERVE_USER_STORY_MODE = (
    "preserve_user_story"
)

EXPAND_USER_STORY_MODE = (
    "expand_user_story"
)

VALID_STORY_MODES = {
    AI_STORY_MODE,
    PRESERVE_USER_STORY_MODE,
    EXPAND_USER_STORY_MODE,
}


def ensure_directories():

    directories = [
        STORIES_DIR,
        CHARACTERS_DIR,
        SCENES_DIR,
        SHOTS_DIR,
    ]

    for directory in directories:

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )
