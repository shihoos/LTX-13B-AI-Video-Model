from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

QWEN_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
QWEN_KAGGLE_PATH = (
    Path("/kaggle/input")
    / "datasets"
    / "shihoos"
    / "qwen3-4b-instruct-2507"
)
QWEN_LOCAL_PATH = PROJECT_ROOT / "models" / "qwen3-4b"

QWEN_MAX_NEW_TOKENS = 4096
QWEN_STORY_TEMPERATURE = 0.7
QWEN_PRESERVE_STORY_TEMPERATURE = 0.2
QWEN_CHARACTER_DETECTION_TEMPERATURE = 0.1
QWEN_CHARACTER_PLAN_TEMPERATURE = 0.25
QWEN_SCENE_PLAN_TEMPERATURE = 0.25
QWEN_SHOT_PLAN_TEMPERATURE = 0.3
QWEN_TOP_P = 0.8

QWEN_PROMPTS_DIR = PROJECT_ROOT / "prompts" / "qwen"
SYSTEM_PROMPT_PATH = QWEN_PROMPTS_DIR / "system.txt"
CREATE_STORY_PROMPT_PATH = QWEN_PROMPTS_DIR / "create_story.txt"
PRESERVE_STORY_PROMPT_PATH = QWEN_PROMPTS_DIR / "preserve_story.txt"
CHARACTER_PLAN_PROMPT_PATH = QWEN_PROMPTS_DIR / "character_plan.txt"
SCENE_PLAN_PROMPT_PATH = QWEN_PROMPTS_DIR / "scene_plan.txt"
SHOT_PLAN_PROMPT_PATH = QWEN_PROMPTS_DIR / "shot_plan.txt"

DATA_DIR = PROJECT_ROOT / "data"
STORIES_DIR = DATA_DIR / "stories"
CHARACTERS_DIR = DATA_DIR / "characters"
GENERATED_CHARACTERS_DIR = CHARACTERS_DIR / "generated"
SCENES_DIR = DATA_DIR / "scenes"
SHOTS_DIR = DATA_DIR / "shots"
PRODUCTION_DIR = DATA_DIR / "production"

H3_DATASET_NAME = os.getenv(
    "H3_DATASET_NAME",
    "MiniMax H3 Ref2VA Q4",
)

H3_FPS = int(os.getenv("H3_FPS", "24"))
H3_WIDTH = int(os.getenv("H3_WIDTH", "960"))
H3_HEIGHT = int(os.getenv("H3_HEIGHT", "544"))
H3_FRAMES_PER_SHOT = int(os.getenv("H3_FRAMES_PER_SHOT", "124"))
H3_STEPS = int(os.getenv("H3_STEPS", "14"))

H3_FINAL_WIDTH = 1280
H3_FINAL_HEIGHT = 720

AI_STORY_MODE = "ai_story"
PRESERVE_USER_STORY_MODE = "preserve_user_story"
EXPAND_USER_STORY_MODE = "expand_user_story"

VALID_STORY_MODES = {
    AI_STORY_MODE,
    PRESERVE_USER_STORY_MODE,
    EXPAND_USER_STORY_MODE,
}


def ensure_directories():
    for directory in (
        STORIES_DIR,
        CHARACTERS_DIR,
        GENERATED_CHARACTERS_DIR,
        SCENES_DIR,
        SHOTS_DIR,
        PRODUCTION_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
