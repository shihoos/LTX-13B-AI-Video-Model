from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)


# ============================================================
# QWEN PLANNER
# ============================================================

QWEN_MODEL_ID = os.getenv(
    "QWEN_MODEL_ID",
    "Qwen/Qwen3-4B-Instruct-2507",
)

QWEN_KAGGLE_PATH = Path(
    os.getenv(
        "QWEN_KAGGLE_PATH",
        "/kaggle/input/datasets/"
        "shihoos/qwen3-4b-instruct-2507",
    )
)

QWEN_LOCAL_PATH = (
    PROJECT_ROOT
    / "models"
    / "qwen3-4b"
)

QWEN_MAX_NEW_TOKENS = int(
    os.getenv(
        "QWEN_MAX_NEW_TOKENS",
        "4096",
    )
)

QWEN_STORY_TEMPERATURE = float(
    os.getenv(
        "QWEN_STORY_TEMPERATURE",
        "0.7",
    )
)

QWEN_PRESERVE_STORY_TEMPERATURE = float(
    os.getenv(
        "QWEN_PRESERVE_STORY_TEMPERATURE",
        "0.15",
    )
)

QWEN_CHARACTER_DETECTION_TEMPERATURE = float(
    os.getenv(
        "QWEN_CHARACTER_DETECTION_TEMPERATURE",
        "0.1",
    )
)

QWEN_CHARACTER_PLAN_TEMPERATURE = float(
    os.getenv(
        "QWEN_CHARACTER_PLAN_TEMPERATURE",
        "0.2",
    )
)

QWEN_SCENE_PLAN_TEMPERATURE = float(
    os.getenv(
        "QWEN_SCENE_PLAN_TEMPERATURE",
        "0.2",
    )
)

QWEN_SHOT_PLAN_TEMPERATURE = float(
    os.getenv(
        "QWEN_SHOT_PLAN_TEMPERATURE",
        "0.2",
    )
)

QWEN_TOP_P = float(
    os.getenv(
        "QWEN_TOP_P",
        "0.8",
    )
)


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

EXPAND_STORY_PROMPT_PATH = (
    QWEN_PROMPTS_DIR
    / "expand_story.txt"
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
# DIRECTORIES
# ============================================================

DATA_DIR = PROJECT_ROOT / "data"

STORIES_DIR = DATA_DIR / "stories"
CHARACTERS_DIR = DATA_DIR / "characters"
GENERATED_CHARACTERS_DIR = (
    CHARACTERS_DIR / "generated"
)
SCENES_DIR = DATA_DIR / "scenes"
SHOTS_DIR = DATA_DIR / "shots"
PRODUCTION_DIR = DATA_DIR / "production"


# ============================================================
# H3
# ============================================================

H3_DATASET_NAME = os.getenv(
    "H3_DATASET_NAME",
    "MiniMax H3 Ref2VA Q4",
)

H3_DATASET_SLUG = os.getenv(
    "H3_DATASET_SLUG",
    "minimax-h3-ref2va-q4",
)

H3_FPS = int(
    os.getenv(
        "H3_FPS",
        "24",
    )
)

# Official H3-Base target:
# 16:9 with 768px short edge.
H3_WIDTH = int(
    os.getenv(
        "H3_WIDTH",
        "1344",
    )
)

H3_HEIGHT = int(
    os.getenv(
        "H3_HEIGHT",
        "768",
    )
)

# Safer fallback for very limited VRAM.
H3_SAFE_WIDTH = int(
    os.getenv(
        "H3_SAFE_WIDTH",
        "960",
    )
)

H3_SAFE_HEIGHT = int(
    os.getenv(
        "H3_SAFE_HEIGHT",
        "544",
    )
)

H3_FRAMES_PER_SHOT = int(
    os.getenv(
        "H3_FRAMES_PER_SHOT",
        "124",
    )
)

H3_STEPS = int(
    os.getenv(
        "H3_STEPS",
        "14",
    )
)

H3_REF_IMAGE_SIZE = os.getenv(
    "H3_REF_IMAGE_SIZE",
    "match",
)

# Use max only when explicitly enabled.
H3_IDENTITY_MAX_REFERENCES = (
    os.getenv(
        "H3_IDENTITY_MAX_REFERENCES",
        "0",
    )
    == "1"
)


# ============================================================
# H3 MODEL FILES
# ============================================================

H3_REF2VA_MODEL = os.getenv(
    "H3_REF2VA_MODEL",
    "minimax_h3_ref2va_pruned-Q4_K_M.gguf",
)

H3_TEXT_ENCODER = os.getenv(
    "H3_TEXT_ENCODER",
    "qwen3vl_32b_minimax_h3-Q4_K_M.gguf",
)

H3_VIDEO_VAE = os.getenv(
    "H3_VIDEO_VAE",
    "minimax_h3_video_vae_fp16.safetensors",
)

H3_AUDIO_VAE = os.getenv(
    "H3_AUDIO_VAE",
    "minimax_h3_audio_vae_fp32.safetensors",
)


# ============================================================
# LORA POLICY
# ============================================================

# IMPORTANT:
# There is currently no verified official H3 identity-lock
# LoRA for Ref2VA that we can safely put into this pipeline.
#
# Therefore identity_lora is OFF.
#
# Do not put a random LoRA here.
H3_IDENTITY_LORA_ENABLED = False
H3_IDENTITY_LORA_PATH = ""


# Optional H3 Turbo LoRA.
# This is acceleration, NOT identity preservation.
H3_TURBO_LORA_ENABLED = (
    os.getenv(
        "H3_TURBO_LORA_ENABLED",
        "0",
    )
    == "1"
)

H3_TURBO_LORA_NAME = os.getenv(
    "H3_TURBO_LORA_NAME",
    "",
)

H3_TURBO_LORA_STRENGTH = float(
    os.getenv(
        "H3_TURBO_LORA_STRENGTH",
        "0.7",
    )
)


# ============================================================
# OFFICIAL H3 REGENERATION
# ============================================================

H3_REGENERATE_2K_ENABLED = (
    os.getenv(
        "H3_REGENERATE_2K_ENABLED",
        "0",
    )
    == "1"
)

H3_API_BASE = os.getenv(
    "MINIMAX_API_BASE",
    "https://api.minimax.io",
)

H3_API_KEY = os.getenv(
    "MINIMAX_API_KEY",
    "",
)

H3_REGENERATE_ENDPOINT = os.getenv(
    "H3_REGENERATE_ENDPOINT",
    "/video-generation/v2/video_generation",
)

# Final delivery.
FINAL_WIDTH = int(
    os.getenv(
        "FINAL_WIDTH",
        "1280",
    )
)

FINAL_HEIGHT = int(
    os.getenv(
        "FINAL_HEIGHT",
        "720",
    )
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
    for directory in (
        STORIES_DIR,
        CHARACTERS_DIR,
        GENERATED_CHARACTERS_DIR,
        SCENES_DIR,
        SHOTS_DIR,
        PRODUCTION_DIR,
    ):
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )
