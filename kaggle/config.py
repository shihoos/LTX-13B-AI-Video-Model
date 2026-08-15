from pathlib import Path

# ---------------------------------------------------------
# Project
# ---------------------------------------------------------

PROJECT_ROOT = Path("/kaggle/working/LTX-13B-AI-Video-Model")
COMFYUI_DIR = PROJECT_ROOT / "ComfyUI"

# ---------------------------------------------------------
# Kaggle datasets
# ---------------------------------------------------------

DATASET_ROOT = Path("/kaggle/input/datasets/shihoos")

Q4_DIR = DATASET_ROOT / "ltx13b-q4"
VAE_DIR = DATASET_ROOT / "ltx13b-vae"
T5_DIR = DATASET_ROOT / "ltx13b-t5"
ENHANCER_DIR = DATASET_ROOT / "ltx13b-enhancers"

# ---------------------------------------------------------
# Model files
# ---------------------------------------------------------

Q4_MODEL = Q4_DIR / "LTXV-13B-0.9.8-distilled-Q4_K_M.gguf"

VAE_MODEL = (
    VAE_DIR /
    "LTXV-13B-0.9.8-distilled-VAE.safetensors"
)

T5_MODEL = (
    T5_DIR /
    "t5-v1_1-xxl-encoder-Q4_K_M.gguf"
)

DETAILER_LORA = (
    ENHANCER_DIR /
    "ltxv-098-ic-lora-detailer-comfyui.safetensors"
)

SPATIAL_UPSCALER = (
    ENHANCER_DIR /
    "ltxv-spatial-upscaler-0.9.8.safetensors"
)

# ---------------------------------------------------------
# Output
# ---------------------------------------------------------

OUTPUT_DIR = PROJECT_ROOT / "output"
WORKFLOW_DIR = PROJECT_ROOT / "workflows"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
WORKFLOW_DIR.mkdir(parents=True, exist_ok=True)
