from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(
    os.getenv(
        "LTX_PROJECT_ROOT",
        "/kaggle/working/LTX-13B-AI-Video-Model",
    )
)

COMFYUI_DIR = (
    PROJECT_ROOT
    / "ComfyUI"
)

LOCK_FILE = (
    PROJECT_ROOT
    / "kaggle"
    / "compatibility_lock.yaml"
)


def load_lock():

    if not LOCK_FILE.exists():

        raise FileNotFoundError(
            "Compatibility lock not found:\n"
            f"{LOCK_FILE}"
        )

    try:

        import yaml

    except ImportError as error:

        raise RuntimeError(
            "PyYAML is required to load "
            "compatibility_lock.yaml."
        ) from error

    with LOCK_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:

        data = yaml.safe_load(
            file
        )

    if not isinstance(
        data,
        dict,
    ):

        raise RuntimeError(
            "compatibility_lock.yaml is invalid."
        )

    return data


LOCK = load_lock()


MODELS = LOCK[
    "models"
]


# ---------------------------------------------------------
# Kaggle dataset roots
# ---------------------------------------------------------

Q4_DIR = Path(
    MODELS[
        "ltx_q4"
    ][
        "dataset"
    ]
)

VAE_DIR = Path(
    MODELS[
        "vae"
    ][
        "dataset"
    ]
)

T5_DIR = Path(
    MODELS[
        "t5_q4"
    ][
        "dataset"
    ]
)

ENHANCER_DIR = Path(
    MODELS[
        "ic_lora"
    ][
        "dataset"
    ]
)


# ---------------------------------------------------------
# Model files
# ---------------------------------------------------------

Q4_MODEL = (
    Q4_DIR
    / MODELS[
        "ltx_q4"
    ][
        "filename"
    ]
)

VAE_MODEL = (
    VAE_DIR
    / MODELS[
        "vae"
    ][
        "filename"
    ]
)

T5_MODEL = (
    T5_DIR
    / MODELS[
        "t5_q4"
    ][
        "filename"
    ]
)

DETAILER_LORA = (
    Path(
        MODELS[
            "ic_lora"
        ][
            "dataset"
        ]
    )
    / MODELS[
        "ic_lora"
    ][
        "filename"
    ]
)

SPATIAL_UPSCALER = (
    Path(
        MODELS[
            "spatial_upscaler"
        ][
            "dataset"
        ]
    )
    / MODELS[
        "spatial_upscaler"
    ][
        "filename"
    ]
)


# ---------------------------------------------------------
# Target model locations inside ComfyUI
# ---------------------------------------------------------

Q4_TARGET = (
    COMFYUI_DIR
    / MODELS[
        "ltx_q4"
    ][
        "target"
    ]
)

T5_TARGET = (
    COMFYUI_DIR
    / MODELS[
        "t5_q4"
    ][
        "target"
    ]
)

VAE_TARGET = (
    COMFYUI_DIR
    / MODELS[
        "vae"
    ][
        "target"
    ]
)

DETAILER_LORA_TARGET = (
    COMFYUI_DIR
    / MODELS[
        "ic_lora"
    ][
        "target"
    ]
)

SPATIAL_UPSCALER_TARGET = (
    COMFYUI_DIR
    / MODELS[
        "spatial_upscaler"
    ][
        "target"
    ]
)


# ---------------------------------------------------------
# Output / workflow locations
# ---------------------------------------------------------

OUTPUT_DIR = (
    PROJECT_ROOT
    / "output"
)

WORKFLOW_DIR = (
    PROJECT_ROOT
    / "workflows"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

WORKFLOW_DIR.mkdir(
    parents=True,
    exist_ok=True,
)
