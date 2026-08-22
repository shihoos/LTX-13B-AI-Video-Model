from __future__ import annotations

import sys
from pathlib import Path


ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

COMFY = (
    ROOT
    / "ComfyUI"
)


REQUIRED_FILES = [
    COMFY / "main.py",

    COMFY
    / "custom_nodes"
    / "ComfyUI-GGUF",

    COMFY
    / "custom_nodes"
    / "ComfyUI-H3-Multishot",

    COMFY
    / "custom_nodes"
    / "ComfyUI-VideoHelperSuite",

    COMFY
    / "models"
    / "diffusion_models"
    / "minimax_h3_ref2va_pruned-Q4_K_M.gguf",

    COMFY
    / "models"
    / "text_encoders"
    / "qwen3vl_32b_minimax_h3-Q4_K_M.gguf",

    COMFY
    / "models"
    / "text_encoders"
    / "Qwen3-VL-32B-Instruct-MiniMax-H3-L0-49-mmproj-BF16.gguf",

    COMFY
    / "models"
    / "vae"
    / "minimax_h3_video_vae_fp16.safetensors",

    COMFY
    / "models"
    / "vae"
    / "minimax_h3_audio_vae_fp32.safetensors",
]


REQUIRED_NODES = [
    "H3ModelLoaderAny",
    "H3ClipLoaderAny",
    "MiniMaxH3ReferenceToVideo",
    "H3FreeTextEncoder",
    "H3ReferenceAudio",
    "H3MultishotMemorySampler",
    "VAEDecode",
    "VAEDecodeAudio",
    "RandomNoise",
    "BasicGuider",
    "KSamplerSelect",
    "BasicScheduler",
    "SamplerCustomAdvanced",
    "CreateVideo",
    "SaveVideo",
]


def main() -> int:

    print(
        "=" * 72
    )

    print(
        "MINIMAX H3 REF2VA PREFLIGHT"
    )

    print(
        "=" * 72
    )

    failed = False

    for path in REQUIRED_FILES:

        ok = (
            path.exists()
            or path.is_symlink()
        )

        print(
            "OK   "
            if ok
            else "FAIL ",
            path,
        )

        if not ok:
            failed = True

    if failed:
        return 2

    sys.path.insert(
        0,
        str(COMFY),
    )

    try:
        import nodes  # noqa: F401
        import comfy_extras.nodes_minimax_h3  # noqa: F401

    except Exception as error:

        print(
            "Native H3 import failed:"
        )

        print(error)

        return 3

    print(
        "\nRequired runtime files are present."
    )

    print(
        "Native MiniMax H3 core imported."
    )

    print(
        "\nRun verify_live_runtime.py after "
        "starting ComfyUI."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
