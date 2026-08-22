from __future__ import annotations

import json
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
    COMFY
    / "main.py",

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
    / "custom_nodes"
    / "comfyui-workflow-to-api-converter-endpoint",

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
    / "text_encoders"
    / "qwen3vl_32b_minimax_h3-mmproj-BF16.gguf",

    COMFY
    / "models"
    / "vae"
    / "minimax_h3_video_vae_fp16.safetensors",

    COMFY
    / "models"
    / "vae"
    / "minimax_h3_audio_vae_fp32.safetensors",
]

WORKFLOWS = [
    ROOT
    / "workflows"
    / "MiniMax-H3"
    / "base"
    / "H3_HardMode_R2V.json",

    ROOT
    / "workflows"
    / "MiniMax-H3"
    / "base"
    / "H3_HardMode_Chained.json",
]


def check_files() -> bool:

    failed = False

    print(
        "\nFILES"
    )
    print(
        "-" * 72
    )

    for path in REQUIRED_FILES:

        exists = (
            path.exists()
            or path.is_symlink()
        )

        print(
            "[OK]   "
            if exists
            else "[FAIL] ",
            path,
        )

        if not exists:
            failed = True

    return not failed


def check_workflows() -> bool:

    failed = False

    print(
        "\nWORKFLOWS"
    )
    print(
        "-" * 72
    )

    for path in WORKFLOWS:

        if not path.is_file():
            print(
                "[FAIL] ",
                path,
            )
            failed = True
            continue

        try:
            data = json.loads(
                path.read_text(
                    encoding="utf-8"
                )
            )
        except Exception as error:
            print(
                "[FAIL] ",
                path,
                error,
            )
            failed = True
            continue

        if not isinstance(
            data,
            dict,
        ):
            print(
                "[FAIL] workflow root is not object:",
                path,
            )
            failed = True
            continue

        print(
            "[OK]   ",
            path,
        )

    return not failed


def check_imports() -> bool:

    sys.path.insert(
        0,
        str(COMFY),
    )

    print(
        "\nIMPORTS"
    )
    print(
        "-" * 72
    )

    try:
        import nodes  # noqa: F401

        print(
            "[OK]   ComfyUI nodes"
        )

    except Exception as error:
        print(
            "[FAIL] ComfyUI nodes:",
            error,
        )
        return False

    try:
        import comfy_extras.nodes_minimax_h3  # noqa: F401

        print(
            "[OK]   native MiniMax H3"
        )

    except Exception as error:
        print(
            "[FAIL] native MiniMax H3:",
            error,
        )
        return False

    return True


def main() -> int:

    print(
        "=" * 72
    )
    print(
        "MINIMAX H3 REF2VA Q4 PREFLIGHT"
    )
    print(
        "=" * 72
    )

    ok = True

    ok &= check_files()
    ok &= check_workflows()
    ok &= check_imports()

    print()

    if ok:
        print(
            "PREFLIGHT PASSED."
        )
        print(
            "Next: start ComfyUI and run verify_live_runtime.py."
        )
        return 0

    print(
        "PREFLIGHT FAILED."
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
