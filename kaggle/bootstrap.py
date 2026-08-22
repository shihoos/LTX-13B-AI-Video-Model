from __future__ import annotations

import os
import shutil
import subprocess
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

CUSTOM = (
    COMFY
    / "custom_nodes"
)


def run(
    *args: str,
) -> None:

    print(
        "+",
        " ".join(args),
    )

    subprocess.run(
        args,
        check=True,
    )


def clone(
    url: str,
    destination: Path,
) -> None:

    if destination.exists():
        print(
            "EXISTS:",
            destination,
        )
        return

    run(
        "git",
        "clone",
        "--depth",
        "1",
        url,
        str(destination),
    )


def find_dataset() -> Path:

    required_names = [
        (
            "models",
            "diffusion_models",
            "minimax_h3_ref2va_pruned-Q4_K_M.gguf",
        ),
        (
            "models",
            "text_encoders",
            "qwen3vl_32b_minimax_h3-Q4_K_M.gguf",
        ),
        (
            "models",
            "text_encoders",
            "Qwen3-VL-32B-Instruct-MiniMax-H3-L0-49-mmproj-BF16.gguf",
        ),
        (
            "models",
            "vae",
            "minimax_h3_video_vae_fp16.safetensors",
        ),
        (
            "models",
            "vae",
            "minimax_h3_audio_vae_fp32.safetensors",
        ),
    ]

    kaggle_root = Path(
        "/kaggle/input"
    )

    if not kaggle_root.is_dir():
        raise FileNotFoundError(
            "/kaggle/input does not exist."
        )

    candidates = []

    for model in kaggle_root.rglob(
        "minimax_h3_ref2va_pruned-Q4_K_M.gguf"
    ):
        candidates.append(
            model.parent.parent.parent
        )

    for root in candidates:

        ok = True

        for parts in required_names:
            path = root.joinpath(
                *parts
            )

            if not path.is_file():
                ok = False
                break

        if ok:
            return root

    raise FileNotFoundError(
        "Complete MiniMax H3 Ref2VA Q4 dataset "
        "was not found under /kaggle/input.\n\n"
        "Expected:\n"
        "models/diffusion_models/"
        "minimax_h3_ref2va_pruned-Q4_K_M.gguf\n"
        "models/text_encoders/"
        "qwen3vl_32b_minimax_h3-Q4_K_M.gguf\n"
        "models/text_encoders/"
        "Qwen3-VL-32B-Instruct-MiniMax-H3-L0-49-mmproj-BF16.gguf\n"
        "models/vae/"
        "minimax_h3_video_vae_fp16.safetensors\n"
        "models/vae/"
        "minimax_h3_audio_vae_fp32.safetensors"
    )


def link(
    source: Path,
    destination: Path,
) -> None:

    if not source.is_file():
        raise FileNotFoundError(
            f"Missing source model: {source}"
        )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if (
        destination.exists()
        or destination.is_symlink()
    ):
        destination.unlink()

    try:
        destination.symlink_to(
            source
        )
        mode = "symlink"
    except OSError:
        shutil.copy2(
            source,
            destination,
        )
        mode = "copy"

    print(
        f"MODEL {mode}:",
        destination,
    )


def install_custom_nodes():

    CUSTOM.mkdir(
        parents=True,
        exist_ok=True,
    )

    clone(
        "https://github.com/city96/ComfyUI-GGUF.git",
        CUSTOM / "ComfyUI-GGUF",
    )

    clone(
        "https://github.com/jlucasmcrell/ComfyUI-H3-Multishot.git",
        CUSTOM / "ComfyUI-H3-Multishot",
    )

    clone(
        "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git",
        CUSTOM / "ComfyUI-VideoHelperSuite",
    )

    clone(
        "https://github.com/SethRobinson/comfyui-workflow-to-api-converter-endpoint.git",
        CUSTOM
        / "comfyui-workflow-to-api-converter-endpoint",
    )


def install_comfy():

    COMFY.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not (
        COMFY
        / "main.py"
    ).is_file():

        clone(
            "https://github.com/comfyanonymous/ComfyUI.git",
            COMFY,
        )

    run(
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "-r",
        str(
            COMFY
            / "requirements.txt"
        ),
    )


def install_models():

    dataset = find_dataset()

    models = (
        dataset
        / "models"
    )

    comfy_models = (
        COMFY
        / "models"
    )

    link(
        models
        / "diffusion_models"
        / "minimax_h3_ref2va_pruned-Q4_K_M.gguf",
        comfy_models
        / "diffusion_models"
        / "minimax_h3_ref2va_pruned-Q4_K_M.gguf",
    )

    link(
        models
        / "text_encoders"
        / "qwen3vl_32b_minimax_h3-Q4_K_M.gguf",
        comfy_models
        / "text_encoders"
        / "qwen3vl_32b_minimax_h3-Q4_K_M.gguf",
    )

    link(
        models
        / "text_encoders"
        / "Qwen3-VL-32B-Instruct-MiniMax-H3-L0-49-mmproj-BF16.gguf",
        comfy_models
        / "text_encoders"
        / "Qwen3-VL-32B-Instruct-MiniMax-H3-L0-49-mmproj-BF16.gguf",
    )

    link(
        models
        / "vae"
        / "minimax_h3_video_vae_fp16.safetensors",
        comfy_models
        / "vae"
        / "minimax_h3_video_vae_fp16.safetensors",
    )

    link(
        models
        / "vae"
        / "minimax_h3_audio_vae_fp32.safetensors",
        comfy_models
        / "vae"
        / "minimax_h3_audio_vae_fp32.safetensors",
    )

    # --------------------------------------------------------------
    # IMPORTANT:
    #
    # ComfyUI-GGUF's automatic mmproj matcher searches for the
    # encoder's base name inside the mmproj filename.
    #
    # Your dataset filename is:
    #
    # Qwen3-VL-32B-Instruct-MiniMax-H3-L0-49-mmproj-BF16.gguf
    #
    # while the encoder is:
    #
    # qwen3vl_32b_minimax_h3-Q4_K_M.gguf
    #
    # Create a compatibility alias without modifying the dataset.
    # --------------------------------------------------------------

    original_mmproj = (
        comfy_models
        / "text_encoders"
        / "Qwen3-VL-32B-Instruct-MiniMax-H3-L0-49-mmproj-BF16.gguf"
    )

    mmproj_alias = (
        comfy_models
        / "text_encoders"
        / "qwen3vl_32b_minimax_h3-mmproj-BF16.gguf"
    )

    if (
        original_mmproj.is_file()
        and not mmproj_alias.exists()
    ):
        try:
            mmproj_alias.symlink_to(
                original_mmproj
            )
            print(
                "Created mmproj compatibility alias:",
                mmproj_alias,
            )
        except OSError:
            shutil.copy2(
                original_mmproj,
                mmproj_alias,
            )
            print(
                "Created mmproj compatibility copy:",
                mmproj_alias,
            )


def main():

    install_comfy()

    install_custom_nodes()

    install_models()

    print()
    print(
        "=" * 72
    )
    print(
        "MiniMax H3 Ref2VA Q4 bootstrap complete."
    )
    print(
        "=" * 72
    )
    print(
        "Diffusion:",
        "minimax_h3_ref2va_pruned-Q4_K_M.gguf",
    )
    print(
        "Text encoder:",
        "qwen3vl_32b_minimax_h3-Q4_K_M.gguf",
    )
    print(
        "Video VAE:",
        "minimax_h3_video_vae_fp16.safetensors",
    )
    print(
        "Audio VAE:",
        "minimax_h3_audio_vae_fp32.safetensors",
    )
    print(
        "Workflow converter:",
        "installed",
    )


if __name__ == "__main__":
    main()
