from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

COMFY_ROOT = (
    PROJECT_ROOT
    / "ComfyUI"
)

CUSTOM_ROOT = (
    COMFY_ROOT
    / "custom_nodes"
)


def run(*args: str) -> None:

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
        return

    run(
        "git",
        "clone",
        "--depth",
        "1",
        url,
        str(destination),
    )


def find_h3_dataset() -> Path:

    marker = (
        "minimax_h3_ref2va_pruned-Q4_K_M.gguf"
    )

    for model in Path(
        "/kaggle/input"
    ).rglob(marker):

        root = (
            model.parent.parent.parent
        )

        required = [
            root
            / "models"
            / "diffusion_models"
            / "minimax_h3_ref2va_pruned-Q4_K_M.gguf",

            root
            / "models"
            / "text_encoders"
            / "qwen3vl_32b_minimax_h3-Q4_K_M.gguf",

            root
            / "models"
            / "text_encoders"
            / "Qwen3-VL-32B-Instruct-MiniMax-H3-L0-49-mmproj-BF16.gguf",

            root
            / "models"
            / "vae"
            / "minimax_h3_video_vae_fp16.safetensors",

            root
            / "models"
            / "vae"
            / "minimax_h3_audio_vae_fp32.safetensors",
        ]

        if all(
            path.is_file()
            for path in required
        ):
            return root

    raise FileNotFoundError(
        "Complete MiniMax H3 Ref2VA Q4 dataset "
        "was not found under /kaggle/input."
    )


def link(
    source: Path,
    destination: Path,
) -> None:

    if not source.is_file():
        raise FileNotFoundError(
            source
        )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if (
        destination.is_symlink()
        or destination.exists()
    ):
        destination.unlink()

    destination.symlink_to(
        source
    )

    print(
        "LINK:",
        destination,
        "->",
        source,
    )


def install_models(
    dataset_root: Path,
) -> None:

    models = (
        dataset_root
        / "models"
    )

    link(
        models
        / "diffusion_models"
        / "minimax_h3_ref2va_pruned-Q4_K_M.gguf",
        COMFY_ROOT
        / "models"
        / "diffusion_models"
        / "minimax_h3_ref2va_pruned-Q4_K_M.gguf",
    )

    link(
        models
        / "text_encoders"
        / "qwen3vl_32b_minimax_h3-Q4_K_M.gguf",
        COMFY_ROOT
        / "models"
        / "text_encoders"
        / "qwen3vl_32b_minimax_h3-Q4_K_M.gguf",
    )

    link(
        models
        / "text_encoders"
        / "Qwen3-VL-32B-Instruct-MiniMax-H3-L0-49-mmproj-BF16.gguf",
        COMFY_ROOT
        / "models"
        / "text_encoders"
        / "Qwen3-VL-32B-Instruct-MiniMax-H3-L0-49-mmproj-BF16.gguf",
    )

    link(
        models
        / "vae"
        / "minimax_h3_video_vae_fp16.safetensors",
        COMFY_ROOT
        / "models"
        / "vae"
        / "minimax_h3_video_vae_fp16.safetensors",
    )

    link(
        models
        / "vae"
        / "minimax_h3_audio_vae_fp32.safetensors",
        COMFY_ROOT
        / "models"
        / "vae"
        / "minimax_h3_audio_vae_fp32.safetensors",
    )


def main():

    COMFY_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not (
        COMFY_ROOT
        / "main.py"
    ).exists():

        clone(
            "https://github.com/comfyanonymous/ComfyUI.git",
            COMFY_ROOT,
        )

    run(
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "-r",
        str(
            COMFY_ROOT
            / "requirements.txt"
        ),
    )

    CUSTOM_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    clone(
        "https://github.com/city96/ComfyUI-GGUF.git",
        CUSTOM_ROOT
        / "ComfyUI-GGUF",
    )

    clone(
        "https://github.com/jlucasmcrell/ComfyUI-H3-Multishot.git",
        CUSTOM_ROOT
        / "ComfyUI-H3-Multishot",
    )

    clone(
        "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git",
        CUSTOM_ROOT
        / "ComfyUI-VideoHelperSuite",
    )

    dataset = (
        find_h3_dataset()
    )

    install_models(
        dataset
    )

    print(
        "\nH3 bootstrap complete."
    )


if __name__ == "__main__":
    main()
