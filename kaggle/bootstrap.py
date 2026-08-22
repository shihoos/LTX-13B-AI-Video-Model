from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = PROJECT_ROOT / "ComfyUI"
CUSTOM_ROOT = COMFY_ROOT / "custom_nodes"


def run(*args: str) -> None:
    print("+", " ".join(args))
    subprocess.run(args, check=True)


def clone(url: str, destination: Path) -> None:
    if destination.exists():
        run("git", "-C", str(destination), "pull", "--ff-only")
        return
    run("git", "clone", "--depth", "1", url, str(destination))


def discover_dataset_root() -> Path:
    candidates = Path("/kaggle/input").rglob(
        "minimax_h3_ref2va_pruned-Q4_K_M.gguf"
    )

    for model in candidates:
        parent = model
        for _ in range(3):
            parent = parent.parent
        if (parent / "models").is_dir():
            return parent

    # User's combined dataset currently mounts under this conventional path.
    fallback = Path(
        "/kaggle/input/datasets/shihoos/minimax-h3-ref2va-q4"
    )
    if fallback.exists():
        return fallback

    raise FileNotFoundError(
        "MiniMax H3 Ref2VA dataset could not be discovered under /kaggle/input."
    )


def link_dataset_models(dataset_root: Path) -> None:
    models = dataset_root / "models"

    expected = {
        "diffusion_models": "minimax_h3_ref2va_pruned-Q4_K_M.gguf",
        "text_encoders": "qwen3vl_32b_minimax_h3-Q4_K_M.gguf",
        "vae": "minimax_h3_video_vae_fp16.safetensors",
        "vae_audio": "minimax_h3_audio_vae_fp32.safetensors",
    }

    ref2va = models / "diffusion_models" / expected["diffusion_models"]
    encoder = models / "text_encoders" / expected["text_encoders"]
    video_vae = models / "vae" / expected["vae"]
    audio_vae = models / "vae" / expected["vae_audio"]

    for p in (ref2va, encoder, video_vae, audio_vae):
        if not p.is_file():
            raise FileNotFoundError(
                f"Required H3 dataset file is missing: {p}"
            )

    for folder in ("diffusion_models", "text_encoders", "vae"):
        target = COMFY_ROOT / "models" / folder
        target.mkdir(parents=True, exist_ok=True)

    links = {
        COMFY_ROOT / "models/diffusion_models" / ref2va.name: ref2va,
        COMFY_ROOT / "models/text_encoders" / encoder.name: encoder,
        COMFY_ROOT / "models/vae" / video_vae.name: video_vae,
        COMFY_ROOT / "models/vae" / audio_vae.name: audio_vae,
    }

    for target, source in links.items():
        if target.is_symlink() or target.exists():
            target.unlink()
        target.symlink_to(source)


def main():
    COMFY_ROOT.parent.mkdir(parents=True, exist_ok=True)

    if not COMFY_ROOT.exists():
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
        str(COMFY_ROOT / "requirements.txt"),
    )

    CUSTOM_ROOT.mkdir(parents=True, exist_ok=True)

    clone(
        "https://github.com/city96/ComfyUI-GGUF.git",
        CUSTOM_ROOT / "ComfyUI-GGUF",
    )

    clone(
        "https://github.com/jlucasmcrell/ComfyUI-H3-Multishot.git",
        CUSTOM_ROOT / "ComfyUI-H3-Multishot",
    )

    clone(
        "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git",
        CUSTOM_ROOT / "ComfyUI-VideoHelperSuite",
    )

    dataset_root = discover_dataset_root()
    print("H3 dataset:", dataset_root)
    link_dataset_models(dataset_root)

    print("\nH3 bootstrap complete.")
    print("ComfyUI:", COMFY_ROOT)
    print("H3 model tree:", COMFY_ROOT / "models")


if __name__ == "__main__":
    main()
