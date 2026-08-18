from __future__ import annotations

import importlib.metadata
import os
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT = Path(
    os.getenv(
        "LTX_PROJECT_ROOT",
        "/kaggle/working/LTX-13B-AI-Video-Model",
    )
)

COMFY = (
    PROJECT
    / "ComfyUI"
)

CUSTOM = (
    COMFY
    / "custom_nodes"
)


COMFY_REPO = (
    "https://github.com/Comfy-Org/ComfyUI.git"
)

COMFY_COMMIT = (
    "72865f4f27eaf5396f8f36370e0a2be3a9a090ee"
)


NODES = {

    "ComfyUI-LTXVideo": (
        "https://github.com/Lightricks/"
        "ComfyUI-LTXVideo.git",

        "ac4d99839020b983e956a8ab67ec38aec1b6e65a",
    ),

    "ComfyUI-KJNodes": (
        "https://github.com/kijai/"
        "ComfyUI-KJNodes.git",

        "7ecb190ef91d988420cf0e682efb79ac7433c0b7",
    ),

    "ComfyUI-VideoHelperSuite": (
        "https://github.com/Kosinkadink/"
        "ComfyUI-VideoHelperSuite.git",

        "4ee72c065db22c9d96c2427954dc69e7b908444b",
    ),

    "ComfyUI-GGUF": (
        "https://github.com/city96/"
        "ComfyUI-GGUF.git",

        "6ea2651e7df66d7585f6ffee804b20e92fb38b8a",
    ),
}


MODEL_SOURCES = {

    "ltx_q4": Path(
        "/kaggle/input/datasets/shihoos/"
        "ltx13b-q4/"
        "LTXV-13B-0.9.8-distilled-Q4_K_M.gguf"
    ),

    "t5_q4": Path(
        "/kaggle/input/datasets/shihoos/"
        "ltx13b-t5/"
        "t5-v1_1-xxl-encoder-Q4_K_M.gguf"
    ),

    "vae": Path(
        "/kaggle/input/datasets/shihoos/"
        "ltx13b-vae/"
        "LTXV-13B-0.9.8-distilled-VAE.safetensors"
    ),

    "ic_lora": Path(
        "/kaggle/input/datasets/shihoos/"
        "ltx13b-enhancers/"
        "ltxv-098-ic-lora-detailer-comfyui.safetensors"
    ),

    "spatial": Path(
        "/kaggle/input/datasets/shihoos/"
        "ltx13b-enhancers/"
        "ltxv-spatial-upscaler-0.9.8.safetensors"
    ),
}


PINNED_PACKAGES = {

    "comfyui-frontend-package":
        "1.48.7",

    "comfyui-workflow-templates":
        "0.11.41",

    "comfyui-embedded-docs":
        "0.5.9",

    "comfy-kitchen":
        "0.2.31",

    "comfy-aimdo":
        "0.4.13",

    "torchsde":
        "0.2.6",

    "spandrel":
        "0.4.2",

    "av":
        "18.1.0",

    "gguf":
        "0.19.0",
}


FORBIDDEN_TORCH = {
    "torch",
    "torchvision",
    "torchaudio",
}


def run(
    command,
    cwd=None,
):

    print(
        "$ "
        + " ".join(
            map(
                str,
                command,
            )
        )
    )

    subprocess.run(
        command,
        cwd=cwd,
        check=True,
    )


def git_current(
    path,
):

    return subprocess.check_output(
        [
            "git",
            "-C",
            str(path),
            "rev-parse",
            "HEAD",
        ],
        text=True,
    ).strip()


def sync_repo(
    url,
    path,
    commit,
):

    if (
        path.exists()
        and not (
            path
            / ".git"
        ).exists()
    ):

        shutil.rmtree(
            path
        )

    if not path.exists():

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        run(
            [
                "git",
                "clone",
                url,
                str(path),
            ]
        )

    run(
        [
            "git",
            "fetch",
            "--all",
            "--tags",
            "--prune",
        ],
        cwd=path,
    )

    run(
        [
            "git",
            "checkout",
            "--force",
            commit,
        ],
        cwd=path,
    )

    actual = git_current(
        path
    )

    if actual != commit:

        raise RuntimeError(
            f"Revision mismatch:\n"
            f"Expected: {commit}\n"
            f"Found:    {actual}"
        )


def verify_torch():

    import torch

    print()
    print(
        "=" * 80
    )

    print(
        "PYTORCH / CUDA"
    )

    print(
        "=" * 80
    )

    print(
        "Torch:",
        torch.__version__,
    )

    print(
        "CUDA:",
        torch.version.cuda,
    )

    print(
        "CUDA available:",
        torch.cuda.is_available(),
    )

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA unavailable. "
            "Turn Kaggle GPU ON."
        )

    if not (
        torch.__version__
        .startswith(
            "2.10.0+cu128"
        )
    ):

        raise RuntimeError(
            f"Wrong Torch version: "
            f"{torch.__version__}"
        )

    for index in range(
        torch.cuda.device_count()
    ):

        print(
            f"GPU {index}: "
            f"{torch.cuda.get_device_name(index)}"
        )


def filter_requirements(
    source,
):

    destination = (
        PROJECT
        / ".runtime_requirements_filtered.txt"
    )

    lines = []

    for raw in source.read_text(
        encoding="utf-8"
    ).splitlines():

        line = raw.strip()

        if (
            not line
            or line.startswith("#")
        ):

            continue

        package = (
            line
            .split(
                "==",
                1,
            )[0]
            .split(
                ">=",
                1,
            )[0]
            .split(
                "<=",
                1,
            )[0]
            .split(
                "~=",
                1,
            )[0]
            .split(
                ">",
                1,
            )[0]
            .split(
                "<",
                1,
            )[0]
            .strip()
            .lower()
        )

        if (
            package
            in FORBIDDEN_TORCH
        ):

            continue

        lines.append(
            line
        )

    destination.write_text(
        "\n".join(
            lines
        )
        + "\n",
        encoding="utf-8",
    )

    return destination


def install_requirements(
    source,
):

    if not source.exists():

        return

    filtered = (
        filter_requirements(
            source
        )
    )

    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "-r",
            str(filtered),
        ]
    )

    try:

        filtered.unlink()

    except OSError:

        pass


def install_pinned_packages():

    for (
        package,
        version,
    ) in PINNED_PACKAGES.items():

        run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-deps",
                f"{package}=={version}",
            ]
        )


def verify_pinned_packages():

    for (
        package,
        expected,
    ) in PINNED_PACKAGES.items():

        actual = (
            importlib.metadata
            .version(
                package
            )
        )

        if actual != expected:

            raise RuntimeError(
                f"{package}: "
                f"expected {expected}, "
                f"found {actual}"
            )

        print(
            f"✅ {package}=={actual}"
        )


def link_models():

    direct = {

        "ltx_q4":
            COMFY
            / "models"
            / "unet"
            / MODEL_SOURCES[
                "ltx_q4"
            ].name,

        "t5_q4":
            COMFY
            / "models"
            / "clip"
            / MODEL_SOURCES[
                "t5_q4"
            ].name,

        "vae":
            COMFY
            / "models"
            / "vae"
            / MODEL_SOURCES[
                "vae"
            ].name,

        "ic_lora":
            COMFY
            / "models"
            / "loras"
            / MODEL_SOURCES[
                "ic_lora"
            ].name,
    }

    spatial_dirs = [

        COMFY
        / "models"
        / "latent_upscale_models",

        COMFY
        / "models"
        / "latent_upscale",

        COMFY
        / "models"
        / "latent_upscalers",

        COMFY
        / "models"
        / "ltxv",

        COMFY
        / "models"
        / "upscalers",
    ]

    for (
        key,
        source,
    ) in MODEL_SOURCES.items():

        if not source.exists():

            raise FileNotFoundError(
                f"{key} model not found:\n"
                f"{source}"
            )

    for (
        key,
        target,
    ) in direct.items():

        target.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if (
            target.exists()
            or target.is_symlink()
        ):

            target.unlink()

        target.symlink_to(
            MODEL_SOURCES[
                key
            ].resolve()
        )

        print(
            f"✅ {key}: "
            f"{target}"
        )

    spatial = (
        MODEL_SOURCES[
            "spatial"
        ]
    )

    for directory in (
        spatial_dirs
    ):

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        target = (
            directory
            / spatial.name
        )

        if (
            target.exists()
            or target.is_symlink()
        ):

            target.unlink()

        target.symlink_to(
            spatial.resolve()
        )

    print(
        "✅ spatial model linked"
    )


def write_extra_model_paths():

    content = f"""
ltx_project:
  is_default: true

  diffusion_models: |
    {MODEL_SOURCES["ltx_q4"].parent}

  unet: |
    {MODEL_SOURCES["ltx_q4"].parent}

  text_encoders: |
    {MODEL_SOURCES["t5_q4"].parent}

  clip: |
    {MODEL_SOURCES["t5_q4"].parent}

  vae: |
    {MODEL_SOURCES["vae"].parent}

  loras: |
    {MODEL_SOURCES["ic_lora"].parent}

  upscale_models: |
    {MODEL_SOURCES["spatial"].parent}

  latent_upscale_models: |
    {MODEL_SOURCES["spatial"].parent}
"""

    (
        COMFY
        / "extra_model_paths.yaml"
    ).write_text(
        content,
        encoding="utf-8",
    )


def prepare_compatibility():

    script = (
        PROJECT
        / "compatibility"
        / "prepare_modern_ltx.py"
    )

    run(
        [
            sys.executable,
            str(script),
        ]
    )


def main():

    print(
        "=" * 80
    )

    print(
        "LTX-13B MODERN STACK BOOTSTRAP"
    )

    print(
        "=" * 80
    )

    if not PROJECT.exists():

        raise FileNotFoundError(
            f"Project not found:\n"
            f"{PROJECT}"
        )

    verify_torch()

    sync_repo(
        COMFY_REPO,
        COMFY,
        COMFY_COMMIT,
    )

    for (
        name,
        pair,
    ) in NODES.items():

        url, commit = pair

        sync_repo(
            url,
            CUSTOM / name,
            commit,
        )

    install_requirements(
        COMFY
        / "requirements.txt"
    )

    install_requirements(
        CUSTOM
        / "ComfyUI-LTXVideo"
        / "requirements.txt"
    )

    for name in (
        "ComfyUI-KJNodes",
        "ComfyUI-VideoHelperSuite",
        "ComfyUI-GGUF",
    ):

        install_requirements(
            CUSTOM
            / name
            / "requirements.txt"
        )

    install_pinned_packages()

    verify_pinned_packages()

    prepare_compatibility()

    link_models()

    write_extra_model_paths()

    print(
        "=" * 80
    )

    print(
        "MODERN STACK BOOTSTRAP COMPLETE"
    )

    print(
        "=" * 80
    )

    print(
        "ComfyUI:",
        COMFY,
    )

    print(
        "ComfyUI commit:",
        git_current(
            COMFY
        ),
    )

    print(
        "Current LTXVideo:",
        git_current(
            CUSTOM
            / "ComfyUI-LTXVideo"
        ),
    )

    print(
        "Compatibility:",
        CUSTOM
        / "LTX098ModernCompat",
    )


if __name__ == "__main__":
    main()
