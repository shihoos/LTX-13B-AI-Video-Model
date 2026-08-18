from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


LEGACY_LTX_COMMIT = (
    "ee11be3ce229c3afd5fadf8a1258eb8b84af33b1"
)

LTX_REPO = (
    "https://github.com/Lightricks/"
    "ComfyUI-LTXVideo.git"
)


def run(
    command: list[str],
    cwd: Path | None = None,
) -> None:

    subprocess.run(
        command,
        cwd=cwd,
        check=True,
    )


def ensure_repo(
    path: Path,
    commit: str,
) -> None:

    if (
        path.exists()
        and not (path / ".git").exists()
    ):
        shutil.rmtree(path)

    if not path.exists():

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        run(
            [
                "git",
                "clone",
                LTX_REPO,
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

    actual = subprocess.check_output(
        [
            "git",
            "rev-parse",
            "HEAD",
        ],
        cwd=path,
        text=True,
    ).strip()

    if actual != commit:
        raise RuntimeError(
            f"LTX revision mismatch.\n"
            f"Expected: {commit}\n"
            f"Found: {actual}"
        )


def patch_blur(
    guide_file: Path,
) -> None:

    text = guide_file.read_text(
        encoding="utf-8"
    )

    start = text.find(
        "def blur_internal(image, blur_radius):"
    )

    if start < 0:
        raise RuntimeError(
            "blur_internal() not found."
        )

    end = text.find(
        "\n\n@",
        start,
    )

    if end < 0:
        raise RuntimeError(
            "Could not find end of blur_internal()."
        )

    replacement = "\n".join(
        [
            "def blur_internal(image, blur_radius):",
            "    \"\"\"",
            "    Modern ComfyUI-compatible",
            "    implementation of the legacy",
            "    LTX 0.9.8 conditioning blur.",
            "",
            "    The sampler calls:",
            "        blur_internal(image, blur)",
            "",
            "    The tested workflow uses blur=1.",
            "    \"\"\"",
            "",
            "    import torch",
            "    import torch.nn.functional as F",
            "",
            "    radius = int(blur_radius)",
            "",
            "    if radius <= 0:",
            "        return image",
            "",
            "    if image.ndim != 4:",
            "        raise ValueError(",
            "            f\"Expected IMAGE [B,H,W,C], got {tuple(image.shape)}\"",
            "        )",
            "",
            "    sigma = 1.0",
            "    kernel_size = radius * 2 + 1",
            "",
            "    coords = torch.arange(",
            "        kernel_size,",
            "        device=image.device,",
            "        dtype=torch.float32,",
            "    ) - float(radius)",
            "",
            "    kernel_1d = torch.exp(",
            "        -(coords ** 2)",
            "        / (2.0 * sigma * sigma)",
            "    )",
            "",
            "    kernel_1d = (",
            "        kernel_1d",
            "        / kernel_1d.sum()",
            "    )",
            "",
            "    kernel_2d = (",
            "        kernel_1d[:, None]",
            "        * kernel_1d[None, :]",
            "    )",
            "",
            "    channels = image.shape[-1]",
            "",
            "    kernel = (",
            "        kernel_2d",
            "        .to(",
            "            device=image.device,",
            "            dtype=image.dtype,",
            "        )",
            "        .repeat(",
            "            channels,",
            "            1,",
            "            1,",
            "        )",
            "        .unsqueeze(1)",
            "    )",
            "",
            "    work = image.permute(",
            "        0,",
            "        3,",
            "        1,",
            "        2,",
            "    )",
            "",
            "    work = F.pad(",
            "        work,",
            "        (",
            "            radius,",
            "            radius,",
            "            radius,",
            "            radius,",
            "        ),",
            "        mode=\"reflect\",",
            "    )",
            "",
            "    work = F.conv2d(",
            "        work,",
            "        kernel,",
            "        padding=0,",
            "        groups=channels,",
            "    )",
            "",
            "    return work.permute(",
            "        0,",
            "        2,",
            "        3,",
            "        1,",
            "    )",
        ]
    )

    patched = (
        text[:start]
        + replacement
        + text[end:]
    )

    if (
        "post_processing.Blur().blur("
        in patched
    ):
        raise RuntimeError(
            "Old Blur().blur() still remains."
        )

    guide_file.write_text(
        patched,
        encoding="utf-8",
    )


def write_curated_init(
    target: Path,
) -> None:

    init_code = """from .decoder_noise import DecoderNoise
from .easy_samplers import LTXVBaseSampler
from .film_grain import LTXVFilmGrain
from .latent_upsampler import (
    LTXVLatentUpsampler,
    LTXVLatentUpsamplerModelLoader,
)
from .looping_sampler import LTXVLoopingSampler
from .stg import STGGuiderAdvancedNode
from .tiled_sampler import LTXVTiledSampler
from .tiled_vae_decode import LTXVTiledVAEDecode


NODE_CLASS_MAPPINGS = {
    "LTXVBaseSampler":
        LTXVBaseSampler,

    "LTXVLoopingSampler":
        LTXVLoopingSampler,

    "LTXVTiledSampler":
        LTXVTiledSampler,

    "LTXVTiledVAEDecode":
        LTXVTiledVAEDecode,

    "LTXVLatentUpsampler":
        LTXVLatentUpsampler,

    "LTXVLatentUpsamplerModelLoader":
        LTXVLatentUpsamplerModelLoader,

    "LTXVFilmGrain":
        LTXVFilmGrain,

    "STGGuiderAdvanced":
        STGGuiderAdvancedNode,

    "Set VAE Decoder Noise":
        DecoderNoise,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    key: key
    for key in NODE_CLASS_MAPPINGS
}


__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
"""

    (
        target / "__init__.py"
    ).write_text(
        init_code,
        encoding="utf-8",
    )


def build_compat_package(
    custom_nodes: Path,
    cache_root: Path,
) -> Path:

    legacy = (
        cache_root
        / "ltx098_source"
    )

    target = (
        custom_nodes
        / "LTX098ModernCompat"
    )

    ensure_repo(
        legacy,
        LEGACY_LTX_COMMIT,
    )

    if target.exists():
        shutil.rmtree(target)

    target.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copytree(
        legacy,
        target,
        ignore=shutil.ignore_patterns(
            ".git",
            ".github",
            "__pycache__",
            "*.pyc",
            "*.pyo",
        ),
    )

    write_curated_init(
        target
    )

    patch_blur(
        target / "guide.py"
    )

    return target


if __name__ == "__main__":

    project = Path(
        os.getenv(
            "LTX_PROJECT_ROOT",
            "/kaggle/working/LTX-13B-AI-Video-Model",
        )
    )

    result = build_compat_package(
        project
        / "ComfyUI"
        / "custom_nodes",

        project
        / ".runtime_ltx098",
    )

    print(
        "=" * 80
    )

    print(
        "LTX 0.9.8 MODERN COMPATIBILITY PACKAGE READY"
    )

    print(
        "=" * 80
    )

    print(
        "Package:",
        result,
    )

    print(
        "Legacy commit:",
        LEGACY_LTX_COMMIT,
    )

    print(
        "Blur: native torch"
    )
