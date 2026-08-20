from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path


PROJECT_DEFAULT = (
    "/kaggle/working/LTX-13B-AI-Video-Model"
)


LTX_REPO = (
    "https://github.com/Lightricks/"
    "ComfyUI-LTXVideo.git"
)


def run(
    command: list[str],
    cwd: Path | None = None,
) -> None:

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


def load_lock(
    project: Path,
):

    lock_file = (
        project
        / "kaggle"
        / "compatibility_lock.yaml"
    )

    if not lock_file.exists():

        raise FileNotFoundError(
            "Compatibility lock not found:\n"
            f"{lock_file}"
        )

    try:

        import yaml

    except ImportError as error:

        raise RuntimeError(
            "PyYAML is required to read "
            "compatibility_lock.yaml."
        ) from error

    with lock_file.open(
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


def get_legacy_commit(
    lock,
):

    try:

        commit = (
            lock[
                "legacy_ltx_098_compat"
            ][
                "commit"
            ]
        )

    except KeyError as error:

        raise RuntimeError(
            "compatibility_lock.yaml is missing "
            "legacy_ltx_098_compat.commit"
        ) from error

    if not isinstance(
        commit,
        str,
    ):

        raise RuntimeError(
            "Legacy LTX commit must be a string."
        )

    commit = commit.strip()

    if len(commit) != 40:

        raise RuntimeError(
            "Legacy LTX commit must be a "
            "40-character SHA."
        )

    return commit


def ensure_repo(
    path: Path,
    commit: str,
) -> None:

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

    actual = (
        subprocess.check_output(
            [
                "git",
                "rev-parse",
                "HEAD",
            ],
            cwd=path,
            text=True,
        )
        .strip()
    )

    if actual != commit:

        raise RuntimeError(
            "LTX revision mismatch.\n"
            f"Expected: {commit}\n"
            f"Found:    {actual}"
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
            "Could not determine end of "
            "blur_internal()."
        )

    replacement = """
def blur_internal(image, blur_radius):
    \"\"\"
    Modern ComfyUI-compatible
    implementation of the legacy
    LTX 0.9.8 conditioning blur.

    The tested production workflow
    uses blur=1.
    \"\"\"

    import torch
    import torch.nn.functional as F

    radius = int(
        blur_radius
    )

    if radius <= 0:

        return image

    if image.ndim != 4:

        raise ValueError(
            "Expected IMAGE "
            "[B,H,W,C], got "
            f"{tuple(image.shape)}"
        )

    sigma = 1.0

    kernel_size = (
        radius * 2 + 1
    )

    coords = torch.arange(
        kernel_size,
        device=image.device,
        dtype=torch.float32,
    ) - float(radius)

    kernel_1d = torch.exp(
        -(coords ** 2)
        / (
            2.0
            * sigma
            * sigma
        )
    )

    kernel_1d = (
        kernel_1d
        / kernel_1d.sum()
    )

    kernel_2d = (
        kernel_1d[:, None]
        * kernel_1d[None, :]
    )

    channels = (
        image.shape[-1]
    )

    kernel = (
        kernel_2d
        .to(
            device=image.device,
            dtype=image.dtype,
        )
        .repeat(
            channels,
            1,
            1,
        )
        .unsqueeze(1)
    )

    work = image.permute(
        0,
        3,
        1,
        2,
    )

    work = F.pad(
        work,
        (
            radius,
            radius,
            radius,
            radius,
        ),
        mode="reflect",
    )

    work = F.conv2d(
        work,
        kernel,
        padding=0,
        groups=channels,
    )

    return work.permute(
        0,
        2,
        3,
        1,
    )
""".lstrip()

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
            "Old Blur().blur() call still exists."
        )

    guide_file.write_text(
        patched,
        encoding="utf-8",
    )
def patch_modern_pyramid_blending(
    pyramid_file: Path,
) -> None:
    """
    Apply the Kornia >= 0.8.3 compatibility fix to the
    pinned ComfyUI-LTXVideo checkout.

    The upstream file currently imports `pad` from:
        kornia.geometry.transform.pyramid

    Modern Kornia no longer exports that symbol there.
    PyTorch's F.pad is the equivalent operation.

    This patch is intentionally strict:
    if the expected source shape changes, fail instead
    of silently modifying the wrong code.
    """

    if not pyramid_file.exists():
        raise FileNotFoundError(
            "Modern LTXVideo pyramid_blending.py not found:\n"
            f"{pyramid_file}"
        )

    text = pyramid_file.read_text(
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # Already patched: verify and return.
    # --------------------------------------------------------

    if "pad = F.pad" in text:
        if re.search(
            r"from\s+kornia\.geometry\.transform\.pyramid\s+import\s*\("
            r"(?P<body>.*?)\)",
            text,
            flags=re.DOTALL,
        ):
            match = re.search(
                r"from\s+kornia\.geometry\.transform\.pyramid\s+import\s*\("
                r"(?P<body>.*?)\)",
                text,
                flags=re.DOTALL,
            )

            if match and re.search(
                r"^\s*pad\s*,?\s*$",
                match.group("body"),
                flags=re.MULTILINE,
            ):
                raise RuntimeError(
                    "pyramid_blending.py contains both "
                    "pad = F.pad and the old Kornia pad import."
                )

        print(
            "✅ Modern LTXVideo pyramid_blending.py "
            "already patched."
        )
        return

    # --------------------------------------------------------
    # F.pad must already exist in this exact source.
    # The pinned ac4d998 file already imports it.
    # --------------------------------------------------------

    if (
        "import torch.nn.functional as F"
        not in text
    ):
        raise RuntimeError(
            "Expected `import torch.nn.functional as F` "
            "was not found in pyramid_blending.py."
        )

    # --------------------------------------------------------
    # Find the exact Kornia pyramid import block.
    # --------------------------------------------------------

    pattern = re.compile(
        r"from\s+kornia\.geometry\.transform\.pyramid\s+import\s*\("
        r"(?P<body>.*?)"
        r"\)",
        flags=re.DOTALL,
    )

    match = pattern.search(text)

    if not match:
        raise RuntimeError(
            "Expected Kornia pyramid import block "
            "was not found."
        )

    body = match.group("body")

    # --------------------------------------------------------
    # Require the exact obsolete symbol.
    # --------------------------------------------------------

    if not re.search(
        r"^\s*pad\s*,?\s*$",
        body,
        flags=re.MULTILINE,
    ):
        raise RuntimeError(
            "Expected obsolete Kornia `pad` import "
            "was not found."
        )

    # --------------------------------------------------------
    # Remove ONLY `pad` from that import block.
    # --------------------------------------------------------

    new_body = re.sub(
        r"^\s*pad\s*,?\s*$\n?",
        "",
        body,
        count=1,
        flags=re.MULTILINE,
    )

    new_import = (
        "from kornia.geometry.transform.pyramid import ("
        + new_body
        + ")"
    )

    text = (
        text[:match.start()]
        + new_import
        + text[match.end():]
    )

    # --------------------------------------------------------
    # Define pad from PyTorch immediately after the F import.
    # --------------------------------------------------------

    marker = (
        "import torch.nn.functional as F\n"
    )

    replacement = (
        "import torch.nn.functional as F\n"
        "\n"
        "# Kornia >= 0.8.3 compatibility:\n"
        "# `pad` is no longer re-exported by Kornia.\n"
        "pad = F.pad\n"
    )

    if text.count(marker) != 1:
        raise RuntimeError(
            "Could not locate a unique "
            "`torch.nn.functional as F` import."
        )

    text = text.replace(
        marker,
        replacement,
        1,
    )

    # --------------------------------------------------------
    # Final source-level verification.
    # --------------------------------------------------------

    if "pad = F.pad" not in text:
        raise RuntimeError(
            "Failed to add `pad = F.pad`."
        )

    remaining = re.search(
        r"from\s+kornia\.geometry\.transform\.pyramid\s+import\s*\("
        r"(?P<body>.*?)\)",
        text,
        flags=re.DOTALL,
    )

    if (
        remaining
        and re.search(
            r"^\s*pad\s*,?\s*$",
            remaining.group("body"),
            flags=re.MULTILINE,
        )
    ):
        raise RuntimeError(
            "Old Kornia `pad` import still remains."
        )

    pyramid_file.write_text(
        text,
        encoding="utf-8",
    )

    print(
        "✅ Applied modern Kornia compatibility patch:"
    )
    print(
        f"   {pyramid_file}"
    )

def write_curated_init(
    target: Path,
) -> None:

    init_code = """
from .decoder_noise import DecoderNoise

from .easy_samplers import (
    LTXVBaseSampler,
)

from .film_grain import (
    LTXVFilmGrain,
)

from .latent_upsampler import (
    LTXVLatentUpsampler,
    LTXVLatentUpsamplerModelLoader,
)

from .looping_sampler import (
    LTXVLoopingSampler,
)

from .stg import (
    STGGuiderAdvancedNode,
)

from .tiled_sampler import (
    LTXVTiledSampler,
)

from .tiled_vae_decode import (
    LTXVTiledVAEDecode,
)


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
""".lstrip()

    (
        target
        / "__init__.py"
    ).write_text(
        init_code,
        encoding="utf-8",
    )


def build_compat_package(
    project: Path,
    lock,
) -> Path:

    commit = get_legacy_commit(
        lock
    )

    custom_nodes = (
        project
        / "ComfyUI"
        / "custom_nodes"
    )

    cache_root = (
        project
        / ".runtime_ltx098"
    )

    legacy_source = (
        cache_root
        / "ltx098_source"
    )

    target = (
        custom_nodes
        / lock[
            "legacy_ltx_098_compat"
        ][
            "runtime_package"
        ]
    )

    ensure_repo(
        legacy_source,
        commit,
    )

    if target.exists():

        shutil.rmtree(
            target
        )

    target.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copytree(
        legacy_source,
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
        target
        / "guide.py"
    )

    return target
def patch_active_ltxvideo(
    project: Path,
) -> None:
    """
    Patch the exact pinned ComfyUI-LTXVideo Git checkout
    that ComfyUI will import.
    """

    runtime_package = (
        project
        / "ComfyUI"
        / "custom_nodes"
        / "ComfyUI-LTXVideo"
    )

    pyramid_file = (
        runtime_package
        / "pyramid_blending.py"
    )

    if not runtime_package.exists():
        raise FileNotFoundError(
            "Active ComfyUI-LTXVideo checkout not found:\n"
            f"{runtime_package}"
        )

    if not (
        runtime_package / ".git"
    ).exists():
        raise RuntimeError(
            "Active ComfyUI-LTXVideo directory is not "
            "a Git checkout:\n"
            f"{runtime_package}"
        )

    patch_modern_pyramid_blending(
        pyramid_file
    )

def main():

    project = Path(
        os.getenv(
            "LTX_PROJECT_ROOT",
            PROJECT_DEFAULT,
        )
    )

    lock = load_lock(
        project
    )

    commit = get_legacy_commit(
        lock
    )

    package = build_compat_package(
        project,
        lock,
    )

    print()
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
        f"Package: {package}"
    )

    print(
        f"Legacy LTX commit: {commit}"
    )

    print(
        "Blur implementation: native torch"
    )

    print(
        "Workflow blur: "
        f"{lock['legacy_ltx_098_compat']['patches']['blur_internal']['workflow_blur']}"
    )


if __name__ == "__main__":

    main()
