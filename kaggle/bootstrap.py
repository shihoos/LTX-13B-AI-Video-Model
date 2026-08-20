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

LOCK_FILE = (
    PROJECT
    / "kaggle"
    / "compatibility_lock.yaml"
)

FORBIDDEN_TORCH_INSTALL = {
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


def load_lock():

    try:
        import yaml

    except ImportError:

        run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "pyyaml",
            ]
        )

        import yaml

    if not LOCK_FILE.exists():

        raise FileNotFoundError(
            "Compatibility lock not found:\n"
            f"{LOCK_FILE}"
        )

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
    """
    Synchronize a repository to one exact locked commit.

    Handles:
      - fresh clones
      - existing normal repositories
      - existing shallow repositories

    This prevents a pinned historical commit from becoming
    unreachable when an existing depth-1/shallow clone is reused.
    """

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

    # ---------------------------------------------------------
    # Detect shallow repository state.
    # ---------------------------------------------------------

    shallow_result = subprocess.run(
        [
            "git",
            "rev-parse",
            "--is-shallow-repository",
        ],
        cwd=path,
        capture_output=True,
        text=True,
        check=True,
    )

    is_shallow = (
        shallow_result.stdout.strip()
        == "true"
    )

    if is_shallow:

        print(
            f"⚠️ Shallow repository detected: "
            f"{path.name}"
        )

        print(
            "   Converting to full history before "
            "checking out the locked commit..."
        )

        run(
            [
                "git",
                "fetch",
                "--unshallow",
                "origin",
            ],
            cwd=path,
        )

    else:

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

    # ---------------------------------------------------------
    # Refresh origin refs/tags after the history is complete.
    # ---------------------------------------------------------

    run(
        [
            "git",
            "fetch",
            "--tags",
            "--prune",
            "origin",
        ],
        cwd=path,
    )

    # ---------------------------------------------------------
    # Checkout the exact locked commit.
    # ---------------------------------------------------------

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
            "Git revision mismatch:\n"
            f"Path:     {path}\n"
            f"Expected: {commit}\n"
            f"Actual:   {actual}"
        )

    print(
        f"✅ {path.name}: {actual}"
    )


def install_requirements(
    source,
):

    if not source.exists():
        return

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

        normalized = (
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

        if normalized in (
            FORBIDDEN_TORCH_INSTALL
        ):

            continue

        lines.append(
            line
        )

    destination.write_text(
        "\n".join(lines)
        + "\n",
        encoding="utf-8",
    )

    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "-r",
            str(destination),
        ]
    )

    try:

        destination.unlink()

    except OSError:
        pass


def install_locked_torch(
    lock,
):
    runtime = lock[
        "python_runtime"
    ]

    torch_version = runtime[
        "torch"
    ]

    torchvision_version = runtime[
        "torchvision"
    ]

    torchaudio_version = runtime[
        "torchaudio"
    ]

    index_url = runtime[
        "torch_index"
    ][
        "url"
    ]

    print()
    print(
        "=" * 80
    )

    print(
        "LOCKED PYTORCH / CUDA INSTALL"
    )

    print(
        "=" * 80
    )

    print(
        "Torch:",
        torch_version,
    )

    print(
        "Torchvision:",
        torchvision_version,
    )

    print(
        "Torchaudio:",
        torchaudio_version,
    )

    print(
        "Index:",
        index_url,
    )

    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "--index-url",
            index_url,
            torch_version,
            torchvision_version,
            torchaudio_version,
        ]
    )

    print(
        "✅ Locked PyTorch family installed."
    )

def install_locked_packages(
    lock,
):

    frontend = (
        lock[
            "comfyui"
        ][
            "frontend"
        ]
    )

    templates = (
        lock[
            "comfyui"
        ][
            "workflow_templates"
        ]
    )

    docs = (
        lock[
            "comfyui"
        ][
            "embedded_docs"
        ]
    )

    kitchen = (
        lock[
            "comfyui"
        ][
            "comfy_kitchen"
        ]
    )

    aimdo = (
        lock[
            "comfyui"
        ][
            "comfy_aimdo"
        ]
    )

    runtime = (
        lock[
            "python_runtime"
        ]
    )

    packages = [

        (
            frontend[
                "package"
            ],
            frontend[
                "version"
            ],
        ),

        (
            templates[
                "package"
            ],
            templates[
                "version"
            ],
        ),

        (
            docs[
                "package"
            ],
            docs[
                "version"
            ],
        ),

        (
            kitchen[
                "package"
            ],
            kitchen[
                "version"
            ],
        ),

        (
            aimdo[
                "package"
            ],
            aimdo[
                "version"
            ],
        ),

        (
            "torchsde",
            runtime[
                "torchsde"
            ],
        ),

        (
            "spandrel",
            runtime[
                "spandrel"
            ],
        ),

        (
            "av",
            runtime[
                "av"
            ],
        ),

        (
            "gguf",
            runtime[
                "gguf"
            ],
        ),
    ]

    for (
        package,
        version,
    ) in packages:

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


def verify_runtime(
    lock,
):

    import torch

    runtime = (
        lock[
            "python_runtime"
        ]
    )

    expected_torch = (
        runtime[
            "torch"
        ]
    )

    expected_torchvision = (
        runtime[
            "torchvision"
        ]
    )

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
            "CUDA is unavailable. "
            "Turn the Kaggle GPU ON."
        )

    if (
        torch.__version__
        != expected_torch
    ):

        raise RuntimeError(
            "Torch version mismatch.\n"
            f"Expected: {expected_torch}\n"
            f"Actual:   {torch.__version__}"
        )

    try:

        import torchvision

        actual_torchvision = (
            torchvision.__version__
        )

    except Exception as error:

        raise RuntimeError(
            "Could not import torchvision.\n"
            f"{error}"
        ) from error

    if (
        actual_torchvision
        != expected_torchvision
    ):

        raise RuntimeError(
            "Torchvision version mismatch.\n"
            f"Expected: {expected_torchvision}\n"
            f"Actual:   {actual_torchvision}"
        )

    for index in range(
        torch.cuda.device_count()
    ):

        print(
            f"GPU {index}: "
            f"{torch.cuda.get_device_name(index)}"
        )


def verify_locked_packages(
    lock,
):

    comfy = lock[
        "comfyui"
    ]

    runtime = lock[
        "python_runtime"
    ]

    expected = {

        comfy[
            "frontend"
        ][
            "package"
        ]:
            comfy[
                "frontend"
            ][
                "version"
            ],

        comfy[
            "workflow_templates"
        ][
            "package"
        ]:
            comfy[
                "workflow_templates"
            ][
                "version"
            ],

        comfy[
            "embedded_docs"
        ][
            "package"
        ]:
            comfy[
                "embedded_docs"
            ][
                "version"
            ],

        comfy[
            "comfy_kitchen"
        ][
            "package"
        ]:
            comfy[
                "comfy_kitchen"
            ][
                "version"
            ],

        comfy[
            "comfy_aimdo"
        ][
            "package"
        ]:
            comfy[
                "comfy_aimdo"
            ][
                "version"
            ],

        "torchsde":
            runtime[
                "torchsde"
            ],

        "spandrel":
            runtime[
                "spandrel"
            ],

        "av":
            runtime[
                "av"
            ],

        "gguf":
            runtime[
                "gguf"
            ],
    }

    print()
    print(
        "=" * 80
    )

    print(
        "LOCKED PYTHON PACKAGES"
    )

    print(
        "=" * 80
    )

    for (
        package,
        expected_version,
    ) in expected.items():

        try:

            actual = (
                importlib.metadata
                .version(
                    package
                )
            )

        except importlib.metadata.PackageNotFoundError:

            raise RuntimeError(
                f"Missing required package: "
                f"{package}"
            )

        if (
            actual
            != expected_version
        ):

            raise RuntimeError(
                f"{package} mismatch.\n"
                f"Expected: {expected_version}\n"
                f"Actual:   {actual}"
            )

        print(
            f"✅ {package}=={actual}"
        )


def install_comfyui(
    lock,
):

    comfy = lock[
        "comfyui"
    ]

    sync_repo(
        comfy[
            "repository"
        ],
        COMFY,
        comfy[
            "commit"
        ],
    )


def install_custom_nodes(
    lock,
):

    nodes = lock[
        "custom_nodes"
    ]

    for name, spec in (
        nodes.items()
    ):

        sync_repo(
            spec[
                "repository"
            ],
            CUSTOM / name,
            spec[
                "commit"
            ],
        )


def install_node_requirements():

    install_requirements(
        COMFY
        / "requirements.txt"
    )

    for name in (
        "ComfyUI-LTXVideo",
        "ComfyUI-KJNodes",
        "ComfyUI-VideoHelperSuite",
        "ComfyUI-GGUF",
    ):

        install_requirements(
            CUSTOM
            / name
            / "requirements.txt"
        )


def link_one_model(
    source,
    target,
):

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
        source.resolve()
    )

    print(
        f"✅ {target}"
    )


def link_models(
    lock,
):

    sources = {}

    for name, spec in (
        lock[
            "models"
        ].items()
    ):

        source = (
            Path(
                spec[
                    "dataset"
                ]
            )
            / spec[
                "filename"
            ]
        )

        if not source.exists():

            raise FileNotFoundError(
                f"{name} model not found:\n"
                f"{source}"
            )

        sources[
            name
        ] = source

    model_specs = (
        lock[
            "models"
        ]
    )

    # ---------------------------------------------------------
    # LTX 13B Q4
    # ---------------------------------------------------------

    link_one_model(
        sources[
            "ltx_q4"
        ],
        COMFY
        / model_specs[
            "ltx_q4"
        ][
            "target"
        ],
    )

    # ---------------------------------------------------------
    # T5 Q4
    # ---------------------------------------------------------

    link_one_model(
        sources[
            "t5_q4"
        ],
        COMFY
        / model_specs[
            "t5_q4"
        ][
            "target"
        ],
    )

    # ---------------------------------------------------------
    # VAE
    # ---------------------------------------------------------

    link_one_model(
        sources[
            "vae"
        ],
        COMFY
        / model_specs[
            "vae"
        ][
            "target"
        ],
    )

    # ---------------------------------------------------------
    # IC-LoRA
    # ---------------------------------------------------------

    link_one_model(
        sources[
            "ic_lora"
        ],
        COMFY
        / model_specs[
            "ic_lora"
        ][
            "target"
        ],
    )

    # ---------------------------------------------------------
    # Canonical modern spatial latent upscaler
    # ---------------------------------------------------------

    spatial = sources[
        "spatial_upscaler"
    ]

    canonical_target = (
        COMFY
        / model_specs[
            "spatial_upscaler"
        ][
            "target"
        ]
    )

    link_one_model(
        spatial,
        canonical_target,
    )

    # ---------------------------------------------------------
    # Compatibility discovery paths.
    #
    # Keep these because different LTX/ComfyUI node
    # implementations may discover the same latent model
    # through different model-category registries.
    # ---------------------------------------------------------

    extra_dirs = [

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

    for directory in (
        extra_dirs
    ):

        link_one_model(
            spatial,
            directory
            / spatial.name,
        )


def write_extra_model_paths(
    lock,
):

    models = lock[
        "models"
    ]

    content = f"""
ltx_project:
  is_default: true

  diffusion_models: |
    {models["ltx_q4"]["dataset"]}

  unet: |
    {models["ltx_q4"]["dataset"]}

  text_encoders: |
    {models["t5_q4"]["dataset"]}

  clip: |
    {models["t5_q4"]["dataset"]}

  vae: |
    {models["vae"]["dataset"]}

  loras: |
    {models["ic_lora"]["dataset"]}

  upscale_models: |
    {models["spatial_upscaler"]["dataset"]}

  latent_upscale_models: |
    {models["spatial_upscaler"]["dataset"]}
"""

    (
        COMFY
        / "extra_model_paths.yaml"
    ).write_text(
        content,
        encoding="utf-8",
    )


def build_compatibility(
    lock,
):

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


def verify_git_stack(
    lock,
):

    print()
    print(
        "=" * 80
    )

    print(
        "GIT STACK VERIFICATION"
    )

    print(
        "=" * 80
    )

    expected_comfy = (
        lock[
            "comfyui"
        ][
            "commit"
        ]
    )

    actual_comfy = git_current(
        COMFY
    )

    if (
        actual_comfy
        != expected_comfy
    ):

        raise RuntimeError(
            "ComfyUI revision mismatch."
        )

    print(
        f"✅ ComfyUI {actual_comfy}"
    )

    for name, spec in (
        lock[
            "custom_nodes"
        ].items()
    ):

        actual = git_current(
            CUSTOM
            / name
        )

        if (
            actual
            != spec[
                "commit"
            ]
        ):

            raise RuntimeError(
                f"{name} revision mismatch."
            )

        print(
            f"✅ {name} {actual}"
        )

def verify_ltxvideo_pyramid_patch():

    runtime = (
        COMFY
        / "custom_nodes"
        / "ComfyUI-LTXVideo"
    )

    pyramid = (
        runtime
        / "pyramid_blending.py"
    )

    if not runtime.exists():
        raise RuntimeError(
            "ComfyUI-LTXVideo runtime directory missing:\n"
            f"{runtime}"
        )

    if not (
        runtime / ".git"
    ).exists():
        raise RuntimeError(
            "ComfyUI-LTXVideo is not a Git checkout:\n"
            f"{runtime}"
        )

    text = pyramid.read_text(
        encoding="utf-8"
    )

    if "pad = F.pad" not in text:
        raise RuntimeError(
            "LTXVideo Kornia compatibility patch "
            "was not applied."
        )

    import re

    match = re.search(
        r"from\s+kornia\.geometry\.transform\.pyramid\s+import\s*\("
        r"(?P<body>.*?)\)",
        text,
        flags=re.DOTALL,
    )

    if not match:
        raise RuntimeError(
            "Could not find the Kornia pyramid import block."
        )

    if re.search(
        r"^\s*pad\s*,?\s*$",
        match.group("body"),
        flags=re.MULTILINE,
    ):
        raise RuntimeError(
            "Old Kornia `pad` import still exists."
        )

    print(
        "✅ LTXVideo pyramid compatibility patch verified."
    )

def verify_ltxvideo_import():
    """
    Import the actual patched ComfyUI-LTXVideo package
    before ComfyUI starts.

    The package contains relative imports, so it is
    registered in sys.modules before exec_module().

    The actual ComfyUI root is also added to sys.path so
    imports are resolved the same way they are during
    normal ComfyUI startup.
    """

    import importlib.util
    import sys

    package_dir = (
        COMFY
        / "custom_nodes"
        / "ComfyUI-LTXVideo"
    )

    init_file = (
        package_dir
        / "__init__.py"
    )

    # ---------------------------------------------------------
    # Basic package validation
    # ---------------------------------------------------------

    if not package_dir.exists():
        raise RuntimeError(
            "ComfyUI-LTXVideo directory not found:\n"
            f"{package_dir}"
        )

    if not (
        package_dir
        / ".git"
    ).exists():
        raise RuntimeError(
            "ComfyUI-LTXVideo is not a Git checkout:\n"
            f"{package_dir}"
        )

    if not init_file.exists():
        raise RuntimeError(
            "ComfyUI-LTXVideo __init__.py "
            "not found:\n"
            f"{init_file}"
        )

    # ---------------------------------------------------------
    # Make the real ComfyUI installation importable
    # ---------------------------------------------------------

    comfy_root = str(COMFY)

    if comfy_root not in sys.path:
        sys.path.insert(
            0,
            comfy_root,
        )

    # ---------------------------------------------------------
    # Create package import specification
    # ---------------------------------------------------------

    package_name = (
        "LTXVideoPreflight"
    )

    spec = (
        importlib.util
        .spec_from_file_location(
            package_name,
            str(init_file),
            submodule_search_locations=[
                str(package_dir)
            ],
        )
    )

    if (
        spec is None
        or spec.loader is None
    ):
        raise RuntimeError(
            "Could not create an import "
            "spec for ComfyUI-LTXVideo."
        )

    module = (
        importlib.util
        .module_from_spec(
            spec
        )
    )

    # ---------------------------------------------------------
    # Register package BEFORE executing __init__.py
    #
    # This is required for imports such as:
    #     from .audio_only import ...
    # ---------------------------------------------------------

    sys.modules[
        package_name
    ] = module

    try:

        spec.loader.exec_module(
            module
        )

    except Exception as error:

        raise RuntimeError(
            "Patched ComfyUI-LTXVideo "
            "failed the preflight import:\n"
            f"{error}"
        ) from error

    finally:

        # Remove only our temporary top-level
        # preflight package registration.
        sys.modules.pop(
            package_name,
            None
        )

    print(
        "✅ ComfyUI-LTXVideo "
        "import verified."
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

    lock = load_lock()

    # ---------------------------------------------------------
    # Verify the Kaggle runtime before changing anything.
    # ---------------------------------------------------------

    verify_runtime(
        lock
    )

    # ---------------------------------------------------------
    # Install exact locked ComfyUI revision.
    # ---------------------------------------------------------

    install_comfyui(
        lock
    )

    # ---------------------------------------------------------
    # Install exact locked custom-node revisions.
    # ---------------------------------------------------------

    install_custom_nodes(
        lock
    )

    # ---------------------------------------------------------
    # Install requirements from those exact repositories.
    # ---------------------------------------------------------

    install_node_requirements()

    # ---------------------------------------------------------
    # Install exact packages from central lock.
    # ---------------------------------------------------------

    install_locked_packages(
        lock
    )

    # ---------------------------------------------------------
    # Verify runtime after package installation.
    # ---------------------------------------------------------

    verify_runtime(
        lock
    )

    verify_locked_packages(
        lock
    )

    # ---------------------------------------------------------
    # Build the project-owned LTX 0.9.8 compatibility layer.
    # ---------------------------------------------------------

    build_compatibility(
        lock
    )

    verify_ltxvideo_pyramid_patch()

    verify_ltxvideo_import()

    # ---------------------------------------------------------
    # Link all required model assets.
    # ---------------------------------------------------------

    link_models(
        lock
    )

    # ---------------------------------------------------------
    # Tell ComfyUI about the Kaggle dataset locations.
    # ---------------------------------------------------------

    write_extra_model_paths(
        lock
    )

    # ---------------------------------------------------------
    # Final Git revision verification.
    # ---------------------------------------------------------

    verify_git_stack(
        lock
    )

    print()
    print(
        "=" * 80
    )

    print(
        "✅ MODERN STACK BOOTSTRAP COMPLETE"
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
        "Frontend:",
        importlib.metadata.version(
            lock[
                "comfyui"
            ][
                "frontend"
            ][
                "package"
            ]
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
        "Compatibility package:",
        CUSTOM
        / lock[
            "legacy_ltx_098_compat"
        ][
            "runtime_package"
        ],
    )

    print(
        "Spatial model:",
        lock[
            "models"
        ][
            "spatial_upscaler"
        ][
            "filename"
        ],
    )

    print(
        "Frontend override: NONE"
    )

    print(
        "All versions are controlled by "
        "kaggle/compatibility_lock.yaml"
    )


if __name__ == "__main__":
    main()
