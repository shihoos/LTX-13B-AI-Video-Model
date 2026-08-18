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
            f"Compatibility lock not found:\n"
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
            "Git revision mismatch:\n"
            f"Path:     {path}\n"
            f"Expected: {commit}\n"
            f"Actual:   {actual}"
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
            .split("==", 1)[0]
            .split(">=", 1)[0]
            .split("<=", 1)[0]
            .split("~=", 1)[0]
            .split(">", 1)[0]
            .split("<", 1)[0]
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

    # Base model.
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

    # T5.
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

    # VAE.
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

    # IC-LoRA.
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

    # Canonical modern latent-upscale location.
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

    # Additional compatibility discovery paths
    # used by ComfyUI/LTX versions.
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

    legacy = lock[
        "legacy_ltx_098_compat"
    ]

    expected_commit = (
        legacy[
            "commit"
        ]
    )

    # The compatibility builder owns the actual source download.
    os.environ[
        "LTX_LEGACY_COMMIT"
    ] = expected_commit

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
            CUSTOM / name
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

    verify_runtime(
        lock
    )

    install_comfyui(
        lock
    )

    install_custom_nodes(
        lock
    )

    install_node_requirements()

    install_locked_packages(
        lock
    )

    verify_runtime(
        lock
    )

    verify_locked_packages(
        lock
    )

    build_compatibility(
        lock
    )

    link_models(
        lock
    )

    write_extra_model_paths(
        lock
    )

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
        "Frontend version:",
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
        "Official current LTXVideo:",
        git_current(
            CUSTOM
            / "ComfyUI-LTXVideo"
        ),
    )

    print(
        "Modern compatibility package:",
        CUSTOM
        / "LTX098ModernCompat",
    )

    print(
        "No @latest frontend override is used."
    )


if __name__ == "__main__":
    main()
