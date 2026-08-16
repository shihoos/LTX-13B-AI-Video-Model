from pathlib import Path
import shutil
import subprocess
import sys


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT = Path("/kaggle/working/LTX-13B-AI-Video-Model")

COMFY = PROJECT / "ComfyUI"

CUSTOM = COMFY / "custom_nodes"

VENV = PROJECT / ".venv"


# ============================================================
# VERSION-PINNED CORE
# ============================================================

# ComfyUI state from the LTXV 0.9.8 timeframe.
COMFY_REPO = "https://github.com/Comfy-Org/ComfyUI.git"
COMFY_COMMIT = None

# LTXVideo state corresponding to the 0.9.8 release.
LTX_REPO = "https://github.com/Lightricks/ComfyUI-LTXVideo.git"
LTX_COMMIT = None


# ============================================================
# CUSTOM NODES
# ============================================================

NODES = {
    "ComfyUI-GGUF": {
        "url": "https://github.com/city96/ComfyUI-GGUF.git",
        "commit": None,
    },
    "ComfyUI-LTXVideo": {
        "url": LTX_REPO,
        "commit": LTX_COMMIT,
    },
    "ComfyUI-VideoHelperSuite": {
        "url": "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git",
        "commit": None,
    },
    "rgthree-comfy": {
        "url": "https://github.com/rgthree/rgthree-comfy.git",
        "commit": None,
    },
}


# ============================================================
# REQUIRED TORCH ENVIRONMENT
# ============================================================

EXPECTED_TORCH_PREFIX = "2.10.0+cu128"


# ============================================================
# COMMAND HELPERS
# ============================================================

def run(command, cwd=None):
    print("\n$", " ".join(map(str, command)))

    subprocess.run(
        command,
        cwd=cwd,
        check=True,
    )


def clone_repo(
    url: str,
    destination: Path,
    commit: str | None = None,
):
    """
    Clone a repository.

    An existing incomplete directory is removed automatically.
    """

    if destination.exists():

        git_dir = destination / ".git"

        # A valid repository already exists.
        if git_dir.exists():

            print(
                f"Repository already exists: {destination}"
            )

            if commit:
                run(
                    [
                        "git",
                        "fetch",
                        "--depth",
                        "50",
                        "origin",
                    ],
                    cwd=destination,
                )

                run(
                    [
                        "git",
                        "checkout",
                        commit,
                    ],
                    cwd=destination,
                )

            return

        # Existing directory is not a valid Git repository.
        print(
            f"Removing incomplete directory: {destination}"
        )

        shutil.rmtree(destination)

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            url,
            str(destination),
        ]
    )

    if commit:

        run(
            [
                "git",
                "fetch",
                "--depth",
                "50",
                "origin",
            ],
            cwd=destination,
        )

        run(
            [
                "git",
                "checkout",
                commit,
            ],
            cwd=destination,
        )


# ============================================================
# TORCH VERIFICATION
# ============================================================

def verify_torch():

    try:

        import torch

    except ImportError:

        raise RuntimeError(
            "PyTorch is not installed."
        )

    print("\n" + "=" * 60)
    print("PyTorch verification")
    print("=" * 60)

    print("Torch:", torch.__version__)
    print("CUDA:", torch.version.cuda)
    print(
        "CUDA available:",
        torch.cuda.is_available(),
    )
    print(
        "GPU count:",
        torch.cuda.device_count(),
    )

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA is not available. "
            "Do not continue."
        )

    if not torch.__version__.startswith(
        EXPECTED_TORCH_PREFIX
    ):

        raise RuntimeError(
            "Unexpected PyTorch version.\n"
            f"Expected: {EXPECTED_TORCH_PREFIX}\n"
            f"Found:    {torch.__version__}\n\n"
            "This bootstrap intentionally refuses "
            "to modify the existing Torch installation."
        )

    if torch.cuda.device_count() < 2:

        raise RuntimeError(
            "Expected two CUDA GPUs."
        )

    for index in range(
        torch.cuda.device_count()
    ):

        print(
            f"GPU {index}: "
            f"{torch.cuda.get_device_name(index)}"
        )


# ============================================================
# CORE COMFYUI DEPENDENCIES
# ============================================================

def install_comfy_dependencies():

    requirements = (
        COMFY /
        "requirements.txt"
    )

    if not requirements.exists():

        raise FileNotFoundError(
            f"Missing ComfyUI requirements file: "
            f"{requirements}"
        )

    print(
        "\nInstalling ComfyUI dependencies "
        "without replacing PyTorch..."
    )

    filtered = (
        PROJECT /
        "comfy_requirements_no_torch.txt"
    )

    lines = requirements.read_text(
        encoding="utf-8"
    ).splitlines()

    excluded = {
        "torch",
        "torchvision",
        "torchaudio",
    }

    kept = []

    for line in lines:

        stripped = line.strip()

        if not stripped:
            continue

        if stripped.startswith("#"):
            continue

        package_name = (
            stripped.split("=")[0]
            .split(">")[0]
            .split("<")[0]
            .strip()
        )

        if package_name in excluded:
            continue

        kept.append(stripped)

    filtered.write_text(
        "\n".join(kept) + "\n",
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
            str(filtered),
        ]
    )


# ============================================================
# CUSTOM NODE DEPENDENCIES
# ============================================================

def install_node_dependencies():

    for name in NODES:

        node_dir = CUSTOM / name

        requirements = (
            node_dir /
            "requirements.txt"
        )

        if not requirements.exists():

            print(
                f"No requirements.txt for {name}; "
                "skipping."
            )

            continue

        print(
            f"\nInstalling dependencies for {name}"
        )

        run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "-r",
                str(requirements),
            ]
        )


# ============================================================
# VALIDATE INSTALLATION
# ============================================================

def validate_comfy():

    print(
        "\n" + "=" * 60
    )

    print(
        "Validating ComfyUI installation"
    )

    print(
        "=" * 60
    )

    main_py = COMFY / "main.py"

    if not main_py.exists():

        raise RuntimeError(
            f"ComfyUI main.py was not found: "
            f"{main_py}"
        )

    requirements = (
        COMFY /
        "requirements.txt"
    )

    if not requirements.exists():

        raise RuntimeError(
            "ComfyUI requirements.txt is missing."
        )

    print(
        "ComfyUI:",
        COMFY,
    )

    print(
        "main.py: OK"
    )

    print(
        "requirements.txt: OK"
    )

    print(
        "Custom nodes:"
    )

    for name in NODES:

        node_dir = CUSTOM / name

        if not node_dir.exists():

            raise RuntimeError(
                f"Missing custom node: {name}"
            )

        print(
            f"  {name}: OK"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("LTX-13B AI Video Model Bootstrap")
    print("=" * 60)

    PROJECT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 1. Never alter the known-good Torch environment.
    verify_torch()

    # 2. Install the pinned ComfyUI core.
    clone_repo(
        COMFY_REPO,
        COMFY,
        COMFY_COMMIT,
    )

    # 3. Create custom node directory.
    CUSTOM.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 4. Install custom nodes.
    for name, info in NODES.items():

        clone_repo(
            info["url"],
            CUSTOM / name,
            info["commit"],
        )

    # 5. Install ComfyUI dependencies,
    #    explicitly excluding Torch packages.
    install_comfy_dependencies()

    # 6. Install custom-node dependencies.
    install_node_dependencies()

    # 7. Validate everything.
    validate_comfy()

    # 8. Verify Torch AGAIN because pip is not allowed
    #    to change it.
    verify_torch()

    print(
        "\n" + "=" * 60
    )

    print(
        "✅ BOOTSTRAP COMPLETED SUCCESSFULLY"
    )

    print(
        "=" * 60
    )


if __name__ == "__main__":
    main()
