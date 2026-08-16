from pathlib import Path
import shutil
import subprocess
import sys


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT = Path(
    "/kaggle/working/LTX-13B-AI-Video-Model"
)

COMFY = PROJECT / "ComfyUI"

CUSTOM = COMFY / "custom_nodes"


# ============================================================
# VERSION-PINNED ENVIRONMENT
# ============================================================

# This is the ComfyUI version currently confirmed working
# in your Kaggle environment.
COMFY_REPO = (
    "https://github.com/Comfy-Org/ComfyUI.git"
)

COMFY_COMMIT = "38d049382533c6662d815b08ca3395e96cca9f57"


# Verified LTXVideo revision containing:
# - LTXVLatentUpsamplerModelLoader
# - LTXVLatentUpsampler
# - STGGuiderAdvanced
# - LTXVBaseSampler
# - LTXVAdainLatent
LTXVIDEO_COMMIT = (
    "ee11be3ce229c3afd5fadf8a1258eb8b84af33b1"
)


# ============================================================
# CUSTOM NODES
# ============================================================

NODES = {

    "ComfyUI-GGUF": {
        "url":
            "https://github.com/city96/ComfyUI-GGUF.git",
        "commit": None,
    },

    "ComfyUI-LTXVideo": {
        "url":
            "https://github.com/Lightricks/ComfyUI-LTXVideo.git",
        "commit": LTXVIDEO_COMMIT,
    },

    "ComfyUI-VideoHelperSuite": {
        "url":
            "https://github.com/Kosinkadink/"
            "ComfyUI-VideoHelperSuite.git",
        "commit": None,
    },

    "rgthree-comfy": {
        "url":
            "https://github.com/rgthree/rgthree-comfy.git",
        "commit": None,
    },

    "ComfyUI-KJNodes": {
        "url":
            "https://github.com/kijai/ComfyUI-KJNodes.git",
        "commit": None,
    },
}


# ============================================================
# REQUIRED TORCH ENVIRONMENT
# ============================================================

EXPECTED_TORCH_PREFIX = "2.10.0+cu128"


# ============================================================
# PACKAGES THAT MUST NEVER BE REPLACED
# ============================================================

FORBIDDEN_PACKAGES = {
    "torch",
    "torchvision",
    "torchaudio",
}


# ============================================================
# COMMAND HELPER
# ============================================================

def run(command, cwd=None):
    print(
        "\n$ " +
        " ".join(
            map(str, command)
        )
    )

    subprocess.run(
        command,
        cwd=cwd,
        check=True,
    )


# ============================================================
# GIT HELPERS
# ============================================================

def get_current_commit(
    repository: Path,
):
    result = subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "rev-parse",
            "HEAD",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    return result.stdout.strip()


def checkout_pinned_commit(
    repository: Path,
    commit: str,
):
    """
    Deterministically switch an existing repository to
    an exact revision.

    No shallow-depth assumptions are used.
    """

    print(
        f"\nPinning repository to: {commit}"
    )

    run(
        [
            "git",
            "fetch",
            "--all",
            "--tags",
            "--prune",
        ],
        cwd=repository,
    )

    run(
        [
            "git",
            "checkout",
            "--force",
            commit,
        ],
        cwd=repository,
    )

    actual = get_current_commit(
        repository
    )

    if actual != commit:
        raise RuntimeError(
            "\nRepository pin verification failed.\n"
            f"Expected: {commit}\n"
            f"Found:    {actual}"
        )

    print(
        f"✅ Exact revision active: {actual}"
    )


def clone_repo(
    url: str,
    destination: Path,
    commit: str | None = None,
):
    """
    Clone a repository or update an existing repository.

    When a commit is supplied, the exact commit is checked
    out using full Git history. No --depth-based checkout
    is used for pinned repositories.
    """

    # --------------------------------------------------------
    # Existing directory
    # --------------------------------------------------------

    if destination.exists():

        git_dir = (
            destination / ".git"
        )

        if git_dir.exists():

            print(
                f"\nRepository already exists:"
                f" {destination}"
            )

            # A pinned repository must be moved to the
            # requested exact revision.
            if commit:
                checkout_pinned_commit(
                    destination,
                    commit,
                )

            return

        # Directory exists but is not a Git repository.
        print(
            "\nRemoving incomplete repository:"
            f" {destination}"
        )

        shutil.rmtree(
            destination
        )

    # --------------------------------------------------------
    # Fresh clone
    # --------------------------------------------------------

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"\nCloning repository:\n{url}"
    )

    # Full clone deliberately used here because we may need
    # an historical pinned commit.
    run(
        [
            "git",
            "clone",
            url,
            str(destination),
        ]
    )

    # --------------------------------------------------------
    # Pin exact revision
    # --------------------------------------------------------

    if commit:
        checkout_pinned_commit(
            destination,
            commit,
        )


# ============================================================
# PYTORCH VERIFICATION
# ============================================================

def verify_torch():

    try:
        import torch

    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is not installed."
        ) from exc

    print(
        "\n" + "=" * 70
    )
    print(
        "PYTORCH / CUDA VERIFICATION"
    )
    print(
        "=" * 70
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

    print(
        "GPU count:",
        torch.cuda.device_count(),
    )

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA is unavailable."
        )

    if not torch.__version__.startswith(
        EXPECTED_TORCH_PREFIX
    ):

        raise RuntimeError(
            "Unexpected PyTorch version.\n"
            f"Expected prefix: "
            f"{EXPECTED_TORCH_PREFIX}\n"
            f"Found: {torch.__version__}"
        )

    if torch.cuda.device_count() < 2:

        raise RuntimeError(
            "This project requires two CUDA GPUs."
        )

    for index in range(
        torch.cuda.device_count()
    ):

        print(
            f"GPU {index}: "
            f"{torch.cuda.get_device_name(index)}"
        )

    print(
        "\n✅ PyTorch environment is correct."
    )


# ============================================================
# REQUIREMENT FILTERING
# ============================================================

def filter_requirements(
    source: Path,
    destination: Path,
):
    """
    Remove packages that are not allowed to replace
    the known-good Torch environment.
    """

    filtered = []

    for raw_line in source.read_text(
        encoding="utf-8"
    ).splitlines():

        line = raw_line.strip()

        if not line:
            continue

        if line.startswith("#"):
            continue

        # Remove inline comments.
        line = line.split(
            "#",
            1
        )[0].strip()

        if not line:
            continue

        package_name = (
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

        if package_name in FORBIDDEN_PACKAGES:
            continue

        filtered.append(line)

    destination.write_text(
        "\n".join(filtered) + "\n",
        encoding="utf-8",
    )


# ============================================================
# COMFYUI DEPENDENCIES
# ============================================================

def install_comfy_dependencies():

    requirements = (
        COMFY /
        "requirements.txt"
    )

    if not requirements.exists():

        raise FileNotFoundError(
            f"Missing requirements file:\n"
            f"{requirements}"
        )

    filtered = (
        PROJECT /
        ".comfy_requirements_safe.txt"
    )

    filter_requirements(
        requirements,
        filtered,
    )

    print(
        "\nInstalling ComfyUI dependencies "
        "without modifying Torch..."
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


# ============================================================
# CUSTOM NODE DEPENDENCIES
# ============================================================

def install_node_dependencies():

    print(
        "\n" + "=" * 70
    )
    print(
        "CUSTOM NODE DEPENDENCIES"
    )
    print(
        "=" * 70
    )

    for name in NODES:

        node_dir = (
            CUSTOM /
            name
        )

        requirements = (
            node_dir /
            "requirements.txt"
        )

        if not requirements.exists():

            print(
                f"{name}: "
                "no requirements.txt"
            )

            continue

        filtered = (
            PROJECT /
            f".{name}_requirements_safe.txt"
        )

        filter_requirements(
            requirements,
            filtered,
        )

        if not filtered.read_text(
            encoding="utf-8"
        ).strip():

            print(
                f"{name}: "
                "no additional safe dependencies"
            )

            try:
                filtered.unlink()
            except OSError:
                pass

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
                str(filtered),
            ]
        )

        try:
            filtered.unlink()
        except OSError:
            pass


# ============================================================
# VALIDATE REPOSITORIES
# ============================================================

def validate_repositories():

    print(
        "\n" + "=" * 70
    )
    print(
        "VALIDATING REPOSITORIES"
    )
    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # ComfyUI
    # --------------------------------------------------------

    if not (
        COMFY /
        "main.py"
    ).exists():

        raise RuntimeError(
            "ComfyUI main.py is missing."
        )

    comfy_revision = (
        get_current_commit(
            COMFY
        )
    )

    print(
        "ComfyUI:",
        comfy_revision,
    )

    # --------------------------------------------------------
    # Custom nodes
    # --------------------------------------------------------

    for name, info in NODES.items():

        node_dir = (
            CUSTOM /
            name
        )

        if not node_dir.exists():

            raise RuntimeError(
                f"Missing custom node: {name}"
            )

        revision = (
            get_current_commit(
                node_dir
            )
        )

        requested = info["commit"]

        if requested:

            if revision != requested:

                raise RuntimeError(
                    f"\n{name} revision mismatch.\n"
                    f"Expected: {requested}\n"
                    f"Found:    {revision}"
                )

            print(
                f"{name}: {revision} ✅"
            )

        else:

            print(
                f"{name}: {revision}"
            )


# ============================================================
# VALIDATE COMFYUI FILES
# ============================================================

def validate_comfy():

    print(
        "\n" + "=" * 70
    )
    print(
        "VALIDATING COMFYUI INSTALLATION"
    )
    print(
        "=" * 70
    )

    required_files = [
        COMFY / "main.py",
        COMFY / "requirements.txt",
        CUSTOM / "ComfyUI-GGUF",
        CUSTOM / "ComfyUI-LTXVideo",
        CUSTOM / "ComfyUI-VideoHelperSuite",
        CUSTOM / "rgthree-comfy",
        CUSTOM / "ComfyUI-KJNodes",
    ]

    for path in required_files:

        if not path.exists():

            raise RuntimeError(
                f"Missing required path:\n{path}"
            )

        print(
            f"✅ {path}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 70
    )

    print(
        "LTX-13B AI VIDEO MODEL BOOTSTRAP"
    )

    print(
        "=" * 70
    )

    PROJECT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # 1. Verify the known-good Torch environment
    # --------------------------------------------------------

    verify_torch()

    # --------------------------------------------------------
    # 2. Install / synchronize pinned ComfyUI
    # --------------------------------------------------------

    print(
        "\nInstalling / synchronizing ComfyUI..."
    )

    clone_repo(
        COMFY_REPO,
        COMFY,
        COMFY_COMMIT,
    )

    # --------------------------------------------------------
    # 3. Create custom-node directory
    # --------------------------------------------------------

    CUSTOM.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # 4. Install / synchronize custom nodes
    # --------------------------------------------------------

    print(
        "\nInstalling / synchronizing custom nodes..."
    )

    for name, info in NODES.items():

        clone_repo(
            info["url"],
            CUSTOM / name,
            info["commit"],
        )

    # --------------------------------------------------------
    # 5. Install ComfyUI dependencies safely
    # --------------------------------------------------------

    install_comfy_dependencies()

    # --------------------------------------------------------
    # 6. Install custom-node dependencies safely
    # --------------------------------------------------------

    install_node_dependencies()

    # --------------------------------------------------------
    # 7. Validate filesystem
    # --------------------------------------------------------

    validate_comfy()

    # --------------------------------------------------------
    # 8. Validate exact Git revisions
    # --------------------------------------------------------

    validate_repositories()

    # --------------------------------------------------------
    # 9. Verify Torch again
    # --------------------------------------------------------

    verify_torch()

    print(
        "\n" + "=" * 70
    )

    print(
        "✅ BOOTSTRAP COMPLETED SUCCESSFULLY"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()
