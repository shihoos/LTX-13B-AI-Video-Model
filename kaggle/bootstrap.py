from pathlib import Path
import re
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
# REPOSITORY CONFIGURATION
# ============================================================

COMFY_REPO = (
    "https://github.com/Comfy-Org/ComfyUI.git"
)

# None = follow the current default branch of ComfyUI.
# This is intentional while testing modern ComfyUI.
COMFY_COMMIT = "483b3e62e00624fc52da8ad67e88f863abe975d2"

# Keep the proven LTX 0.9.8-era node implementation.
LTXVIDEO_COMMIT = (
    "ee11be3ce229c3afd5fadf8a1258eb8b84af33b1"
)

# Keep the verified KJNodes 1.0.8 source.
KJNODES_COMMIT = (
    "89fb17ae84951995ab1eee19e205ea48ceed27c9"
)

NODES = {
    "ComfyUI-GGUF": {
        "url": "https://github.com/city96/ComfyUI-GGUF.git",
        "commit": None,
    },
    "ComfyUI-LTXVideo": {
        "url": "https://github.com/Lightricks/ComfyUI-LTXVideo.git",
        "commit": LTXVIDEO_COMMIT,
    },
    "ComfyUI-VideoHelperSuite": {
        "url": (
            "https://github.com/Kosinkadink/"
            "ComfyUI-VideoHelperSuite.git"
        ),
        "commit": None,
    },
    "rgthree-comfy": {
        "url": "https://github.com/rgthree/rgthree-comfy.git",
        "commit": None,
    },
    "ComfyUI-KJNodes": {
        "url": "https://github.com/kijai/ComfyUI-KJNodes.git",
        "commit": KJNODES_COMMIT,
    },
}


# ============================================================
# RUNTIME SAFETY
# ============================================================

EXPECTED_TORCH_PREFIX = "2.10.0+cu128"

FORBIDDEN_PACKAGES = {
    "torch",
    "torchvision",
    "torchaudio",
}


# ============================================================
# COMMAND HELPERS
# ============================================================

def run(command, cwd=None):
    print(
        "\n$ " + " ".join(map(str, command))
    )
    subprocess.run(
        command,
        cwd=cwd,
        check=True,
    )


def git_output(repository: Path, args):
    result = subprocess.run(
        ["git", "-C", str(repository), *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def git_current_commit(repository: Path) -> str:
    return git_output(
        repository,
        ["rev-parse", "HEAD"],
    )


def repository_is_git(repository: Path) -> bool:
    return (repository / ".git").exists()


# ============================================================
# REPOSITORY SYNCHRONIZATION
# ============================================================

def get_default_branch(repository: Path) -> str:
    """
    Return origin's default branch name.
    Falls back to master, then main.
    """
    result = subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "remote",
            "show",
            "origin",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    match = re.search(
        r"HEAD branch:\s*(\S+)",
        result.stdout,
    )

    if match:
        return match.group(1)

    for branch in ("master", "main"):
        exists = subprocess.run(
            [
                "git",
                "-C",
                str(repository),
                "show-ref",
                "--verify",
                f"refs/remotes/origin/{branch}",
            ],
            capture_output=True,
            text=True,
        ).returncode == 0

        if exists:
            return branch

    raise RuntimeError(
        f"Could not determine default branch for {repository}"
    )


def checkout_pinned_commit(
    repository: Path,
    commit: str,
):
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

    actual = git_current_commit(repository)

    if actual != commit:
        raise RuntimeError(
            "\nRepository pin verification failed.\n"
            f"Expected: {commit}\n"
            f"Found:    {actual}"
        )

    print(
        f"✅ Exact revision active: {actual}"
    )


def update_to_latest(
    repository: Path,
):
    """
    Put an existing checkout on the remote default branch.
    This also handles a previous detached HEAD state.
    """
    run(
        [
            "git",
            "fetch",
            "origin",
            "--tags",
            "--prune",
        ],
        cwd=repository,
    )

    branch = get_default_branch(repository)

    run(
        [
            "git",
            "checkout",
            "--force",
            f"origin/{branch}",
        ],
        cwd=repository,
    )

    actual = git_current_commit(repository)

    print(
        f"✅ Latest {branch} revision active: {actual}"
    )


def clone_or_sync(
    url: str,
    destination: Path,
    commit: str | None = None,
):
    """
    commit=None:
        clone or update to the repository's default branch.

    commit=<SHA>:
        clone or force-checkout the exact SHA.
    """
    if destination.exists():

        if not repository_is_git(destination):
            print(
                f"\nRemoving incomplete repository: {destination}"
            )
            shutil.rmtree(destination)

        else:
            print(
                f"\nRepository already exists: {destination}"
            )

            if commit:
                checkout_pinned_commit(
                    destination,
                    commit,
                )
            else:
                update_to_latest(
                    destination,
                )

            return

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"\nCloning repository:\n{url}"
    )

    run(
        [
            "git",
            "clone",
            url,
            str(destination),
        ]
    )

    if commit:
        checkout_pinned_commit(
            destination,
            commit,
        )
    else:
        print(
            f"✅ Latest default branch active: "
            f"{git_current_commit(destination)}"
        )


# ============================================================
# TORCH / CUDA
# ============================================================

def verify_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is not installed."
        ) from exc

    print("\n" + "=" * 70)
    print("PYTORCH / CUDA VERIFICATION")
    print("=" * 70)

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
            "CUDA is unavailable."
        )

    if not torch.__version__.startswith(
        EXPECTED_TORCH_PREFIX
    ):
        raise RuntimeError(
            "Unexpected PyTorch version.\n"
            f"Expected prefix: {EXPECTED_TORCH_PREFIX}\n"
            f"Found: {torch.__version__}"
        )

    if torch.cuda.device_count() < 2:
        raise RuntimeError(
            "This project requires two CUDA GPUs."
        )

    for index in range(torch.cuda.device_count()):
        print(
            f"GPU {index}: "
            f"{torch.cuda.get_device_name(index)}"
        )

    print(
        "\n✅ PyTorch environment is correct."
    )


# ============================================================
# REQUIREMENTS
# ============================================================

def filter_requirements(
    source: Path,
    destination: Path,
):
    filtered = []

    for raw_line in source.read_text(
        encoding="utf-8"
    ).splitlines():

        line = raw_line.strip()

        if not line or line.startswith("#"):
            continue

        line = line.split("#", 1)[0].strip()

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


def install_requirements(
    source: Path,
    temporary_name: str,
):
    if not source.exists():
        return

    destination = PROJECT / temporary_name

    filter_requirements(
        source,
        destination,
    )

    if not destination.read_text(
        encoding="utf-8"
    ).strip():
        try:
            destination.unlink()
        except OSError:
            pass
        return

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


def install_comfy_dependencies():
    print(
        "\nInstalling ComfyUI dependencies "
        "without modifying Torch..."
    )

    install_requirements(
        COMFY / "requirements.txt",
        ".comfy_requirements_safe.txt",
    )


def install_node_dependencies():
    print("\n" + "=" * 70)
    print("CUSTOM NODE DEPENDENCIES")
    print("=" * 70)

    for name in NODES:

        requirements = (
            CUSTOM / name / "requirements.txt"
        )

        if not requirements.exists():
            print(
                f"{name}: no requirements.txt"
            )
            continue

        print(
            f"\nInstalling dependencies for {name}"
        )

        install_requirements(
            requirements,
            f".{name}_requirements_safe.txt",
        )


# ============================================================
# SOURCE COMPATIBILITY
# ============================================================

def verify_ltx_source_compatibility():
    print("\n" + "=" * 70)
    print(
        "VERIFYING LTX / COMFYUI SOURCE COMPATIBILITY"
    )
    print("=" * 70)

    ltx_repo = CUSTOM / "ComfyUI-LTXVideo"
    ltx_revision = git_current_commit(
        ltx_repo
    )

    # Use the actual checked-out revision, not the config value.
    comfy_revision = git_current_commit(
        COMFY
    )

    result = subprocess.run(
        [
            "git",
            "-C",
            str(ltx_repo),
            "show",
            f"{ltx_revision}:tricks/modules/ltx_model.py",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    ltx_source = result.stdout

    required = [
        "BasicTransformerBlock",
        "LTXVModel",
        "apply_rotary_emb",
        "precompute_freqs_cis",
    ]

    result = subprocess.run(
        [
            "git",
            "-C",
            str(COMFY),
            "show",
            f"{comfy_revision}:comfy/ldm/lightricks/model.py",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    comfy_source = result.stdout

    print(
        "\nComfyUI revision:",
        comfy_revision,
    )

    print(
        "LTXVideo revision:",
        ltx_revision,
    )

    print(
        "\nRequired LTXVideo symbols:"
    )

    for symbol in required:

        present = symbol in comfy_source

        print(
            f"{symbol:35}"
            f"{'✅' if present else '❌'}"
        )

        if not present:
            raise RuntimeError(
                "\nLTXVideo / ComfyUI source "
                "compatibility check failed.\n"
                f"Missing: {symbol}\n"
                f"ComfyUI revision: {comfy_revision}\n"
                f"LTXVideo revision: {ltx_revision}"
            )

    print(
        "\n✅ LTXVideo ↔ ComfyUI API compatibility verified."
    )


def verify_kjnodes_source():
    print("\n" + "=" * 70)
    print("VERIFYING KJ NODES")
    print("=" * 70)

    repository = CUSTOM / "ComfyUI-KJNodes"
    revision = git_current_commit(
        repository
    )

    required_nodes = [
        "StringToFloatList",
        "FloatToSigmas",
        "ImageResizeKJ",
    ]

    python_files = subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "ls-tree",
            "-r",
            "--name-only",
            revision,
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()

    python_files = [
        path
        for path in python_files
        if path.endswith(".py")
    ]

    combined = ""

    for path in python_files:

        result = subprocess.run(
            [
                "git",
                "-C",
                str(repository),
                "show",
                f"{revision}:{path}",
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            combined += (
                "\n" + result.stdout
            )

    print(
        "\nKJNodes revision:",
        revision,
    )

    for node in required_nodes:

        if node not in combined:
            raise RuntimeError(
                "\nRequired KJNodes node missing:\n"
                f"{node}"
            )

        print(
            f"{node:30} ✅"
        )

    breaking = (
        "_append_guide_attention_entry"
    )

    if breaking in combined:
        raise RuntimeError(
            "\nIncompatible KJNodes API detected:\n"
            f"{breaking}"
        )

    print(
        f"{breaking:30} ✅ absent"
    )

    print(
        "\n✅ KJNodes source verified."
    )


# ============================================================
# VALIDATION
# ============================================================

def validate_comfy():
    print("\n" + "=" * 70)
    print("VALIDATING COMFYUI INSTALLATION")
    print("=" * 70)

    required_paths = [
        COMFY / "main.py",
        COMFY / "requirements.txt",
        CUSTOM / "ComfyUI-GGUF",
        CUSTOM / "ComfyUI-LTXVideo",
        CUSTOM / "ComfyUI-VideoHelperSuite",
        CUSTOM / "rgthree-comfy",
        CUSTOM / "ComfyUI-KJNodes",
    ]

    for path in required_paths:

        if not path.exists():
            raise RuntimeError(
                f"Missing required path:\n{path}"
            )

        print(f"✅ {path}")


def validate_repositories():
    print("\n" + "=" * 70)
    print("VALIDATING REPOSITORIES")
    print("=" * 70)

    # --------------------------------------------------------
    # ComfyUI
    # --------------------------------------------------------

    if not (COMFY / "main.py").exists():
        raise RuntimeError(
            "ComfyUI main.py is missing."
        )

    comfy_revision = git_current_commit(
        COMFY
    )

    if COMFY_COMMIT:

        if comfy_revision != COMFY_COMMIT:

            raise RuntimeError(
                "\nComfyUI revision mismatch.\n"
                f"Expected: {COMFY_COMMIT}\n"
                f"Found:    {comfy_revision}"
            )

        print(
            f"ComfyUI: {comfy_revision} ✅"
        )

    else:

        print(
            f"ComfyUI: {comfy_revision} "
            "(latest/unpinned) ✅"
        )

    # --------------------------------------------------------
    # Custom nodes
    # --------------------------------------------------------

    for name, info in NODES.items():

        node_dir = CUSTOM / name

        if not node_dir.exists():
            raise RuntimeError(
                f"Missing custom node: {name}"
            )

        revision = git_current_commit(
            node_dir
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
                f"{name}: {revision} "
                "(latest/unpinned)"
            )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("LTX-13B AI VIDEO MODEL BOOTSTRAP")
    print("=" * 70)

    PROJECT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 1. Verify runtime.
    verify_torch()

    # 2. Sync ComfyUI.
    print(
        "\nInstalling / synchronizing ComfyUI..."
    )

    clone_or_sync(
        COMFY_REPO,
        COMFY,
        COMFY_COMMIT,
    )

    # 3. Ensure custom node directory.
    CUSTOM.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 4. Sync custom nodes.
    print(
        "\nInstalling / synchronizing custom nodes..."
    )

    for name, info in NODES.items():

        clone_or_sync(
            info["url"],
            CUSTOM / name,
            info["commit"],
        )

    # 5. Dependencies.
    install_comfy_dependencies()
    install_node_dependencies()

    # 6. Filesystem validation.
    validate_comfy()

    # 7. Repository validation.
    validate_repositories()

    # 8. Source compatibility.
    verify_ltx_source_compatibility()
    verify_kjnodes_source()

    # 9. Final runtime verification.
    verify_torch()

    print("\n" + "=" * 70)
    print("✅ BOOTSTRAP COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()
