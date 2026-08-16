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
# VERIFIED VERSION PINS
# ============================================================

# Official recovered LTX 0.9.8 workflow metadata:
# comfy-core 0.3.30
#
# Full commit SHA resolved from tag v0.3.30.
COMFY_REPO = (
    "https://github.com/Comfy-Org/ComfyUI.git"
)

COMFY_COMMIT =  None


# Verified LTXVideo revision containing the required
# LTX 0.9.8 nodes and STG pipeline.
LTXVIDEO_COMMIT = (
    "ee11be3ce229c3afd5fadf8a1258eb8b84af33b1"
)


# Verified KJNodes 1.0.8 revision.
KJNODES_COMMIT = (
    "89fb17ae84951995ab1eee19e205ea48ceed27c9"
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
        "commit": KJNODES_COMMIT,
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

def get_current_commit(repository: Path) -> str:

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


def repository_is_git(repository: Path) -> bool:

    return (
        (repository / ".git").exists()
    )


def checkout_pinned_commit(
    repository: Path,
    commit: str,
):

    print(
        f"\nPinning repository to: {commit}"
    )

    # Full history is deliberately used for pinned
    # historical revisions.
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

    # Force checkout so stale working-tree state can never
    # prevent the pinned revision from being selected.
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

    # --------------------------------------------------------
    # Existing repository
    # --------------------------------------------------------

    if destination.exists():

        if repository_is_git(
            destination
        ):

            print(
                f"\nRepository already exists:"
                f" {destination}"
            )

            if commit:

                checkout_pinned_commit(
                    destination,
                    commit,
                )

            return

        # Directory exists but isn't a Git repository.
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

    # Full clone is intentional for pinned historical
    # revisions.
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


# ============================================================
# PYTORCH / CUDA VERIFICATION
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
# CUSTOM-NODE DEPENDENCIES
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
# SOURCE COMPATIBILITY VERIFICATION
# ============================================================

def verify_ltx_source_compatibility():

    print(
        "\n" + "=" * 70
    )

    print(
        "VERIFYING LTX / COMFYUI SOURCE COMPATIBILITY"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # Read the exact imports required by our pinned LTXVideo.
    # --------------------------------------------------------

    result = subprocess.run(
        [
            "git",
            "-C",
            str(CUSTOM / "ComfyUI-LTXVideo"),
            "show",
            (
                f"{LTXVIDEO_COMMIT}:"
                "tricks/modules/ltx_model.py"
            ),
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

    # --------------------------------------------------------
    # Read pinned ComfyUI Lightricks model.
    # --------------------------------------------------------

    result = subprocess.run(
        [
            "git",
            "-C",
            str(COMFY),
            "show",
            (
                f"{COMFY_COMMIT}:"
                "comfy/ldm/lightricks/model.py"
            ),
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    comfy_source = result.stdout

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
                f"Missing: {symbol}"
            )

    print(
        "\n✅ LTXVideo ↔ ComfyUI API compatibility verified."
    )


# ============================================================
# KJ NODES SOURCE VERIFICATION
# ============================================================

def verify_kjnodes_source():

    print(
        "\n" + "=" * 70
    )

    print(
        "VERIFYING KJ NODES 1.0.8"
    )

    print(
        "=" * 70
    )

    repository = (
        CUSTOM /
        "ComfyUI-KJNodes"
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
            KJNODES_COMMIT,
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
                f"{KJNODES_COMMIT}:{path}",
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            combined += (
                "\n" +
                result.stdout
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
        "\n✅ KJNodes 1.0.8 source verified."
    )


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
                f"{name}: "
                f"{revision} ✅"
            )

        else:

            print(
                f"{name}: "
                f"{revision}"
            )


# ============================================================
# VALIDATE FILESYSTEM
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

        CUSTOM /
        "ComfyUI-GGUF",

        CUSTOM /
        "ComfyUI-LTXVideo",

        CUSTOM /
        "ComfyUI-VideoHelperSuite",

        CUSTOM /
        "rgthree-comfy",

        CUSTOM /
        "ComfyUI-KJNodes",
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
    # 1. GPU / Torch verification
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
    # 3. Custom nodes directory
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
    # 5. Install ComfyUI dependencies
    # --------------------------------------------------------

    install_comfy_dependencies()

    # --------------------------------------------------------
    # 6. Install custom-node dependencies
    # --------------------------------------------------------

    install_node_dependencies()

    # --------------------------------------------------------
    # 7. Validate filesystem
    # --------------------------------------------------------

    validate_comfy()

    # --------------------------------------------------------
    # 8. Validate Git revisions
    # --------------------------------------------------------

    validate_repositories()

    # --------------------------------------------------------
    # 9. Verify actual source compatibility
    # --------------------------------------------------------

    verify_ltx_source_compatibility()

    verify_kjnodes_source()

    # --------------------------------------------------------
    # 10. Verify Torch again
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
