from pathlib import Path
import importlib.metadata
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

# Proven ComfyUI v0.3.44 revision.
COMFY_COMMIT = (
    "c5de4955bb91a2b136027a698aaecb8d19e3d892"
)

# Proven LTX 0.9.8 workflow implementation.
LTXVIDEO_COMMIT = (
    "ee11be3ce229c3afd5fadf8a1258eb8b84af33b1"
)

# Proven KJNodes revision used successfully with the workflow.
KJNODES_COMMIT = (
    "7ecb190ef91d988420cf0e682efb79ac7433c0b7"
)


# ============================================================
# PROVEN CUSTOM NODE REVISIONS
# ============================================================

NODES = {
    "ComfyUI-GGUF": {
        "url": (
            "https://github.com/city96/"
            "ComfyUI-GGUF.git"
        ),
        "commit": (
            "6ea2651e7df66d7585f6ffee804b20e92fb38b8a"
        ),
    },

    "ComfyUI-LTXVideo": {
        "url": (
            "https://github.com/Lightricks/"
            "ComfyUI-LTXVideo.git"
        ),
        "commit": LTXVIDEO_COMMIT,
    },

    "ComfyUI-VideoHelperSuite": {
        "url": (
            "https://github.com/Kosinkadink/"
            "ComfyUI-VideoHelperSuite.git"
        ),
        "commit": (
            "4ee72c065db22c9d96c2427954dc69e7b908444b"
        ),
    },

    "rgthree-comfy": {
        "url": (
            "https://github.com/rgthree/"
            "rgthree-comfy.git"
        ),
        "commit": None,
    },

    "ComfyUI-KJNodes": {
        "url": (
            "https://github.com/kijai/"
            "ComfyUI-KJNodes.git"
        ),
        "commit": KJNODES_COMMIT,
    },
}


# ============================================================
# PROVEN PYTHON PACKAGE VERSIONS
# ============================================================

FRONTEND_PACKAGES = {
    "comfyui-frontend-package": "1.23.4",
    "comfyui-workflow-templates": "0.1.35",
    "comfyui-embedded-docs": "0.2.4",
}

RUNTIME_PACKAGES = {
    "torchsde": "0.2.6",
    "spandrel": "0.4.2",
    "av": "18.1.0",
    "gguf": "0.19.0",
}


# ============================================================
# TORCH / CUDA SAFETY
# ============================================================

EXPECTED_TORCH_PREFIX = "2.10.0+cu128"

FORBIDDEN_PACKAGES = {
    "torch",
    "torchvision",
    "torchaudio",
}


# ============================================================
# MODEL DATASET PATHS
# ============================================================

MODEL_SOURCES = {
    "ltx_q4": (
        Path(
            "/kaggle/input/datasets/shihoos/"
            "ltx13b-q4"
        )
        / "LTXV-13B-0.9.8-distilled-Q4_K_M.gguf"
    ),

    "t5_q4": (
        Path(
            "/kaggle/input/datasets/shihoos/"
            "ltx13b-t5"
        )
        / "t5-v1_1-xxl-encoder-Q4_K_M.gguf"
    ),

    "vae": (
        Path(
            "/kaggle/input/datasets/shihoos/"
            "ltx13b-vae"
        )
        / "LTXV-13B-0.9.8-distilled-VAE.safetensors"
    ),

    "ic_lora": (
        Path(
            "/kaggle/input/datasets/shihoos/"
            "ltx13b-enhancers"
        )
        / "ltxv-098-ic-lora-detailer-comfyui.safetensors"
    ),

    "spatial_upscaler": (
        Path(
            "/kaggle/input/datasets/shihoos/"
            "ltx13b-enhancers"
        )
        / "ltxv-spatial-upscaler-0.9.8.safetensors"
    ),
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
        [
            "git",
            "-C",
            str(repository),
            *args,
        ],
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
    return (
        repository / ".git"
    ).exists()


# ============================================================
# REPOSITORY SYNCHRONIZATION
# ============================================================

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

    actual = git_current_commit(
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


def clone_or_sync(
    url: str,
    destination: Path,
    commit: str | None = None,
):
    if destination.exists():

        if not repository_is_git(
            destination
        ):
            print(
                f"\nRemoving incomplete repository: "
                f"{destination}"
            )

            shutil.rmtree(
                destination
            )

        else:

            print(
                f"\nRepository already exists: "
                f"{destination}"
            )

            checkout_pinned_commit(
                destination,
                commit,
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

    checkout_pinned_commit(
        destination,
        commit,
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

        if (
            not line
            or line.startswith("#")
        ):
            continue

        line = (
            line
            .split("#", 1)[0]
            .strip()
        )

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

    destination = (
        PROJECT / temporary_name
    )

    filter_requirements(
        source,
        destination,
    )

    content = destination.read_text(
        encoding="utf-8"
    ).strip()

    if not content:

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

        requirements = (
            CUSTOM
            / name
            / "requirements.txt"
        )

        if not requirements.exists():

            print(
                f"{name}: "
                "no requirements.txt"
            )

            continue

        print(
            f"\nInstalling dependencies "
            f"for {name}"
        )

        install_requirements(
            requirements,
            f".{name}_requirements_safe.txt",
        )


def install_proven_runtime_packages():
    print(
        "\n" + "=" * 70
    )
    print(
        "PROVEN LTX-13B RUNTIME PACKAGES"
    )
    print(
        "=" * 70
    )

    packages = {
        **FRONTEND_PACKAGES,
        **RUNTIME_PACKAGES,
    }

    for package, version in packages.items():

        requirement = (
            f"{package}=={version}"
        )

        print(
            f"\nInstalling {requirement}"
        )

        run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--force-reinstall",
                "--no-deps",
                requirement,
            ]
        )

    print(
        "\n✅ Proven runtime packages installed."
    )


def verify_proven_runtime_packages():
    print(
        "\n" + "=" * 70
    )
    print(
        "VERIFYING PROVEN RUNTIME PACKAGES"
    )
    print(
        "=" * 70
    )

    expected = {
        **FRONTEND_PACKAGES,
        **RUNTIME_PACKAGES,
    }

    for package, wanted in expected.items():

        try:
            actual = (
                importlib.metadata.version(
                    package
                )
            )
        except (
            importlib.metadata.PackageNotFoundError
        ):
            actual = None

        print(
            f"{package}: "
            f"{actual}"
        )

        if actual != wanted:
            raise RuntimeError(
                f"\nPackage version mismatch:\n"
                f"{package}\n"
                f"Expected: {wanted}\n"
                f"Found: {actual}"
            )

    print(
        "\n✅ All proven runtime versions verified."
    )


# ============================================================
# MODEL PATHS
# ============================================================

def install_model_links():
    print(
        "\n" + "=" * 70
    )
    print(
        "CONNECTING KAGGLE MODEL DATASETS"
    )
    print(
        "=" * 70
    )

    targets = {
        "ltx_q4": (
            COMFY
            / "models"
            / "unet"
            / MODEL_SOURCES[
                "ltx_q4"
            ].name
        ),

        "t5_q4": (
            COMFY
            / "models"
            / "clip"
            / MODEL_SOURCES[
                "t5_q4"
            ].name
        ),

        "vae": (
            COMFY
            / "models"
            / "vae"
            / MODEL_SOURCES[
                "vae"
            ].name
        ),

        "ic_lora": (
            COMFY
            / "models"
            / "loras"
            / MODEL_SOURCES[
                "ic_lora"
            ].name
        ),
    }

    for name, source in MODEL_SOURCES.items():

        if not source.exists():
            raise FileNotFoundError(
                f"{name} model not found:\n"
                f"{source}"
            )

        print(
            f"✅ {name}: "
            f"{source}"
        )

    for key, target in targets.items():

        source = MODEL_SOURCES[key]

        target.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if (
            target.is_symlink()
            or target.exists()
        ):
            target.unlink()

        target.symlink_to(
            source.resolve()
        )

        print(
            f"✅ {key} → {target}"
        )

    # Spatial upscaler is registered by
    # the pinned LTXVideo implementation.
    # Keep it available in the likely model locations.
    spatial = MODEL_SOURCES[
        "spatial_upscaler"
    ]

    spatial_dirs = [
        COMFY / "models" / "upscale_models",
        COMFY / "models" / "latent_upscale",
        COMFY / "models" / "latent_upscalers",
        COMFY / "models" / "ltxv",
        COMFY / "models" / "upscalers",
    ]

    for directory in spatial_dirs:

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        target = (
            directory / spatial.name
        )

        if (
            target.is_symlink()
            or target.exists()
        ):
            target.unlink()

        target.symlink_to(
            spatial.resolve()
        )

    print(
        "✅ Spatial upscaler connected."
    )


# ============================================================
# SOURCE COMPATIBILITY
# ============================================================

def verify_ltx_source_compatibility():
    print(
        "\n" + "=" * 70
    )
    print(
        "VERIFYING LTX / COMFYUI "
        "SOURCE COMPATIBILITY"
    )
    print(
        "=" * 70
    )

    ltx_repo = (
        CUSTOM / "ComfyUI-LTXVideo"
    )

    ltx_revision = (
        git_current_commit(
            ltx_repo
        )
    )

    comfy_revision = (
        git_current_commit(
            COMFY
        )
    )

    result = subprocess.run(
        [
            "git",
            "-C",
            str(ltx_repo),
            "show",
            (
                f"{ltx_revision}:"
                "tricks/modules/ltx_model.py"
            ),
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    ltx_source = result.stdout

    required_ltx_symbols = [
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
            (
                f"{comfy_revision}:"
                "comfy/ldm/lightricks/model.py"
            ),
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    comfy_source = result.stdout

    print(
        "ComfyUI revision:",
        comfy_revision,
    )

    print(
        "LTXVideo revision:",
        ltx_revision,
    )

    for symbol in required_ltx_symbols:

        present = (
            symbol in comfy_source
        )

        print(
            f"{symbol:35}"
            f"{'✅' if present else '❌'}"
        )

        if not present:
            raise RuntimeError(
                "\nLTXVideo / ComfyUI "
                "source compatibility failed.\n"
                f"Missing: {symbol}\n"
                f"ComfyUI: {comfy_revision}\n"
                f"LTXVideo: {ltx_revision}"
            )

    print(
        "\n✅ LTXVideo ↔ ComfyUI "
        "source compatibility verified."
    )


def verify_kjnodes_source():
    print(
        "\n" + "=" * 70
    )
    print(
        "VERIFYING KJ NODES"
    )
    print(
        "=" * 70
    )

    repository = (
        CUSTOM / "ComfyUI-KJNodes"
    )

    revision = (
        git_current_commit(
            repository
        )
    )

    required_nodes = [
        "StringToFloatList",
        "FloatToSigmas",
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

    combined = ""

    for path in python_files:

        if not path.endswith(".py"):
            continue

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
        "KJNodes revision:",
        revision,
    )

    for node in required_nodes:

        if node not in combined:
            raise RuntimeError(
                f"Required KJNodes node missing: "
                f"{node}"
            )

        print(
            f"{node:30} ✅"
        )

    breaking_symbol = (
        "_append_guide_attention_entry"
    )

    if breaking_symbol in combined:
        raise RuntimeError(
            "Incompatible KJNodes API detected:\n"
            f"{breaking_symbol}"
        )

    print(
        f"{breaking_symbol:30} ✅ absent"
    )

    print(
        "\n✅ KJNodes source verified."
    )


# ============================================================
# COMFYUI VALIDATION
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
                f"Missing required path:\n"
                f"{path}"
            )

        print(
            f"✅ {path}"
        )


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

    comfy_revision = (
        git_current_commit(COMFY)
    )

    if comfy_revision != COMFY_COMMIT:
        raise RuntimeError(
            "ComfyUI revision mismatch.\n"
            f"Expected: {COMFY_COMMIT}\n"
            f"Found: {comfy_revision}"
        )

    print(
        f"ComfyUI: "
        f"{comfy_revision} ✅"
    )

    for name, info in NODES.items():

        node_dir = (
            CUSTOM / name
        )

        if not node_dir.exists():
            raise RuntimeError(
                f"Missing custom node: {name}"
            )

        revision = (
            git_current_commit(
                node_dir
            )
        )

        requested = info["commit"]

        if requested:

            if revision != requested:
                raise RuntimeError(
                    f"\n{name} revision mismatch.\n"
                    f"Expected: {requested}\n"
                    f"Found: {revision}"
                )

            print(
                f"{name}: "
                f"{revision} ✅"
            )

        else:

            print(
                f"{name}: "
                f"{revision} ✅"
            )


# ============================================================
# FINAL NODE REGISTRATION TEST
# ============================================================

def verify_registered_nodes():
    print(
        "\n" + "=" * 70
    )
    print(
        "VERIFYING REQUIRED WORKFLOW NODES"
    )
    print(
        "=" * 70
    )

    import json
    import urllib.request

    url = (
        "http://127.0.0.1:8188/object_info"
    )

    try:

        with urllib.request.urlopen(
            url,
            timeout=10,
        ) as response:

            data = json.loads(
                response.read()
            )

    except Exception:

        print(
            "ComfyUI is not running yet; "
            "skipping live node check."
        )

        return

    required = [
        "LTXVBaseSampler",
        "LTXVConditioning",
        "STGGuiderAdvanced",
        "FloatToSigmas",
        "StringToFloatList",
        "LTXVLoopingSampler",
        "LTXVLatentUpsampler",
        "LTXVLatentUpsamplerModelLoader",
        "LTXVTiledVAEDecode",
        "LTXVFilmGrain",
        "VHS_LoadVideo",
        "VHS_VideoCombine",
    ]

    missing = []

    for node in required:

        if node in data:
            print(
                f"✅ {node}"
            )
        else:
            print(
                f"❌ {node}"
            )
            missing.append(node)

    if missing:
        raise RuntimeError(
            "Missing workflow nodes:\n"
            + "\n".join(missing)
        )

    print(
        "\n✅ Required workflow nodes registered."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 70
    )

    print(
        "LTX-13B AI VIDEO MODEL "
        "PRODUCTION BOOTSTRAP"
    )

    print(
        "=" * 70
    )

    PROJECT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 1. Runtime verification
    verify_torch()

    # 2. ComfyUI
    print(
        "\nInstalling / synchronizing "
        "pinned ComfyUI..."
    )

    clone_or_sync(
        COMFY_REPO,
        COMFY,
        COMFY_COMMIT,
    )

    # 3. Custom nodes
    CUSTOM.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "\nInstalling / synchronizing "
        "pinned custom nodes..."
    )

    for name, info in NODES.items():

        clone_or_sync(
            info["url"],
            CUSTOM / name,
            info["commit"],
        )

    # 4. Requirements
    install_comfy_dependencies()
    install_node_dependencies()

    # 5. Restore exact proven runtime package versions.
    install_proven_runtime_packages()

    # 6. Verify runtime versions.
    verify_proven_runtime_packages()

    # 7. Model dataset links.
    install_model_links()

    # 8. Filesystem validation.
    validate_comfy()

    # 9. Repository validation.
    validate_repositories()

    # 10. Source compatibility.
    verify_ltx_source_compatibility()
    verify_kjnodes_source()

    # 11. Final Torch verification.
    verify_torch()

    # 12. Live node check if server already exists.
    verify_registered_nodes()

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
