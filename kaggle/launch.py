from pathlib import Path
import importlib
import importlib.util
import os
import socket
import subprocess
import sys


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT = Path(
    "/kaggle/working/LTX-13B-AI-Video-Model"
)

COMFY = PROJECT / "ComfyUI"

BOOTSTRAP = (
    PROJECT / "kaggle" / "bootstrap.py"
)

TUNNEL = (
    PROJECT
    / "kaggle"
    / "start_comfyui_tunnel.py"
)

EXTRA_PATHS = (
    COMFY / "extra_model_paths.yaml"
)

PORT = 8188


# ============================================================
# GPU REQUIREMENT
# ============================================================

MIN_GPUS = 2

EXPECTED_TORCH_PREFIX = "2.10.0+cu128"


# ============================================================
# MODEL FILES
# ============================================================

MODEL_FILES = {

    "ltx_q4":
        "LTXV-13B-0.9.8-distilled-Q4_K_M.gguf",

    "vae":
        "LTXV-13B-0.9.8-distilled-VAE.safetensors",

    "t5":
        "t5-v1_1-xxl-encoder-Q4_K_M.gguf",

    "detailer":
        "ltxv-098-ic-lora-detailer-comfyui.safetensors",

    "upscaler":
        "ltxv-spatial-upscaler-0.9.8.safetensors",
}


# ============================================================
# REQUIRED PYTHON MODULES
#
# These are the modules needed by:
# - ComfyUI
# - ComfyUI-GGUF
# - ComfyUI-LTXVideo
# - VideoHelperSuite
#
# Torch itself is deliberately NOT installed here.
# ============================================================

REQUIRED_MODULES = {

    "comfy_aimdo":
        "comfy-aimdo",

    "comfy_kitchen":
        "comfy-kitchen",

    "torchsde":
        "torchsde",

    "av":
        "av",

    "gguf":
        "gguf",

    "spandrel":
        "spandrel",

    "simpleeval":
        "simpleeval",

    "comfy_angle":
        "comfy-angle",
}


# These are separate ComfyUI packages that are checked with
# pip because their package/module relationship is different.

REQUIRED_COMFY_PACKAGES = {

    "comfyui-frontend-package":
        "comfyui-frontend-package",

    "comfyui-workflow-templates":
        "comfyui-workflow-templates",

    "comfyui-embedded-docs":
        "comfyui-embedded-docs",
}


# ============================================================
# CUSTOM NODES
# ============================================================

REQUIRED_CUSTOM_NODES = [

    "ComfyUI-GGUF",

    "ComfyUI-LTXVideo",

    "ComfyUI-VideoHelperSuite",

    "rgthree-comfy",

    "ComfyUI-KJNodes",
]


# ============================================================
# COMMAND HELPER
# ============================================================

def run(
    command,
    cwd=None,
):
    print(
        "\n$",
        " ".join(
            map(str, command)
        ),
        flush=True,
    )

    subprocess.run(
        command,
        cwd=cwd,
        check=True,
    )


# ============================================================
# GPU CHECK
# ============================================================

def check_gpu():

    import torch

    print()
    print("=" * 70)
    print("GPU / CUDA CHECK")
    print("=" * 70)

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
            "CUDA is not available."
        )

    gpu_count = (
        torch.cuda.device_count()
    )

    print(
        "GPU count:",
        gpu_count,
    )

    if gpu_count < MIN_GPUS:

        raise RuntimeError(
            f"This project requires at least "
            f"{MIN_GPUS} GPUs, but only "
            f"{gpu_count} were detected."
        )

    for index in range(gpu_count):

        print(
            f"GPU {index}:",
            torch.cuda.get_device_name(index),
        )

    # Do not silently replace the known-good Torch stack.
    if not torch.__version__.startswith(
        EXPECTED_TORCH_PREFIX
    ):

        raise RuntimeError(
            "\nUnexpected PyTorch version.\n"
            f"Expected: {EXPECTED_TORCH_PREFIX}...\n"
            f"Found:    {torch.__version__}\n\n"
            "This launcher intentionally refuses "
            "to replace the existing Torch/CUDA "
            "installation."
        )


# ============================================================
# REPOSITORY CHECK
# ============================================================

def ensure_repository():

    if PROJECT.exists():

        print(
            "✅ Project repository already exists."
        )

        return

    raise RuntimeError(
        "\nThe project repository does not exist:\n"
        f"{PROJECT}\n\n"
        "Use the notebook's one-cell startup "
        "cell to clone GitHub first."
    )


# ============================================================
# COMFYUI CHECK / BOOTSTRAP
# ============================================================

def ensure_comfyui():

    main_py = (
        COMFY / "main.py"
    )

    if main_py.exists():

        print(
            "✅ ComfyUI installation found."
        )

        return

    print()
    print(
        "⚠️ ComfyUI installation missing."
    )

    if not BOOTSTRAP.exists():

        raise FileNotFoundError(
            f"Bootstrap script not found:\n"
            f"{BOOTSTRAP}"
        )

    print(
        "Running bootstrap.py..."
    )

    run(
        [
            sys.executable,
            str(BOOTSTRAP),
        ]
    )

    if not main_py.exists():

        raise RuntimeError(
            "Bootstrap completed, but "
            "ComfyUI/main.py was not created."
        )

    print(
        "✅ ComfyUI installed successfully."
    )


# ============================================================
# PACKAGE VERSION LOOKUP
# ============================================================

def build_requirement_lookup():

    requirements = (
        COMFY / "requirements.txt"
    )

    if not requirements.exists():

        raise FileNotFoundError(
            f"ComfyUI requirements.txt not found:\n"
            f"{requirements}"
        )

    lookup = {}

    for raw_line in requirements.read_text(
        encoding="utf-8"
    ).splitlines():

        line = raw_line.strip()

        if (
            not line
            or line.startswith("#")
        ):
            continue

        # Remove inline comments.
        line = line.split(
            " #",
            1,
        )[0].strip()

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

        lookup[normalized] = line

    return lookup


# ============================================================
# REQUIRED PACKAGE CHECK
# ============================================================

def package_installed(
    package_name,
):
    """
    Check pip package installation without importing it.
    """

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "show",
            package_name,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )

    return result.returncode == 0


# ============================================================
# RUNTIME DEPENDENCY REPAIR
# ============================================================

def ensure_runtime_dependencies():

    print()
    print("=" * 70)
    print(
        "CHECKING COMFYUI RUNTIME DEPENDENCIES"
    )
    print("=" * 70)

    requirement_lookup = (
        build_requirement_lookup()
    )

    packages_to_install = []

    # --------------------------------------------------------
    # Python modules
    # --------------------------------------------------------

    for module_name, package_name in (
        REQUIRED_MODULES.items()
    ):

        if importlib.util.find_spec(
            module_name
        ) is not None:

            continue

        requirement = (
            requirement_lookup.get(
                package_name.lower(),
                package_name,
            )
        )

        packages_to_install.append(
            requirement
        )

        print(
            f"MISSING MODULE: "
            f"{module_name} "
            f"→ {requirement}"
        )

    # --------------------------------------------------------
    # ComfyUI packages
    # --------------------------------------------------------

    for package_name in (
        REQUIRED_COMFY_PACKAGES.values()
    ):

        if package_installed(
            package_name
        ):

            continue

        requirement = (
            requirement_lookup.get(
                package_name.lower(),
                package_name,
            )
        )

        packages_to_install.append(
            requirement
        )

        print(
            f"MISSING PACKAGE: "
            f"{package_name} "
            f"→ {requirement}"
        )

    # Remove duplicates.
    packages_to_install = list(
        dict.fromkeys(
            packages_to_install
        )
    )

    if not packages_to_install:

        print(
            "✅ All required runtime packages "
            "are installed."
        )

        return

    print()
    print(
        "Installing:",
        packages_to_install,
    )

    # IMPORTANT:
    # Never install torch/torchvision/torchaudio here.
    forbidden = {
        "torch",
        "torchvision",
        "torchaudio",
    }

    safe_packages = []

    for package in packages_to_install:

        normalized = (
            package
            .split("==", 1)[0]
            .split(">=", 1)[0]
            .split("<=", 1)[0]
            .split("~=", 1)[0]
            .strip()
            .lower()
        )

        if normalized in forbidden:
            continue

        safe_packages.append(
            package
        )

    if safe_packages:

        run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                *safe_packages,
            ]
        )

    print(
        "✅ Runtime dependency repair complete."
    )


# ============================================================
# CUSTOM NODE REQUIREMENTS
# ============================================================

def ensure_custom_nodes():

    print()
    print("=" * 70)
    print(
        "CHECKING CUSTOM NODE INSTALLATIONS"
    )
    print("=" * 70)

    custom_nodes_dir = (
        COMFY / "custom_nodes"
    )

    custom_nodes_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    custom_node_repositories = {

        "ComfyUI-GGUF":
            "https://github.com/city96/ComfyUI-GGUF.git",

        "ComfyUI-LTXVideo":
            "https://github.com/Lightricks/ComfyUI-LTXVideo.git",

        "ComfyUI-VideoHelperSuite":
            "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git",

        "rgthree-comfy":
            "https://github.com/rgthree/rgthree-comfy.git",

        "ComfyUI-KJNodes":
            "https://github.com/kijai/ComfyUI-KJNodes.git",
    }

    for name, url in custom_node_repositories.items():

        target = (
            custom_nodes_dir / name
        )

        if target.exists():

            print(
                f"✅ {name}"
            )

            continue

        print(
            f"⚠️ {name} missing."
        )

        print(
            f"Cloning {name}..."
        )

        run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                url,
                str(target),
            ]
        )

        print(
            f"✅ {name} cloned."
        )

def install_custom_node_requirements():

    print()
    print("=" * 70)
    print(
        "CHECKING CUSTOM-NODE DEPENDENCIES"
    )
    print("=" * 70)

    custom_nodes_dir = (
        COMFY / "custom_nodes"
    )

    forbidden = {
        "torch",
        "torchvision",
        "torchaudio",
    }

    for node_name in (
        REQUIRED_CUSTOM_NODES
    ):

        node_dir = (
            custom_nodes_dir
            / node_name
        )

        requirements = (
            node_dir / "requirements.txt"
        )

        if not requirements.exists():

            print(
                f"{node_name}: "
                "no requirements.txt"
            )

            continue

        filtered_lines = []

        for raw_line in (
            requirements.read_text(
                encoding="utf-8"
            ).splitlines()
        ):

            line = raw_line.strip()

            if (
                not line
                or line.startswith("#")
            ):
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

            if package_name in forbidden:
                continue

            filtered_lines.append(
                line
            )

        if not filtered_lines:

            print(
                f"{node_name}: "
                "no additional dependencies"
            )

            continue

        temp_requirements = (
            PROJECT
            / f".{node_name}_requirements.txt"
        )

        temp_requirements.write_text(
            "\n".join(
                filtered_lines
            ) + "\n",
            encoding="utf-8",
        )

        print(
            f"Installing requirements for "
            f"{node_name}..."
        )

        run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "-r",
                str(temp_requirements),
            ]
        )

        try:
            temp_requirements.unlink()
        except OSError:
            pass

    print(
        "✅ Custom-node dependency setup complete."
    )


# ============================================================
# LTXVIDEO / KORNIA COMPATIBILITY
# ============================================================

LTXVIDEO_COMMIT = "ee11be3ce229c3afd5fadf8a1258eb8b84af33b1"


def get_git_commit(repository):
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


def ensure_ltxvideo_revision():
    """Ensure the installed LTXVideo checkout is the verified revision."""

    ltx_dir = (
        COMFY
        / "custom_nodes"
        / "ComfyUI-LTXVideo"
    )

    if not ltx_dir.exists():
        raise FileNotFoundError(
            f"ComfyUI-LTXVideo not found:\n{ltx_dir}"
        )

    print()
    print("Checking LTXVideo Git revision...")

    current = get_git_commit(ltx_dir)

    if current != LTXVIDEO_COMMIT:
        print(
            f"⚠️ LTXVideo revision is {current}; "
            f"switching to {LTXVIDEO_COMMIT}."
        )

        run(
            [
                "git",
                "fetch",
                "--all",
                "--tags",
                "--prune",
            ],
            cwd=ltx_dir,
        )

        run(
            [
                "git",
                "checkout",
                "--force",
                LTXVIDEO_COMMIT,
            ],
            cwd=ltx_dir,
        )

        current = get_git_commit(ltx_dir)

    if current != LTXVIDEO_COMMIT:
        raise RuntimeError(
            "LTXVideo revision verification failed.\n"
            f"Expected: {LTXVIDEO_COMMIT}\n"
            f"Found:    {current}"
        )

    print(
        f"✅ LTXVideo revision: {current}"
    )


def patch_ltx_kornia_compatibility():

    print()
    print("=" * 70)
    print("CHECKING LTXVIDEO / KORNIA COMPATIBILITY")
    print("=" * 70)

    pyramid_file = (
        COMFY
        / "custom_nodes"
        / "ComfyUI-LTXVideo"
        / "pyramid_blending.py"
    )

    # The verified 0.9.8-era LTXVideo revision does not
    # contain pyramid_blending.py, so there is nothing to patch.
    if not pyramid_file.exists():
        print(
            "✅ Pinned LTXVideo revision has no "
            "pyramid_blending.py; Kornia patch not required."
        )
        return

    text = pyramid_file.read_text(
        encoding="utf-8"
    )

    original = text

    old_import = """from kornia.geometry.transform.pyramid import (
    PyrUp,
    build_laplacian_pyramid,
    build_pyramid,
    find_next_powerof_two,
    is_powerof_two,
    pad,
)"""

    new_import = """from kornia.geometry.transform.pyramid import (
    PyrUp,
    build_laplacian_pyramid,
    build_pyramid,
    find_next_powerof_two,
    is_powerof_two,
)"""

    if old_import in text:
        text = text.replace(
            old_import,
            new_import,
            1,
        )

    elif (
        "from kornia.geometry.transform.pyramid" in text
        and "pad," in text
    ):
        # Refuse to guess at an unexpected source layout.
        raise RuntimeError(
            "LTXVideo pyramid_blending.py uses an "
            "unexpected Kornia import layout; refusing "
            "to patch it blindly."
        )

    if "import torch.nn.functional as F" not in text:
        if "import torch\n" in text:
            text = text.replace(
                "import torch\n",
                "import torch\n"
                "import torch.nn.functional as F\n",
                1,
            )
        else:
            text = (
                "import torch.nn.functional as F\n"
                + text
            )

    if "pad = F.pad" not in text:
        marker = "import torch.nn.functional as F"
        text = text.replace(
            marker,
            marker
            + "\n\n"
            + "# Kornia >= 0.8.3 compatibility.\n"
            + "pad = F.pad",
            1,
        )

    if text != original:
        pyramid_file.write_text(
            text,
            encoding="utf-8",
        )
        print(
            "✅ Applied LTXVideo/Kornia compatibility patch."
        )
    else:
        print(
            "✅ LTXVideo/Kornia compatibility already correct."
        )


# ============================================================
# CRITICAL IMPORT VALIDATION
# ============================================================

def verify_critical_imports():

    print()
    print("=" * 70)
    print(
        "VERIFYING CRITICAL IMPORTS"
    )
    print("=" * 70)

    modules = [

        "comfy_aimdo",

        "comfy_kitchen",

        "torchsde",

        "av",

        "gguf",

        "spandrel",

        "simpleeval",

        "comfy_angle",
    ]

    failures = []

    for module_name in modules:

        try:

            importlib.import_module(
                module_name
            )

            print(
                f"{module_name:25} OK"
            )

        except Exception as exc:

            failures.append(
                (
                    module_name,
                    str(exc),
                )
            )

            print(
                f"{module_name:25} FAILED"
            )

            print(
                f"  {exc}"
            )

    # --------------------------------------------------------
    # ComfyUI package checks
    # --------------------------------------------------------

    package_checks = [
        "comfyui-frontend-package",
        "comfyui-workflow-templates",
        "comfyui-embedded-docs",
    ]

    for package in package_checks:

        if package_installed(
            package
        ):

            print(
                f"{package:25} OK"
            )

        else:

            failures.append(
                (
                    package,
                    "package is not installed",
                )
            )

            print(
                f"{package:25} FAILED"
            )

    # --------------------------------------------------------
    # LTXVideo source / revision check
    # --------------------------------------------------------

    ltx_dir = (
        COMFY
        / "custom_nodes"
        / "ComfyUI-LTXVideo"
    )

    if not ltx_dir.exists():

        failures.append(
            (
                "LTXVideo",
                "custom node directory is missing",
            )
        )

        print(
            f"{'LTXVideo':25} FAILED"
        )

    else:

        current = get_git_commit(ltx_dir)

        if current != LTXVIDEO_COMMIT:

            failures.append(
                (
                    "LTXVideo revision",
                    f"expected {LTXVIDEO_COMMIT}, found {current}",
                )
            )

            print(
                f"{'LTXVideo revision':25} FAILED"
            )

        else:

            print(
                f"{'LTXVideo revision':25} OK"
            )

        # pyramid_blending.py only needs compatibility validation
        # when the installed revision actually contains it.
        ltx_file = ltx_dir / "pyramid_blending.py"

        if not ltx_file.exists():

            print(
                f"{'LTXVideo/Kornia':25} N/A (pinned revision has no pyramid_blending.py)"
            )

        else:

            text = ltx_file.read_text(
                encoding="utf-8"
            )

            if "pad = F.pad" in text:
                print(
                    f"{'LTXVideo/Kornia':25} OK"
                )
            else:
                failures.append(
                    (
                        "LTXVideo/Kornia",
                        "compatibility patch missing",
                    )
                )

                print(
                    f"{'LTXVideo/Kornia':25} FAILED"
                )

    if failures:

        message = [
            "",
            "=" * 70,
            "CRITICAL DEPENDENCY CHECK FAILED",
            "=" * 70,
            "",
        ]

        for name, error in failures:

            message.append(
                f"{name}: {error}"
            )

        message.extend(
            [
                "",
                "ComfyUI was NOT started.",
                "Fix the above environment issue first.",
                "",
            ]
        )

        raise RuntimeError(
            "\n".join(message)
        )

    print()
    print(
        "✅ All critical imports/packages passed."
    )


# ============================================================
# MODEL DISCOVERY
# ============================================================

def find_model(filename):

    root = Path(
        "/kaggle/input"
    )

    matches = list(
        root.rglob(filename)
    )

    if not matches:

        raise FileNotFoundError(
            "\nModel file not found:\n"
            f"{filename}\n\n"
            f"Search root:\n{root}"
        )

    if len(matches) > 1:

        print(
            f"WARNING: multiple copies found "
            f"for {filename}"
        )

    path = matches[0]

    print(
        f"FOUND {filename}\n"
        f"  {path}"
    )

    return path


def find_all_models():

    print()
    print("=" * 70)
    print(
        "SEARCHING KAGGLE MODEL DATASETS"
    )
    print("=" * 70)

    models = {}

    for key, filename in (
        MODEL_FILES.items()
    ):

        models[key] = find_model(
            filename
        )

    return models


# ============================================================
# COMFYUI MODEL PATHS
# ============================================================

def create_model_paths(
    models
):

    ltx_q4 = models["ltx_q4"]

    vae = models["vae"]

    t5 = models["t5"]

    detailer = models["detailer"]

    upscaler = models["upscaler"]

    config = f"""# Automatically generated by kaggle/launch.py
# Do not edit manually.

ltx_project:
    is_default: true

    diffusion_models: |
        {ltx_q4.parent}

    unet: |
        {ltx_q4.parent}

    text_encoders: |
        {t5.parent}

    clip: |
        {t5.parent}

    vae: |
        {vae.parent}

    loras: |
        {detailer.parent}

    upscale_models: |
        {upscaler.parent}
"""

    EXTRA_PATHS.write_text(
        config,
        encoding="utf-8",
    )

    print()
    print(
        "✅ Created ComfyUI model configuration:"
    )

    print(
        EXTRA_PATHS
    )


# ============================================================
# CUSTOM NODE CHECK
# ============================================================

def verify_nodes():

    print()
    print("=" * 70)
    print(
        "CUSTOM NODE CHECK"
    )
    print("=" * 70)

    custom_nodes = (
        COMFY / "custom_nodes"
    )

    for node in REQUIRED_CUSTOM_NODES:

        path = (
            custom_nodes / node
        )

        if not path.exists():

            raise RuntimeError(
                f"Missing custom node:\n"
                f"{node}"
            )

        print(
            f"{node:30} OK"
        )


# ============================================================
# PORT CHECK
# ============================================================

def port_open():

    try:

        with socket.create_connection(
            (
                "127.0.0.1",
                PORT,
            ),
            timeout=1,
        ):

            return True

    except OSError:

        return False


# ============================================================
# START COMFYUI + CLOUDFLARE
# ============================================================

def start_tunnel():

    if not TUNNEL.exists():

        raise FileNotFoundError(
            f"Tunnel launcher not found:\n"
            f"{TUNNEL}"
        )

    print()
    print("=" * 70)
    print(
        "STARTING COMFYUI + CLOUDFLARE"
    )
    print("=" * 70)

    run(
        [
            sys.executable,
            "-u",
            str(TUNNEL),
        ]
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print(
        "LTX-13B AI VIDEO SYSTEM"
    )
    print(
        "ONE-CELL STARTUP"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # 1. GPU / Torch
    # --------------------------------------------------------

    check_gpu()

    # --------------------------------------------------------
    # 2. Repository
    # --------------------------------------------------------

    ensure_repository()

    # --------------------------------------------------------
    # 3. ComfyUI
    # --------------------------------------------------------

    ensure_comfyui()

    # --------------------------------------------------------
    # 4. Runtime dependencies
    # --------------------------------------------------------

    ensure_runtime_dependencies()

    # --------------------------------------------------------
    # 5. Custom-node requirements
    # --------------------------------------------------------
    ensure_custom_nodes()
    
    install_custom_node_requirements()

    # --------------------------------------------------------
    # 6. LTXVideo revision / Kornia compatibility
    # --------------------------------------------------------

    ensure_ltxvideo_revision()

    patch_ltx_kornia_compatibility()

    # --------------------------------------------------------
    # 7. Critical import validation
    # --------------------------------------------------------

    verify_critical_imports()

    # --------------------------------------------------------
    # 8. Model discovery
    # --------------------------------------------------------

    models = find_all_models()

    # --------------------------------------------------------
    # 9. ComfyUI model configuration
    # --------------------------------------------------------

    create_model_paths(
        models
    )

    # --------------------------------------------------------
    # 10. Custom nodes
    # --------------------------------------------------------

    verify_nodes()

    # --------------------------------------------------------
    # 11. Already-running protection
    # --------------------------------------------------------

    if port_open():

        print()
        print(
            "✅ ComfyUI is already running "
            f"on port {PORT}."
        )

        print(
            "No second instance will be started."
        )

        return

    # --------------------------------------------------------
    # 12. Start ComfyUI + tunnel
    # --------------------------------------------------------

    start_tunnel()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
