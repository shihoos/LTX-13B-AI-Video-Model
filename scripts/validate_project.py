from __future__ import annotations

import importlib.metadata
import json
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)

LOCK_FILE = (
    PROJECT_ROOT
    / "kaggle"
    / "compatibility_lock.yaml"
)


def fail(message: str) -> None:
    raise RuntimeError(message)


def check_file(path: Path) -> None:
    if not path.exists():
        fail(
            f"Missing file:\n{path}"
        )

    print(
        f"OK   {path}"
    )


def check_json(path: Path) -> dict:
    try:
        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

    except Exception as error:
        fail(
            "Invalid JSON:\n"
            f"{path}\n"
            f"{error}"
        )

    if not isinstance(data, dict):
        fail(
            f"JSON root must be an object:\n{path}"
        )

    print(
        f"OK   JSON {path}"
    )

    return data


def load_lock() -> dict:
    check_file(LOCK_FILE)

    try:
        import yaml

    except Exception as error:
        fail(
            "PyYAML unavailable:\n"
            f"{error}"
        )

    try:
        with LOCK_FILE.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = yaml.safe_load(file)

    except Exception as error:
        fail(
            "Could not parse compatibility_lock.yaml:\n"
            f"{error}"
        )

    if not isinstance(data, dict):
        fail(
            "compatibility_lock.yaml "
            "is not a valid mapping."
        )

    required_sections = {
        "comfyui",
        "python_runtime",
        "custom_nodes",
        "legacy_ltx_098_compat",
        "detailer_compat",
        "models",
        "validation",
    }

    missing = (
        required_sections
        - set(data)
    )

    if missing:
        fail(
            "Compatibility lock is missing:\n"
            + "\n".join(
                sorted(missing)
            )
        )

    print(
        "OK   compatibility lock structure"
    )

    return data


def get_git_revision(
    path: Path,
) -> str:

    return (
        subprocess.check_output(
            [
                "git",
                "-C",
                str(path),
                "rev-parse",
                "HEAD",
            ],
            text=True,
        )
        .strip()
    )


def check_git_revision(
    path: Path,
    expected: str,
) -> None:

    if not (
        path
        / ".git"
    ).exists():
        fail(
            f"Not a Git repository:\n{path}"
        )

    actual = get_git_revision(path)

    if actual != expected:
        fail(
            "Git revision mismatch:\n"
            f"Path:     {path}\n"
            f"Expected: {expected}\n"
            f"Actual:   {actual}"
        )

    print(
        f"OK   {path.name}: {actual}"
    )


def validate_git_stack(
    lock: dict,
) -> None:

    comfy = lock[
        "comfyui"
    ]

    comfy_dir = (
        PROJECT_ROOT
        / "ComfyUI"
    )

    # ComfyUI is deliberately not committed into the project.
    # It is materialized by kaggle/bootstrap.py.
    if not comfy_dir.exists():
        print(
            "SKIP Git runtime tree: "
            "ComfyUI has not been materialized locally."
        )
        return

    check_git_revision(
        comfy_dir,
        comfy[
            "commit"
        ],
    )

    custom_dir = (
        comfy_dir
        / "custom_nodes"
    )

    for name, spec in (
        lock[
            "custom_nodes"
        ].items()
    ):

        node_path = (
            custom_dir
            / name
        )

        if node_path.exists():

            check_git_revision(
                node_path,
                spec[
                    "commit"
                ],
            )

        else:

            print(
                f"SKIP {name}: "
                "runtime custom node not materialized."
            )


def validate_runtime_packages(
    lock: dict,
) -> None:

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

    print(
        "PACKAGE LOCK CHECK"
    )

    for (
        package,
        expected_version,
    ) in expected.items():

        try:
            actual = (
                importlib.metadata
                .version(package)
            )

        except importlib.metadata.PackageNotFoundError:
            print(
                f"SKIP {package}: "
                "runtime package is not installed "
                "in this environment."
            )
            continue

        if actual != expected_version:
            fail(
                f"{package} mismatch.\n"
                f"Expected: {expected_version}\n"
                f"Actual:   {actual}"
            )

        print(
            f"OK   {package}=={actual}"
        )


def validate_torch(
    lock: dict,
) -> None:

    try:
        import torch

    except Exception as error:
        print(
            "SKIP Torch runtime check: "
            f"{error}"
        )
        return

    runtime = lock[
        "python_runtime"
    ]

    expected_torch = runtime[
        "torch"
    ]

    if torch.__version__ != expected_torch:
        fail(
            "Torch mismatch.\n"
            f"Expected: {expected_torch}\n"
            f"Actual:   {torch.__version__}"
        )

    print(
        f"OK   torch=={torch.__version__}"
    )

    try:
        import torchvision

    except Exception as error:
        fail(
            "torchvision import failed:\n"
            f"{error}"
        )

    expected_torchvision = runtime[
        "torchvision"
    ]

    if (
        torchvision.__version__
        != expected_torchvision
    ):
        fail(
            "torchvision mismatch.\n"
            f"Expected: {expected_torchvision}\n"
            f"Actual:   {torchvision.__version__}"
        )

    print(
        f"OK   torchvision=="
        f"{torchvision.__version__}"
    )


def validate_init_files() -> None:

    required = [
        "execution/__init__.py",
        "pipeline/__init__.py",
        "planner/__init__.py",
        "schemas/__init__.py",
        "scheduler/__init__.py",
    ]

    for relative in required:
        check_file(
            PROJECT_ROOT
            / relative
        )

    print(
        "OK   all package __init__.py files"
    )


def validate_general_structure() -> None:

    required = [
        "planner/config.py",
        "planner/qwen_loader.py",
        "planner/story_planner.py",
        "planner/character_detector.py",
        "planner/character_planner.py",
        "planner/scene_planner.py",
        "planner/shot_planner.py",

        "pipeline/continuity_manager.py",
        "pipeline/modes.py",
        "pipeline/production_manager.py",
        "pipeline/production_orchestrator.py",
        "pipeline/reference_manager.py",

        "execution/checkpoint_manager.py",
        "execution/comfy_client.py",
        "execution/comfy_workflow_adapter.py",
        "execution/shot_executor.py",
        "execution/assembly_manager.py",
        "execution/production_runner.py",

        "scheduler/gpu_scheduler.py",
        "scheduler/shot_queue.py",

        "schemas/character.py",
        "schemas/scene.py",
        "schemas/shot.py",
        "schemas/parser.py",

        "kaggle/bootstrap.py",
        "kaggle/config.py",
        "kaggle/compatibility_lock.yaml",
        "kaggle/launch.py",
        "kaggle/model_paths.yaml",
        "kaggle/preflight_modern.py",
        "kaggle/start_comfyui.py",
        "kaggle/start_comfyui_tunnel.py",

        "compatibility/prepare_modern_ltx.py",

        "workflows/baseline/"
        "ltxv-13b-dist-i2v-base.json",

        "workflows/detailer/"
        "ltxv-13b-098-ic-lora-upscale.json",
    ]

    forbidden_runtime_file = (
        PROJECT_ROOT
        / "kaggle"
        / "runtime_requirements.lock"
    )

    if forbidden_runtime_file.exists():
        fail(
            "Deleted duplicate runtime lock "
            "still exists:\n"
            f"{forbidden_runtime_file}"
        )

    for relative in required:
        check_file(
            PROJECT_ROOT
            / relative
        )


def validate_workflows() -> None:

    base = (
        PROJECT_ROOT
        / "workflows"
        / "baseline"
        / "ltxv-13b-dist-i2v-base.json"
    )

    detailer = (
        PROJECT_ROOT
        / "workflows"
        / "detailer"
        / "ltxv-13b-098-ic-lora-upscale.json"
    )

    base_data = check_json(base)
    detailer_data = check_json(detailer)

    base_types = {
        node.get("type")
        for node in base_data.get(
            "nodes",
            [],
        )
        if isinstance(
            node,
            dict,
        )
    }

    detailer_types = {
        node.get("type")
        for node in detailer_data.get(
            "nodes",
            [],
        )
        if isinstance(
            node,
            dict,
        )
    }

    required_base = {
        "LTXVBaseSampler",
        "LTXVConditioning",
        "STGGuiderAdvanced",
        "FloatToSigmas",
        "StringToFloatList",
        "UnetLoaderGGUF",
        "CLIPLoaderGGUF",
        "VAELoader",
        "Set VAE Decoder Noise",
        "VHS_VideoCombine",
    }

    required_detailer = {
        "VHS_LoadVideo",
        "LTXVLoopingSampler",
        "LTXVLatentUpsampler",
        "LTXVLatentUpsamplerModelLoader",
        "LTXVTiledVAEDecode",
        "LTXVFilmGrain",
        "LoraLoaderModelOnly",
        "VHS_VideoCombine",
    }

    missing_base = (
        required_base
        - base_types
    )

    missing_detailer = (
        required_detailer
        - detailer_types
    )

    if missing_base:
        fail(
            "BASE workflow missing node types:\n"
            + "\n".join(
                sorted(missing_base)
            )
        )

    if missing_detailer:
        fail(
            "DETAILER workflow missing node types:\n"
            + "\n".join(
                sorted(missing_detailer)
            )
        )

    print(
        "OK   BASE workflow node set"
    )

    print(
        "OK   DETAILER workflow node set"
    )


def validate_compatibility_source(
    lock: dict,
) -> None:

    path = (
        PROJECT_ROOT
        / "compatibility"
        / "prepare_modern_ltx.py"
    )

    check_file(path)

    text = path.read_text(
        encoding="utf-8"
    )

    required_markers = [
        "load_lock",
        "get_legacy_commit",
        "compatibility_lock.yaml",
        "blur_internal(image, blur_radius)",
        "LTXVBaseSampler",
        "LTXVLoopingSampler",
        "LTXVTiledSampler",
        "LTXVTiledVAEDecode",
        "LTXVLatentUpsampler",
        "LTXVLatentUpsamplerModelLoader",
        "LTXVFilmGrain",
        "STGGuiderAdvanced",
        "Set VAE Decoder Noise",
    ]

    for marker in required_markers:
        if marker not in text:
            fail(
                "Compatibility builder is missing:\n"
                f"{marker}"
            )

    legacy_commit = (
        lock[
            "legacy_ltx_098_compat"
        ][
            "commit"
        ]
    )

    if legacy_commit in text:
        fail(
            "Legacy LTX commit is still hardcoded "
            "inside prepare_modern_ltx.py.\n"
            "It must come from compatibility_lock.yaml."
        )

    if (
        "post_processing.Blur().blur("
        in text
    ):
        fail(
            "Legacy Blur().blur() call remains "
            "in compatibility source."
        )

    print(
        "OK   compatibility builder"
    )


def validate_preflight_source(
    lock: dict,
) -> None:

    path = (
        PROJECT_ROOT
        / "kaggle"
        / "preflight_modern.py"
    )

    check_file(path)

    text = path.read_text(
        encoding="utf-8"
    )

    required_markers = [
        "compatibility_lock.yaml",
        "python_runtime",
        "models",
        "verify_gpu",
        "verify_locked_packages",
    ]

    for marker in required_markers:
        if marker not in text:
            fail(
                "Preflight is missing:\n"
                f"{marker}"
            )

    expected_torch = (
        lock[
            "python_runtime"
        ][
            "torch"
        ]
    )

    if expected_torch in text:
        fail(
            "Torch version is still hardcoded "
            "inside preflight_modern.py."
        )

    print(
        "OK   modern preflight"
    )


def validate_config_source(
    lock: dict,
) -> None:

    path = (
        PROJECT_ROOT
        / "kaggle"
        / "config.py"
    )

    check_file(path)

    text = path.read_text(
        encoding="utf-8"
    )

    required_markers = [
        "compatibility_lock.yaml",
        "MODELS",
        "Q4_MODEL",
        "VAE_MODEL",
        "T5_MODEL",
        "DETAILER_LORA",
        "SPATIAL_UPSCALER",
    ]

    for marker in required_markers:
        if marker not in text:
            fail(
                "kaggle/config.py is missing:\n"
                f"{marker}"
            )

    # Confirm that actual model filenames are not duplicated
    # as independent hardcoded values in config.py.
    for name, spec in (
        lock[
            "models"
        ].items()
    ):

        filename = spec[
            "filename"
        ]

        if filename in text:
            fail(
                "Model filename is still duplicated "
                f"inside kaggle/config.py: {filename}"
            )

    print(
        "OK   central model configuration"
    )


def validate_adapter() -> None:

    path = (
        PROJECT_ROOT
        / "execution"
        / "comfy_workflow_adapter.py"
    )

    check_file(path)

    text = path.read_text(
        encoding="utf-8"
    )

    required_markers = [
        "LTXVLatentUpsamplerModelLoader",
        "LatentUpscaleModelLoader",
        "apply_modern_compatibility",
        "validate_modern_detailer",
        "set_input_video",
    ]

    for marker in required_markers:
        if marker not in text:
            fail(
                "Workflow adapter missing:\n"
                f"{marker}"
            )

    print(
        "OK   modern workflow adapter"
    )


def validate_executor() -> None:

    path = (
        PROJECT_ROOT
        / "execution"
        / "shot_executor.py"
    )

    check_file(path)

    text = path.read_text(
        encoding="utf-8"
    )

    required_markers = [
        "_copy_raw_to_comfy_input",
        "VHS_LoadVideo",
        "execute_raw",
        "execute_detailer",
        "mark_raw_complete",
        "mark_upscaled_complete",
    ]

    for marker in required_markers:
        if marker not in text:
            fail(
                "ShotExecutor missing:\n"
                f"{marker}"
            )

    print(
        "OK   BASE→DETAILER physical handoff"
    )


def validate_model_lock(
    lock: dict,
) -> None:

    models = lock[
        "models"
    ]

    required_model_keys = {
        "ltx_q4",
        "t5_q4",
        "vae",
        "ic_lora",
        "spatial_upscaler",
    }

    missing = (
        required_model_keys
        - set(models)
    )

    if missing:
        fail(
            "Model lock missing:\n"
            + "\n".join(
                sorted(missing)
            )
        )

    for name, spec in (
        models.items()
    ):

        if not spec.get("dataset"):
            fail(
                f"{name} has no dataset path."
            )

        if not spec.get("filename"):
            fail(
                f"{name} has no filename."
            )

        if not spec.get("target"):
            fail(
                f"{name} has no ComfyUI target."
            )

        print(
            f"OK   model lock: {name}"
        )


def validate_frontend_policy(
    lock: dict,
) -> None:

    frontend = lock[
        "comfyui"
    ][
        "frontend"
    ]

    package = frontend[
        "package"
    ]

    version = frontend[
        "version"
    ]

    if not package:
        fail(
            "Frontend package is not defined "
            "in compatibility_lock.yaml."
        )

    if not version:
        fail(
            "Frontend version is not defined "
            "in compatibility_lock.yaml."
        )

    launch_path = (
        PROJECT_ROOT
        / "kaggle"
        / "start_comfyui_tunnel.py"
    )

    start_path = (
        PROJECT_ROOT
        / "kaggle"
        / "start_comfyui.py"
    )

    for path in (
        launch_path,
        start_path,
    ):

        text = path.read_text(
            encoding="utf-8"
        )

        if (
            "--front-end-version"
            in text
        ):
            fail(
                f"{path} contains a frontend override. "
                "Production startup must use the "
                "locked frontend package."
            )

        if "@latest" in text:
            fail(
                f"{path} contains @latest. "
                "Production startup must not use "
                "an unpinned frontend."
            )

    print(
        "OK   pinned frontend policy"
    )


def validate_runtime_lock_policy(
) -> None:

    forbidden = (
        PROJECT_ROOT
        / "kaggle"
        / "runtime_requirements.lock"
    )

    if forbidden.exists():
        fail(
            "Duplicate runtime lock still exists:\n"
            f"{forbidden}"
        )

    print(
        "OK   duplicate runtime lock removed"
    )


def validate_external_tools() -> None:

    if (
        shutil.which("ffmpeg")
        is None
    ):
        print(
            "WARNING ffmpeg not found "
            "in current environment."
        )
    else:
        print(
            "OK   ffmpeg"
        )


def main():

    print(
        "=" * 80
    )

    print(
        "LTX-13B FINAL MODERN PROJECT VALIDATION"
    )

    print(
        "=" * 80
    )

    lock = load_lock()

    validate_general_structure()

    validate_init_files()

    validate_runtime_lock_policy()

    validate_git_stack(lock)

    validate_torch(lock)

    validate_runtime_packages(lock)

    validate_model_lock(lock)

    validate_preflight_source(lock)

    validate_compatibility_source(lock)

    validate_config_source(lock)

    validate_frontend_policy(lock)

    validate_adapter()

    validate_executor()

    validate_workflows()

    validate_external_tools()

    print()
    print(
        "=" * 80
    )

    print(
        "✅ FINAL MODERN PROJECT VALIDATION PASSED"
    )

    print(
        "=" * 80
    )

    print(
        "Single source of truth:"
    )

    print(
        "    kaggle/compatibility_lock.yaml"
    )

    print(
        "Duplicate runtime lock:"
    )

    print(
        "    REMOVED"
    )

    print(
        "Frontend:"
    )

    print(
        "    PINNED"
    )

    print(
        "Frontend @latest override:"
    )

    print(
        "    NONE"
    )

    print(
        "BASE workflow:"
    )

    print(
        "    VALIDATED"
    )

    print(
        "DETAILER workflow:"
    )

    print(
        "    VALIDATED"
    )

    print(
        "Compatibility layer:"
    )

    print(
        "    VALIDATED"
    )

    print(
        "Package __init__.py files:"
    )

    print(
        "    VALIDATED"
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":

    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

    main()
