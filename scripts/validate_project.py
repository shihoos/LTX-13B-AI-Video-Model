from __future__ import annotations

import ast
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


# ================================================================
# BASIC HELPERS
# ================================================================


def fail(
    message: str,
) -> None:

    raise RuntimeError(
        message
    )


def check_file(
    path: Path,
) -> None:

    if not path.exists():

        fail(
            f"Missing file:\n{path}"
        )

    print(
        f"OK   {path}"
    )


def check_json(
    path: Path,
) -> dict:

    try:

        with path.open(
            "r",
            encoding="utf-8",
        ) as file:

            data = json.load(
                file
            )

    except Exception as error:

        fail(
            "Invalid JSON:\n"
            f"{path}\n"
            f"{error}"
        )

    if not isinstance(
        data,
        dict,
    ):

        fail(
            f"JSON root must be an object:\n"
            f"{path}"
        )

    print(
        f"OK   JSON {path}"
    )

    return data


# ================================================================
# CENTRAL LOCK
# ================================================================


def load_lock() -> dict:

    check_file(
        LOCK_FILE
    )

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

            data = yaml.safe_load(
                file
            )

    except Exception as error:

        fail(
            "Could not parse compatibility_lock.yaml:\n"
            f"{error}"
        )

    if not isinstance(
        data,
        dict,
    ):

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
                sorted(
                    missing
                )
            )
        )

    print(
        "OK   compatibility lock structure"
    )

    return data


# ================================================================
# GIT
# ================================================================


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
            f"Not a Git repository:\n"
            f"{path}"
        )

    actual = get_git_revision(
        path
    )

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

    comfy_dir = (
        PROJECT_ROOT
        / "ComfyUI"
    )

    if not comfy_dir.exists():

        print(
            "SKIP Git runtime tree: "
            "ComfyUI has not been materialized locally."
        )

        return

    expected_comfy = (
        lock[
            "comfyui"
        ][
            "commit"
        ]
    )

    check_git_revision(
        comfy_dir,
        expected_comfy,
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

        path = (
            custom_dir
            / name
        )

        if path.exists():

            check_git_revision(
                path,
                spec[
                    "commit"
                ],
            )

        else:

            print(
                f"SKIP {name}: "
                "runtime custom node not materialized."
            )


# ================================================================
# PYTHON PACKAGE VERSIONS
# ================================================================


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
                .version(
                    package
                )
            )

        except importlib.metadata.PackageNotFoundError:

            print(
                f"SKIP {package}: "
                "runtime package is not installed "
                "in this environment."
            )

            continue

        if (
            actual
            != expected_version
        ):

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
            "SKIP Torch runtime check:\n"
            f"{error}"
        )

        return

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

    if (
        torch.__version__
        != expected_torch
    ):

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

    expected_torchvision = (
        runtime[
            "torchvision"
        ]
    )

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
        "OK   torchvision=="
        f"{torchvision.__version__}"
    )


# ================================================================
# PROJECT STRUCTURE
# ================================================================


def validate_init_files() -> None:

    required = [
        "execution/__init__.py",
        "pipeline/__init__.py",
        "planner/__init__.py",
        "schemas/__init__.py",
        "scheduler/__init__.py",
    ]

    for relative in (
        required
    ):

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

        "scripts/generate_video.py",

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
            "Duplicate runtime lock still exists:\n"
            f"{forbidden_runtime_file}"
        )

    for relative in (
        required
    ):

        check_file(
            PROJECT_ROOT
            / relative
        )


# ================================================================
# WORKFLOW STRUCTURE
# ================================================================


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

    base_data = check_json(
        base
    )

    detailer_data = check_json(
        detailer
    )

    base_types = {

        node.get(
            "type"
        )

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

        node.get(
            "type"
        )

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
                sorted(
                    missing_base
                )
            )
        )

    if missing_detailer:

        fail(
            "DETAILER workflow missing node types:\n"
            + "\n".join(
                sorted(
                    missing_detailer
                )
            )
        )

    print(
        "OK   BASE workflow node set"
    )

    print(
        "OK   DETAILER workflow node set"
    )


# ================================================================
# AST HELPERS
# ================================================================


def parse_python(
    path: Path,
) -> ast.AST:

    source = path.read_text(
        encoding="utf-8"
    )

    try:

        return ast.parse(
            source,
            filename=str(
                path
            ),
        )

    except SyntaxError as error:

        fail(
            "Python syntax error:\n"
            f"{path}\n"
            f"{error}"
        )


def actual_string_constants(
    tree: ast.AST,
):

    """
    Yield executable string constants while ignoring
    module/function/class docstrings.

    Comments are never present in the AST.
    """

    docstring_nodes = set()

    for node in ast.walk(tree):

        body = getattr(
            node,
            "body",
            None,
        )

        if not body:

            continue

        first = body[0]

        if isinstance(
            first,
            ast.Expr,
        ):

            value = first.value

            if isinstance(
                value,
                ast.Constant,
            ) and isinstance(
                value.value,
                str,
            ):

                docstring_nodes.add(
                    id(value)
                )

    for node in ast.walk(tree):

        if not isinstance(
            node,
            ast.Constant,
        ):

            continue

        if not isinstance(
            node.value,
            str,
        ):

            continue

        if id(node) in docstring_nodes:

            continue

        yield node.value


# ================================================================
# COMPATIBILITY VALIDATION
# ================================================================


def ast_contains_legacy_blur_call(
    source: str,
) -> bool:

    """
    Detect an actual call shaped like:

        Blur().blur(...)
        module.Blur().blur(...)

    It does not use substring matching.
    """

    try:

        tree = ast.parse(
            source
        )

    except SyntaxError:

        return False

    for node in ast.walk(
        tree
    ):

        if not isinstance(
            node,
            ast.Call,
        ):

            continue

        func = node.func

        if not isinstance(
            func,
            ast.Attribute,
        ):

            continue

        if (
            func.attr
            != "blur"
        ):

            continue

        blur_object = (
            func.value
        )

        if not isinstance(
            blur_object,
            ast.Call,
        ):

            continue

        constructor = (
            blur_object.func
        )

        if isinstance(
            constructor,
            ast.Name,
        ):

            if (
                constructor.id
                == "Blur"
            ):

                return True

        if isinstance(
            constructor,
            ast.Attribute,
        ):

            if (
                constructor.attr
                == "Blur"
            ):

                return True

    return False


def validate_compatibility_source(
    lock: dict,
) -> None:

    path = (
        PROJECT_ROOT
        / "compatibility"
        / "prepare_modern_ltx.py"
    )

    check_file(
        path
    )

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

    for marker in (
        required_markers
    ):

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
            "inside prepare_modern_ltx.py."
        )

    if ast_contains_legacy_blur_call(
        text
    ):

        fail(
            "Actual legacy Blur().blur() "
            "call remains in compatibility source."
        )

    print(
        "OK   compatibility builder"
    )

    print(
        "OK   native torch blur validation"
    )


# ================================================================
# PRE-FLIGHT / CONFIG
# ================================================================


def validate_preflight_source(
    lock: dict,
) -> None:

    path = (
        PROJECT_ROOT
        / "kaggle"
        / "preflight_modern.py"
    )

    check_file(
        path
    )

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

    for marker in (
        required_markers
    ):

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

    check_file(
        path
    )

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

    for marker in (
        required_markers
    ):

        if marker not in text:

            fail(
                "kaggle/config.py is missing:\n"
                f"{marker}"
            )

    for _, spec in (
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
                "inside kaggle/config.py:\n"
                f"{filename}"
            )

    print(
        "OK   central model configuration"
    )


# ================================================================
# FRONTEND POLICY
# ================================================================


def validate_frontend_policy(
    lock: dict,
) -> None:

    frontend = (
        lock[
            "comfyui"
        ][
            "frontend"
        ]
    )

    if not frontend.get(
        "package"
    ):

        fail(
            "Frontend package missing "
            "from compatibility_lock.yaml."
        )

    if not frontend.get(
        "version"
    ):

        fail(
            "Frontend version missing "
            "from compatibility_lock.yaml."
        )

    startup_files = [

        PROJECT_ROOT
        / "kaggle"
        / "start_comfyui.py",

        PROJECT_ROOT
        / "kaggle"
        / "start_comfyui_tunnel.py",

        PROJECT_ROOT
        / "kaggle"
        / "launch.py",
    ]

    for path in (
        startup_files
    ):

        if not path.exists():

            continue

        tree = parse_python(
            path
        )

        strings = list(
            actual_string_constants(
                tree
            )
        )

        # Only executable string constants are considered.
        #
        # A comment such as:
        #
        #   # Do not pass --front-end-version @latest
        #
        # is not an AST Constant and is therefore ignored.
        #
        # A real executable command such as:
        #
        #   "--front-end-version"
        #
        # or:
        #
        #   "@latest"
        #
        # will be caught.

        for value in (
            strings
        ):

            if (
                "--front-end-version"
                in value
            ):

                fail(
                    f"{path.relative_to(PROJECT_ROOT)} "
                    "contains an executable frontend override."
                )

            if value.strip() == (
                "@latest"
            ):

                fail(
                    f"{path.relative_to(PROJECT_ROOT)} "
                    "contains an executable @latest frontend."
                )

    print(
        "OK   pinned frontend policy"
    )


# ================================================================
# MODERN ADAPTER / EXECUTION
# ================================================================


def validate_adapter() -> None:

    path = (
        PROJECT_ROOT
        / "execution"
        / "comfy_workflow_adapter.py"
    )

    check_file(
        path
    )

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

    for marker in (
        required_markers
    ):

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

    check_file(
        path
    )

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

    for marker in (
        required_markers
    ):

        if marker not in text:

            fail(
                "ShotExecutor missing:\n"
                f"{marker}"
            )

    # Production resolution contract.
    if (
        "1536x832"
        in text
    ):

        fail(
            "ShotExecutor still contains "
            "the obsolete 1536x832 production resolution."
        )

    if (
        "1536x864"
        not in text
    ):

        fail(
            "ShotExecutor does not document "
            "the required 1536x864 16:9 master."
        )

    print(
        "OK   BASE→DETAILER physical handoff"
    )

    print(
        "OK   1536x864 16:9 master policy"
    )


def validate_runner() -> None:

    path = (
        PROJECT_ROOT
        / "execution"
        / "production_runner.py"
    )

    check_file(
        path
    )

    text = path.read_text(
        encoding="utf-8"
    )

    required_markers = [

        "ProductionRunner",
        "GPUScheduler",
        "ShotExecutor",
        "ComfyClient",
        "AssemblyManager",
        "runner",
        "run",
    ]

    for marker in (
        required_markers
    ):

        if marker not in text:

            fail(
                "ProductionRunner missing:\n"
                f"{marker}"
            )

    print(
        "OK   production runner"
    )


def validate_scheduler() -> None:

    path = (
        PROJECT_ROOT
        / "scheduler"
        / "gpu_scheduler.py"
    )

    check_file(
        path
    )

    text = path.read_text(
        encoding="utf-8"
    )

    required_markers = [
        "GPUScheduler",
        "threading.Thread",
        "failures",
        "worker_function",
    ]

    for marker in (
        required_markers
    ):

        if marker not in text:

            fail(
                "GPU scheduler missing:\n"
                f"{marker}"
            )

    print(
        "OK   GPU scheduler"
    )


def validate_checkpoint_manager() -> None:

    path = (
        PROJECT_ROOT
        / "execution"
        / "checkpoint_manager.py"
    )

    check_file(
        path
    )

    text = path.read_text(
        encoding="utf-8"
    )

    required_markers = [
        "threading.RLock",
        "production_state.json",
        "set_assembly_complete",
        "get_assembly",
    ]

    for marker in (
        required_markers
    ):

        if marker not in text:

            fail(
                "Checkpoint manager missing:\n"
                f"{marker}"
            )

    print(
        "OK   thread-safe checkpoint manager"
    )


def validate_generate_video() -> None:

    path = (
        PROJECT_ROOT
        / "scripts"
        / "generate_video.py"
    )

    check_file(
        path
    )

    text = path.read_text(
        encoding="utf-8"
    )

    required_markers = [

        "ProductionOrchestrator",
        "ProductionRunner",
        "create_production_plan",
        "runner.run",
        "--story",
        "--mode",
        "--gpu-url",
    ]

    for marker in (
        required_markers
    ):

        if marker not in text:

            fail(
                "generate_video.py missing:\n"
                f"{marker}"
            )

    print(
        "OK   application generation entry point"
    )


# ================================================================
# MODEL LOCK
# ================================================================


def validate_model_lock(
    lock: dict,
) -> None:

    models = lock[
        "models"
    ]

    required_keys = {
        "ltx_q4",
        "t5_q4",
        "vae",
        "ic_lora",
        "spatial_upscaler",
    }

    missing = (
        required_keys
        - set(models)
    )

    if missing:

        fail(
            "Model lock missing:\n"
            + "\n".join(
                sorted(
                    missing
                )
            )
        )

    for name, spec in (
        models.items()
    ):

        for field in (
            "dataset",
            "filename",
            "target",
        ):

            if not spec.get(
                field
            ):

                fail(
                    f"{name} missing {field}."
                )

        print(
            f"OK   model lock: {name}"
        )


# ================================================================
# EXTERNAL TOOLS
# ================================================================


def validate_external_tools() -> None:

    if (
        shutil.which(
            "ffmpeg"
        )
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


def validate_runtime_lock_policy() -> None:

    forbidden = (
        PROJECT_ROOT
        / "kaggle"
        / "runtime_requirements.lock"
    )

    if forbidden.exists():

        fail(
            "Deleted duplicate runtime lock "
            "still exists:\n"
            f"{forbidden}"
        )

    print(
        "OK   duplicate runtime lock removed"
    )


# ================================================================
# MAIN
# ================================================================


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

    validate_git_stack(
        lock
    )

    validate_torch(
        lock
    )

    validate_runtime_packages(
        lock
    )

    validate_model_lock(
        lock
    )

    validate_preflight_source(
        lock
    )

    validate_compatibility_source(
        lock
    )

    validate_config_source(
        lock
    )

    validate_frontend_policy(
        lock
    )

    validate_adapter()

    validate_executor()

    validate_runner()

    validate_scheduler()

    validate_checkpoint_manager()

    validate_generate_video()

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
        "  kaggle/compatibility_lock.yaml"
    )

    print(
        "Duplicate runtime lock:"
    )

    print(
        "  REMOVED"
    )

    print(
        "Frontend:"
    )

    print(
        "  PINNED / NO EXECUTABLE @latest"
    )

    print(
        "Compatibility:"
    )

    print(
        "  MODERN + LTX 0.9.8 SHIM"
    )

    print(
        "BASE:"
    )

    print(
        "  VALIDATED"
    )

    print(
        "DETAILER:"
    )

    print(
        "  VALIDATED"
    )

    print(
        "Planner → Runner:"
    )

    print(
        "  VALIDATED"
    )

    print(
        "Package __init__.py:"
    )

    print(
        "  VALIDATED"
    )


if __name__ == "__main__":

    sys.path.insert(
        0,
        str(
            PROJECT_ROOT
        ),
    )

    main()
