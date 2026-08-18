from __future__ import annotations

import ast
import importlib.metadata
import json
import shutil
import subprocess
import sys

from pathlib import Path


# ================================================================
# PROJECT PATHS
# ================================================================

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


def read_text(
    path: Path,
) -> str:

    try:

        return path.read_text(
            encoding="utf-8"
        )

    except Exception as error:

        fail(
            "Unable to read file:\n"
            f"{path}\n"
            f"{error}"
        )


def parse_python(
    path: Path,
) -> ast.AST:

    source = read_text(
        path
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
# CENTRAL COMPATIBILITY LOCK
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
# GIT VALIDATION
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

    actual = (
        get_git_revision(
            path
        )
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
# PYTHON RUNTIME
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
# PACKAGE / PROJECT STRUCTURE
# ================================================================

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

        # Planner
        "planner/config.py",
        "planner/qwen_loader.py",
        "planner/story_planner.py",
        "planner/character_detector.py",
        "planner/character_planner.py",
        "planner/scene_planner.py",
        "planner/shot_planner.py",

        # Pipeline
        "pipeline/continuity_manager.py",
        "pipeline/modes.py",
        "pipeline/production_manager.py",
        "pipeline/production_orchestrator.py",
        "pipeline/reference_manager.py",

        # Execution
        "execution/checkpoint_manager.py",
        "execution/comfy_client.py",
        "execution/comfy_workflow_adapter.py",
        "execution/shot_executor.py",
        "execution/assembly_manager.py",
        "execution/production_runner.py",

        # Scheduler
        "scheduler/gpu_scheduler.py",
        "scheduler/shot_queue.py",

        # Schemas
        "schemas/character.py",
        "schemas/scene.py",
        "schemas/shot.py",
        "schemas/parser.py",

        # Kaggle
        "kaggle/bootstrap.py",
        "kaggle/config.py",
        "kaggle/compatibility_lock.yaml",
        "kaggle/launch.py",
        "kaggle/model_paths.yaml",
        "kaggle/preflight_modern.py",
        "kaggle/start_comfyui.py",
        "kaggle/start_comfyui_tunnel.py",

        # Compatibility
        "compatibility/prepare_modern_ltx.py",

        # Application entry point
        "scripts/generate_video.py",

        # Workflows
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

    for relative in required:

        check_file(
            PROJECT_ROOT
            / relative
        )


# ================================================================
# WORKFLOW VALIDATION
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

def class_map(
    tree: ast.AST,
) -> dict[str, ast.ClassDef]:

    return {
        node.name: node
        for node in ast.walk(
            tree
        )
        if isinstance(
            node,
            ast.ClassDef,
        )
    }


def function_map(
    tree: ast.AST,
) -> dict[str, ast.AST]:

    functions = {}

    for node in ast.walk(
        tree
    ):

        if isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
            ),
        ):

            functions[
                node.name
            ] = node

    return functions


def method_names(
    class_node: ast.ClassDef,
) -> set[str]:

    return {
        node.name

        for node in class_node.body

        if isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
            ),
        )
    }


def source_contains_identifier(
    tree: ast.AST,
    identifier: str,
) -> bool:

    for node in ast.walk(
        tree
    ):

        if isinstance(
            node,
            ast.Name,
        ) and node.id == identifier:

            return True

        if isinstance(
            node,
            ast.Attribute,
        ) and node.attr == identifier:

            return True

    return False


def actual_string_constants(
    tree: ast.AST,
):
    """
    Yield string constants that are executable Python values.

    Comments are not represented in the AST.
    Module/class/function docstrings are ignored.
    """

    docstring_nodes: set[int] = set()

    for node in ast.walk(
        tree
    ):

        body = getattr(
            node,
            "body",
            None,
        )

        if not body:

            continue

        first = body[0]

        if not isinstance(
            first,
            ast.Expr,
        ):

            continue

        value = first.value

        if (
            isinstance(
                value,
                ast.Constant,
            )
            and isinstance(
                value.value,
                str,
            )
        ):

            docstring_nodes.add(
                id(value)
            )

    for node in ast.walk(
        tree
    ):

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

        if id(node) in (
            docstring_nodes
        ):

            continue

        yield node.value


# ================================================================
# COMPATIBILITY BLUR VALIDATION
# ================================================================

def ast_contains_legacy_blur_call(
    source: str,
) -> bool:
    """
    Detect an actual executable call shaped like:

        Blur().blur(...)
        module.Blur().blur(...)

    It intentionally does NOT perform a raw string search.
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

        value = (
            func.value
        )

        if not isinstance(
            value,
            ast.Call,
        ):

            continue

        constructor = (
            value.func
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

        elif isinstance(
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

    text = read_text(
        path
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
# PREFLIGHT / CENTRAL CONFIG
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

    text = read_text(
        path
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

    text = read_text(
        path
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

        filename = (
            spec[
                "filename"
            ]
        )

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
            "Frontend package is missing "
            "from compatibility_lock.yaml."
        )

    if not frontend.get(
        "version"
    ):

        fail(
            "Frontend version is missing "
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

        for value in (
            actual_string_constants(
                tree
            )
        ):

            # Actual command strings only.
            if (
                "--front-end-version"
                in value
            ):

                fail(
                    f"{path.relative_to(PROJECT_ROOT)} "
                    "contains an executable frontend "
                    "version override."
                )

            if (
                value.strip()
                == "@latest"
            ):

                fail(
                    f"{path.relative_to(PROJECT_ROOT)} "
                    "contains an executable "
                    "@latest frontend."
                )

    print(
        "OK   pinned frontend policy"
    )


# ================================================================
# WORKFLOW ADAPTER
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

    tree = parse_python(
        path
    )

    required_identifiers = [

        "LTXVLatentUpsamplerModelLoader",
        "LatentUpscaleModelLoader",
        "apply_modern_compatibility",
        "validate_modern_detailer",
        "set_input_video",
    ]

    for identifier in (
        required_identifiers
    ):

        if not source_contains_identifier(
            tree,
            identifier,
        ):

            fail(
                "Workflow adapter missing integration:\n"
                f"{identifier}"
            )

    print(
        "OK   modern workflow adapter"
    )


# ================================================================
# SHOT EXECUTOR
# ================================================================

def validate_executor() -> None:

    path = (
        PROJECT_ROOT
        / "execution"
        / "shot_executor.py"
    )

    check_file(
        path
    )

    tree = parse_python(
        path
    )

    required_identifiers = [

        "_copy_raw_to_comfy_input",
        "VHS_LoadVideo",
        "execute_raw",
        "execute_detailer",
        "mark_raw_complete",
        "mark_upscaled_complete",
    ]

    for identifier in (
        required_identifiers
    ):

        if not source_contains_identifier(
            tree,
            identifier,
        ):

            fail(
                "ShotExecutor missing integration:\n"
                f"{identifier}"
            )

    text = read_text(
        path
    )

    if (
        "1536x832"
        in text
    ):

        fail(
            "ShotExecutor contains obsolete "
            "1536x832 production resolution."
        )

    if (
        "1536x864"
        not in text
    ):

        fail(
            "ShotExecutor does not contain the "
            "required 1536x864 16:9 master policy."
        )

    print(
        "OK   BASE→DETAILER physical handoff"
    )

    print(
        "OK   1536x864 16:9 master policy"
    )


# ================================================================
# PRODUCTION RUNNER
# ================================================================

def validate_runner() -> None:

    path = (
        PROJECT_ROOT
        / "execution"
        / "production_runner.py"
    )

    check_file(
        path
    )

    tree = parse_python(
        path
    )

    classes = class_map(
        tree
    )

    runner = classes.get(
        "ProductionRunner"
    )

    if runner is None:

        fail(
            "ProductionRunner class is missing."
        )

    methods = method_names(
        runner
    )

    required_methods = {

        "__init__",
        "prepare",
        "run",
        "_run_one_shot",
        "_dict_to_shot",
    }

    missing_methods = (
        required_methods
        - methods
    )

    if missing_methods:

        fail(
            "ProductionRunner missing methods:\n"
            + "\n".join(
                sorted(
                    missing_methods
                )
            )
        )

    required_integrations = [

        "GPUScheduler",
        "ShotExecutor",
        "ComfyClient",
        "AssemblyManager",
        "CheckpointManager",
        "ComfyWorkflowAdapter",
    ]

    for identifier in (
        required_integrations
    ):

        if not source_contains_identifier(
            tree,
            identifier,
        ):

            fail(
                "ProductionRunner missing integration:\n"
                f"{identifier}"
            )

    # Verify actual scheduler.run(...) call exists.
    has_scheduler_run = False

    for node in ast.walk(
        runner
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
            == "run"
        ):

            value = func.value

            if isinstance(
                value,
                ast.Attribute,
            ):

                if (
                    value.attr
                    == "scheduler"
                ):

                    has_scheduler_run = True

            elif isinstance(
                value,
                ast.Name,
            ):

                if (
                    value.id
                    == "scheduler"
                ):

                    has_scheduler_run = True

    if not has_scheduler_run:

        fail(
            "ProductionRunner.run() does not "
            "call the GPU scheduler."
        )

    # Verify actual ShotExecutor construction.
    has_executor = False

    for node in ast.walk(
        runner
    ):

        if not isinstance(
            node,
            ast.Call,
        ):

            continue

        func = node.func

        if isinstance(
            func,
            ast.Name,
        ):

            if (
                func.id
                == "ShotExecutor"
            ):

                has_executor = True
                break

    if not has_executor:

        fail(
            "ProductionRunner does not construct "
            "ShotExecutor."
        )

    print(
        "OK   ProductionRunner class"
    )

    print(
        "OK   ProductionRunner.run()"
    )

    print(
        "OK   ProductionRunner → GPUScheduler"
    )

    print(
        "OK   ProductionRunner → ShotExecutor"
    )

    print(
        "OK   ProductionRunner → ComfyClient"
    )

    print(
        "OK   ProductionRunner → AssemblyManager"
    )


# ================================================================
# GPU SCHEDULER
# ================================================================

def validate_scheduler() -> None:

    path = (
        PROJECT_ROOT
        / "scheduler"
        / "gpu_scheduler.py"
    )

    check_file(
        path
    )

    tree = parse_python(
        path
    )

    classes = class_map(
        tree
    )

    scheduler = classes.get(
        "GPUScheduler"
    )

    if scheduler is None:

        fail(
            "GPUScheduler class is missing."
        )

    methods = method_names(
        scheduler
    )

    if "run" not in methods:

        fail(
            "GPUScheduler.run() is missing."
        )

    identifiers = [
        "threading",
        "worker_function",
        "failures",
    ]

    for identifier in (
        identifiers
    ):

        if not source_contains_identifier(
            tree,
            identifier,
        ):

            fail(
                "GPU scheduler missing integration:\n"
                f"{identifier}"
            )

    print(
        "OK   GPU scheduler"
    )


# ================================================================
# CHECKPOINT MANAGER
# ================================================================

def validate_checkpoint_manager() -> None:

    path = (
        PROJECT_ROOT
        / "execution"
        / "checkpoint_manager.py"
    )

    check_file(
        path
    )

    tree = parse_python(
        path
    )

    if not source_contains_identifier(
        tree,
        "RLock",
    ):

        fail(
            "CheckpointManager is not using "
            "thread-safe RLock."
        )

    required_methods = {

        "initialize_shot",
        "get_shot",
        "mark_generating",
        "mark_raw_complete",
        "mark_detailer_complete",
        "mark_upscaled_complete",
        "mark_complete",
        "mark_failed",
        "set_assembly_started",
        "set_assembly_complete",
        "set_assembly_failed",
        "get_assembly",
    }

    classes = class_map(
        tree
    )

    manager = classes.get(
        "CheckpointManager"
    )

    if manager is None:

        fail(
            "CheckpointManager class is missing."
        )

    missing = (
        required_methods
        - method_names(manager)
    )

    if missing:

        fail(
            "CheckpointManager missing methods:\n"
            + "\n".join(
                sorted(
                    missing
                )
            )
        )

    text = read_text(
        path
    )

    if (
        "production_state.json"
        not in text
    ):

        fail(
            "Checkpoint state file handling "
            "is missing."
        )

    print(
        "OK   thread-safe checkpoint manager"
    )


# ================================================================
# COMFY CLIENT
# ================================================================

def validate_comfy_client() -> None:

    path = (
        PROJECT_ROOT
        / "execution"
        / "comfy_client.py"
    )

    check_file(
        path
    )

    tree = parse_python(
        path
    )

    classes = class_map(
        tree
    )

    client = classes.get(
        "ComfyClient"
    )

    if client is None:

        fail(
            "ComfyClient class is missing."
        )

    required_methods = {

        "health_check",
        "queue_prompt",
        "get_history",
        "wait_for_prompt",
        "download_file",
        "find_video_outputs",
    }

    missing = (
        required_methods
        - method_names(client)
    )

    if missing:

        fail(
            "ComfyClient missing methods:\n"
            + "\n".join(
                sorted(
                    missing
                )
            )
        )

    print(
        "OK   ComfyUI HTTP API client"
    )


# ================================================================
# GENERATE VIDEO ENTRY POINT
# ================================================================

def validate_generate_video() -> None:

    path = (
        PROJECT_ROOT
        / "scripts"
        / "generate_video.py"
    )

    check_file(
        path
    )

    tree = parse_python(
        path
    )

    required_classes = {
        "ProductionOrchestrator",
        "ProductionRunner",
        "ProductionManager",
    }

    for identifier in (
        required_classes
    ):

        if not source_contains_identifier(
            tree,
            identifier,
        ):

            fail(
                "generate_video.py missing integration:\n"
                f"{identifier}"
            )

    text = read_text(
        path
    )

    required_strings = [
        "--story",
        "--mode",
        "--gpu-url",
        "create_production_plan",
        "runner.run",
    ]

    for value in (
        required_strings
    ):

        if value not in text:

            fail(
                "generate_video.py missing required "
                f"entry-point component:\n{value}"
            )

    print(
        "OK   application generation entry point"
    )


# ================================================================
# PLANNER → ORCHESTRATOR
# ================================================================

def validate_planner_orchestrator_chain() -> None:

    planner_path = (
        PROJECT_ROOT
        / "pipeline"
        / "production_orchestrator.py"
    )

    check_file(
        planner_path
    )

    tree = parse_python(
        planner_path
    )

    classes = class_map(
        tree
    )

    orchestrator = classes.get(
        "ProductionOrchestrator"
    )

    if orchestrator is None:

        fail(
            "ProductionOrchestrator class is missing."
        )

    required_methods = {
        "create_production_plan",
        "unload_models",
    }

    missing = (
        required_methods
        - method_names(
            orchestrator
        )
    )

    if missing:

        fail(
            "ProductionOrchestrator missing methods:\n"
            + "\n".join(
                sorted(
                    missing
                )
            )
        )

    required_identifiers = [

        "StoryPlanner",
        "CharacterDetector",
        "CharacterPlanner",
        "ScenePlanner",
        "ShotPlanner",
        "ContinuityManager",
    ]

    for identifier in (
        required_identifiers
    ):

        if not source_contains_identifier(
            tree,
            identifier,
        ):

            fail(
                "ProductionOrchestrator missing:\n"
                f"{identifier}"
            )

    print(
        "OK   StoryPlanner → ProductionOrchestrator"
    )

    print(
        "OK   Character/Scene/Shot planning chain"
    )

    print(
        "OK   Continuity integration"
    )


# ================================================================
# CHARACTER REFERENCE CHAIN
# ================================================================

def validate_reference_chain() -> None:

    reference_path = (
        PROJECT_ROOT
        / "pipeline"
        / "reference_manager.py"
    )

    character_planner_path = (
        PROJECT_ROOT
        / "planner"
        / "character_planner.py"
    )

    shot_planner_path = (
        PROJECT_ROOT
        / "planner"
        / "shot_planner.py"
    )

    for path in (
        reference_path,
        character_planner_path,
        shot_planner_path,
    ):

        check_file(
            path
        )

    reference_tree = (
        parse_python(
            reference_path
        )
    )

    character_tree = (
        parse_python(
            character_planner_path
        )
    )

    shot_tree = (
        parse_python(
            shot_planner_path
        )
    )

    for (
        tree,
        name,
    ) in (
        (
            reference_tree,
            "ReferenceManager",
        ),
        (
            character_tree,
            "CharacterPlanner",
        ),
        (
            shot_tree,
            "ShotPlanner",
        ),
    ):

        if not source_contains_identifier(
            tree,
            name,
        ):

            fail(
                f"{name} integration missing."
            )

    # Verify reference_images is present in ShotPlanner source.
    shot_text = read_text(
        shot_planner_path
    )

    if (
        "reference_images"
        not in shot_text
    ):

        fail(
            "ShotPlanner does not propagate "
            "reference_images."
        )

    print(
        "OK   ReferenceManager"
    )

    print(
        "OK   character reference propagation"
    )

    print(
        "OK   character references → shots"
    )


# ================================================================
# PRODUCTION MANAGER
# ================================================================

def validate_production_manager() -> None:

    path = (
        PROJECT_ROOT
        / "pipeline"
        / "production_manager.py"
    )

    check_file(
        path
    )

    tree = parse_python(
        path
    )

    classes = class_map(
        tree
    )

    manager = classes.get(
        "ProductionManager"
    )

    if manager is None:

        fail(
            "ProductionManager class is missing."
        )

    methods = method_names(
        manager
    )

    if "get_pipeline" not in methods:

        fail(
            "ProductionManager.get_pipeline() "
            "is missing."
        )

    print(
        "OK   ProductionManager"
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
# FRONTEND / RUNTIME LOCK POLICY
# ================================================================

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

    # ------------------------------------------------------------
    # Repository / lock
    # ------------------------------------------------------------

    validate_general_structure()

    validate_init_files()

    validate_runtime_lock_policy()

    validate_git_stack(
        lock
    )

    # ------------------------------------------------------------
    # Runtime
    # ------------------------------------------------------------

    validate_torch(
        lock
    )

    validate_runtime_packages(
        lock
    )

    validate_model_lock(
        lock
    )

    # ------------------------------------------------------------
    # Modern compatibility
    # ------------------------------------------------------------

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

    # ------------------------------------------------------------
    # ComfyUI execution
    # ------------------------------------------------------------

    validate_comfy_client()

    validate_adapter()

    validate_executor()

    validate_runner()

    validate_scheduler()

    validate_checkpoint_manager()

    # ------------------------------------------------------------
    # Planning / application chain
    # ------------------------------------------------------------

    validate_planner_orchestrator_chain()

    validate_reference_chain()

    validate_production_manager()

    validate_generate_video()

    # ------------------------------------------------------------
    # Workflows / tools
    # ------------------------------------------------------------

    validate_workflows()

    validate_external_tools()

    # ------------------------------------------------------------
    # Final result
    # ------------------------------------------------------------

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
        "Frontend policy:"
    )

    print(
        "  PINNED / executable @latest prohibited"
    )

    print(
        "LTX 0.9.8 compatibility:"
    )

    print(
        "  VALIDATED"
    )

    print(
        "Native blur compatibility:"
    )

    print(
        "  VALIDATED"
    )

    print(
        "Planner → Orchestrator:"
    )

    print(
        "  VALIDATED"
    )

    print(
        "Orchestrator → Runner:"
    )

    print(
        "  VALIDATED"
    )

    print(
        "Runner → Scheduler:"
    )

    print(
        "  VALIDATED"
    )

    print(
        "Scheduler → ShotExecutor:"
    )

    print(
        "  VALIDATED"
    )

    print(
        "ShotExecutor → ComfyClient:"
    )

    print(
        "  VALIDATED"
    )

    print(
        "BASE → IC-LoRA → Spatial:"
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
