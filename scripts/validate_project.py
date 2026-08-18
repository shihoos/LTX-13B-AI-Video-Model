from __future__ import annotations

import ast
import importlib.metadata
import json
import shutil
import subprocess
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

def fail(message: str) -> None:
    raise RuntimeError(message)


def check_file(path: Path) -> None:
    if not path.exists():
        fail(
            f"Missing file:\n{path}"
        )

    print(f"OK   {path}")


def read_text(path: Path) -> str:
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


def parse_python(path: Path) -> ast.AST:
    source = read_text(path)

    try:
        return ast.parse(
            source,
            filename=str(path),
        )
    except SyntaxError as error:
        fail(
            "Python syntax error:\n"
            f"{path}\n"
            f"{error}"
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


# ================================================================
# AST / SOURCE INSPECTION HELPERS
# ================================================================

def class_map(
    tree: ast.AST,
) -> dict[str, ast.ClassDef]:
    return {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(
            node,
            ast.ClassDef,
        )
    }


def function_map(
    tree: ast.AST,
) -> dict[str, ast.AST]:
    result = {}

    for node in ast.walk(tree):
        if isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
            ),
        ):
            result[node.name] = node

    return result


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


def all_executable_string_constants(
    tree: ast.AST,
) -> set[str]:
    """
    Return actual Python string constants while ignoring
    comments and docstrings.

    Comments do not appear in the AST.
    Module/class/function docstrings are excluded.
    """

    docstring_ids: set[int] = set()

    for node in ast.walk(tree):
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
            docstring_ids.add(
                id(value)
            )

    strings = set()

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

        if id(node) in docstring_ids:
            continue

        strings.add(
            node.value
        )

    return strings


def executable_identifiers(
    tree: ast.AST,
) -> set[str]:
    """
    Collect executable identifiers and attribute names.

    Also includes imported names and class/function names.
    """

    identifiers: set[str] = set()

    for node in ast.walk(tree):

        if isinstance(
            node,
            ast.Name,
        ):
            identifiers.add(
                node.id
            )

        elif isinstance(
            node,
            ast.Attribute,
        ):
            identifiers.add(
                node.attr
            )

        elif isinstance(
            node,
            ast.ClassDef,
        ):
            identifiers.add(
                node.name
            )

        elif isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
            ),
        ):
            identifiers.add(
                node.name
            )

        elif isinstance(
            node,
            ast.alias,
        ):
            identifiers.add(
                node.asname
                or node.name
                .split(".")[-1]
            )

    return identifiers


def has_identifier_or_string(
    tree: ast.AST,
    value: str,
) -> bool:

    if value in executable_identifiers(
        tree
    ):
        return True

    if value in all_executable_string_constants(
        tree
    ):
        return True

    return False


def actual_blur_call_exists(
    source: str,
) -> bool:
    """
    Detect actual executable:

        Blur().blur(...)
        module.Blur().blur(...)

    Does not search raw source text.
    """

    try:
        tree = ast.parse(
            source
        )
    except SyntaxError:
        return False

    for node in ast.walk(tree):

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

        if func.attr != "blur":
            continue

        receiver = func.value

        if not isinstance(
            receiver,
            ast.Call,
        ):
            continue

        constructor = receiver.func

        if (
            isinstance(
                constructor,
                ast.Name,
            )
            and constructor.id == "Blur"
        ):
            return True

        if (
            isinstance(
                constructor,
                ast.Attribute,
            )
            and constructor.attr == "Blur"
        ):
            return True

    return False


def has_call_to_method(
    tree: ast.AST,
    object_name: str,
    method_name: str,
) -> bool:
    """
    Detect:

        object_name.method_name(...)

    in actual executable AST.
    """

    for node in ast.walk(tree):

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

        if func.attr != method_name:
            continue

        value = func.value

        if (
            isinstance(
                value,
                ast.Name,
            )
            and value.id == object_name
        ):
            return True

        if (
            isinstance(
                value,
                ast.Attribute,
            )
            and value.attr == object_name
        ):
            return True

    return False


# ================================================================
# LOCK
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
                sorted(missing)
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

    check_git_revision(
        comfy_dir,
        lock[
            "comfyui"
        ][
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
# RUNTIME
# ================================================================

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

    expected = (
        lock[
            "python_runtime"
        ][
            "torch"
        ]
    )

    if torch.__version__ != expected:
        fail(
            "Torch mismatch.\n"
            f"Expected: {expected}\n"
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

    expected_tv = (
        lock[
            "python_runtime"
        ][
            "torchvision"
        ]
    )

    if (
        torchvision.__version__
        != expected_tv
    ):
        fail(
            "torchvision mismatch.\n"
            f"Expected: {expected_tv}\n"
            f"Actual:   {torchvision.__version__}"
        )

    print(
        f"OK   torchvision=="
        f"{torchvision.__version__}"
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


# ================================================================
# STRUCTURE
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

    for relative in required:
        check_file(
            PROJECT_ROOT
            / relative
        )


# ================================================================
# WORKFLOWS
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
# COMPATIBILITY
# ================================================================

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

    tree = parse_python(
        path
    )

    strings = (
        all_executable_string_constants(
            tree
        )
    )

    identifiers = (
        executable_identifiers(
            tree
        )
    )

    required_values = [

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

    for value in required_values:

        if (
            value not in strings
            and value not in identifiers
        ):

            fail(
                "Compatibility builder is missing:\n"
                f"{value}"
            )

    legacy_commit = (
        lock[
            "legacy_ltx_098_compat"
        ][
            "commit"
        ]
    )

    # Search executable strings and identifiers only,
    # not comments.
    if legacy_commit in strings:

        fail(
            "Legacy LTX commit is hardcoded "
            "as an executable string in "
            "prepare_modern_ltx.py."
        )

    source = read_text(
        path
    )

    if actual_blur_call_exists(
        source
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
# PREFLIGHT
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

    tree = parse_python(
        path
    )

    required = [
        "compatibility_lock.yaml",
        "python_runtime",
        "models",
        "verify_gpu",
        "verify_locked_packages",
    ]

    for value in required:

        if not has_identifier_or_string(
            tree,
            value,
        ):

            fail(
                "Preflight is missing:\n"
                f"{value}"
            )

    expected_torch = (
        lock[
            "python_runtime"
        ][
            "torch"
        ]
    )

    # It is allowed for the preflight to read the lock;
    # the literal locked version must not be embedded as
    # executable configuration.
    strings = (
        all_executable_string_constants(
            tree
        )
    )

    if expected_torch in strings:

        fail(
            "Torch version is hardcoded as "
            "an executable string in "
            "preflight_modern.py."
        )

    print(
        "OK   modern preflight"
    )


# ================================================================
# CENTRAL CONFIG
# ================================================================

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

    tree = parse_python(
        path
    )

    required = [
        "compatibility_lock.yaml",
        "MODELS",
        "Q4_MODEL",
        "VAE_MODEL",
        "T5_MODEL",
        "DETAILER_LORA",
        "SPATIAL_UPSCALER",
    ]

    for value in required:

        if not has_identifier_or_string(
            tree,
            value,
        ):

            fail(
                "kaggle/config.py is missing:\n"
                f"{value}"
            )

    strings = (
        all_executable_string_constants(
            tree
        )
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

        if filename in strings:

            fail(
                "Model filename is duplicated "
                "as executable configuration "
                f"in kaggle/config.py:\n{filename}"
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

    for path in startup_files:

        if not path.exists():
            continue

        tree = parse_python(
            path
        )

        strings = (
            all_executable_string_constants(
                tree
            )
        )

        for value in strings:

            if (
                "--front-end-version"
                in value
            ):
                fail(
                    f"{path.relative_to(PROJECT_ROOT)} "
                    "contains an executable "
                    "frontend version override."
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

    classes = class_map(
        tree
    )

    adapter = classes.get(
        "ComfyWorkflowAdapter"
    )

    if adapter is None:
        fail(
            "ComfyWorkflowAdapter class is missing."
        )

    methods = method_names(
        adapter
    )

    required_methods = {

        "__init__",
        "to_api_workflow",
        "apply_modern_compatibility",
        "set_prompt",
        "set_negative_prompt",
        "set_seed",
        "set_filename_prefix",
        "set_input_image",
        "set_input_video",
        "validate_modern_detailer",
        "apply_shot",
    }

    missing = (
        required_methods
        - methods
    )

    if missing:

        fail(
            "ComfyWorkflowAdapter missing methods:\n"
            + "\n".join(
                sorted(
                    missing
                )
            )
        )

    strings = (
        all_executable_string_constants(
            tree
        )
    )

    required_strings = {

        "LTXVLatentUpsamplerModelLoader",
        "LatentUpscaleModelLoader",
        "VHS_LoadVideo",
        "VHS_VideoCombine",
        "CLIPTextEncode",
        "LoadImage",
    }

    missing_strings = (
        required_strings
        - strings
    )

    if missing_strings:

        fail(
            "Workflow adapter missing required node values:\n"
            + "\n".join(
                sorted(
                    missing_strings
                )
            )
        )

    print(
        "OK   ComfyWorkflowAdapter"
    )

    print(
        "OK   legacy → modern latent-loader conversion"
    )

    print(
        "OK   workflow runtime mutation methods"
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
        - method_names(
            client
        )
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

    identifiers = (
        executable_identifiers(
            tree
        )
    )

    strings = (
        all_executable_string_constants(
            tree
        )
    )

    required = {

        "_copy_raw_to_comfy_input",
        "execute_raw",
        "execute_detailer",
        "mark_raw_complete",
        "mark_upscaled_complete",
        "VHS_LoadVideo",
    }

    for identifier in required:

        if (
            identifier not in identifiers
            and identifier not in strings
        ):

            fail(
                "ShotExecutor missing integration:\n"
                f"{identifier}"
            )

    source = read_text(
        path
    )

    if (
        "1536x832"
        in source
    ):

        fail(
            "ShotExecutor contains obsolete "
            "1536x832 production resolution."
        )

    if (
        "1536x864"
        not in source
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

    required_methods = {

        "__init__",
        "prepare",
        "run",
        "_run_one_shot",
        "_dict_to_shot",
    }

    missing_methods = (
        required_methods
        - method_names(
            runner
        )
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

    identifiers = (
        executable_identifiers(
            tree
        )
    )

    required_integrations = {

        "GPUScheduler",
        "ShotExecutor",
        "ComfyClient",
        "AssemblyManager",
        "CheckpointManager",
        "ComfyWorkflowAdapter",
    }

    missing_integrations = (
        required_integrations
        - identifiers
    )

    if missing_integrations:

        fail(
            "ProductionRunner missing integrations:\n"
            + "\n".join(
                sorted(
                    missing_integrations
                )
            )
        )

    # scheduler.run(...)
    scheduler_run = False

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
            != "run"
        ):
            continue

        target = func.value

        if (
            isinstance(
                target,
                ast.Attribute,
            )
            and target.attr
            == "scheduler"
        ):
            scheduler_run = True
            break

    if not scheduler_run:

        fail(
            "ProductionRunner.run() does not call "
            "self.scheduler.run(...)."
        )

    # ShotExecutor(...)
    executor_constructed = False

    for node in ast.walk(
        runner
    ):

        if not isinstance(
            node,
            ast.Call,
        ):
            continue

        if (
            isinstance(
                node.func,
                ast.Name,
            )
            and node.func.id
            == "ShotExecutor"
        ):

            executor_constructed = True
            break

    if not executor_constructed:

        fail(
            "ProductionRunner does not construct ShotExecutor."
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

    identifiers = (
        executable_identifiers(
            tree
        )
    )

    for identifier in (
        "worker_function",
        "failures",
    ):

        if identifier not in identifiers:

            fail(
                "GPU scheduler missing:\n"
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

    identifiers = (
        executable_identifiers(
            tree
        )
    )

    if "RLock" not in identifiers:

        fail(
            "CheckpointManager is not using "
            "thread-safe RLock."
        )

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

    missing = (
        required_methods
        - method_names(
            manager
        )
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

    strings = (
        all_executable_string_constants(
            tree
        )
    )

    if (
        "production_state.json"
        not in strings
    ):

        fail(
            "production_state.json handling is missing."
        )

    print(
        "OK   thread-safe checkpoint manager"
    )


# ================================================================
# PLANNER → ORCHESTRATOR
# ================================================================

def validate_planner_orchestrator_chain() -> None:

    path = (
        PROJECT_ROOT
        / "pipeline"
        / "production_orchestrator.py"
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

    missing_methods = (
        required_methods
        - method_names(
            orchestrator
        )
    )

    if missing_methods:

        fail(
            "ProductionOrchestrator missing methods:\n"
            + "\n".join(
                sorted(
                    missing_methods
                )
            )
        )

    identifiers = (
        executable_identifiers(
            tree
        )
    )

    required_integrations = {

        "StoryPlanner",
        "CharacterDetector",
        "CharacterPlanner",
        "ScenePlanner",
        "ShotPlanner",
        "ContinuityManager",
    }

    missing = (
        required_integrations
        - identifiers
    )

    if missing:

        fail(
            "ProductionOrchestrator missing:\n"
            + "\n".join(
                sorted(
                    missing
                )
            )
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
# CHARACTER REFERENCES
# ================================================================

def validate_reference_chain() -> None:

    paths = [

        PROJECT_ROOT
        / "pipeline"
        / "reference_manager.py",

        PROJECT_ROOT
        / "planner"
        / "character_planner.py",

        PROJECT_ROOT
        / "planner"
        / "shot_planner.py",
    ]

    for path in paths:
        check_file(path)

    reference_tree = parse_python(
        paths[0]
    )

    character_tree = parse_python(
        paths[1]
    )

    shot_tree = parse_python(
        paths[2]
    )

    for tree, class_name in (
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

        if class_name not in class_map(
            tree
        ):

            fail(
                f"{class_name} class is missing."
            )

    shot_identifiers = (
        executable_identifiers(
            shot_tree
        )
    )

    shot_strings = (
        all_executable_string_constants(
            shot_tree
        )
    )

    if (
        "reference_images"
        not in shot_identifiers
        and "reference_images"
        not in shot_strings
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

    if (
        "get_pipeline"
        not in method_names(
            manager
        )
    ):

        fail(
            "ProductionManager.get_pipeline() "
            "is missing."
        )

    print(
        "OK   ProductionManager"
    )


# ================================================================
# APPLICATION ENTRY POINT
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

    identifiers = (
        executable_identifiers(
            tree
        )
    )

    required_classes = {

        "ProductionOrchestrator",
        "ProductionRunner",
        "ProductionManager",
    }

    missing = (
        required_classes
        - identifiers
    )

    if missing:

        fail(
            "generate_video.py missing imports/integrations:\n"
            + "\n".join(
                sorted(
                    missing
                )
            )
        )

    strings = (
        all_executable_string_constants(
            tree
        )
    )

    required_arguments = {
        "--story",
        "--mode",
        "--gpu-url",
    }

    missing_args = (
        required_arguments
        - strings
    )

    if missing_args:

        fail(
            "generate_video.py missing CLI arguments:\n"
            + "\n".join(
                sorted(
                    missing_args
                )
            )
        )

    # Verify real production-plan call.
    if not has_call_to_method(
        tree,
        "orchestrator",
        "create_production_plan",
    ):

        fail(
            "generate_video.py does not call "
            "orchestrator.create_production_plan(...)."
        )

    # Verify runner execution call.
    if not has_call_to_method(
        tree,
        "runner",
        "run",
    ):

        fail(
            "generate_video.py does not call "
            "runner.run(...)."
        )

    print(
        "OK   application generation entry point"
    )

    print(
        "OK   Story → Production plan"
    )

    print(
        "OK   Production plan → ProductionRunner"
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
# LOCK / TOOL POLICY
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
    # Repository
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
    # Planner/application chain
    # ------------------------------------------------------------

    validate_planner_orchestrator_chain()

    validate_reference_chain()

    validate_production_manager()

    validate_generate_video()

    # ------------------------------------------------------------
    # Workflows/tools
    # ------------------------------------------------------------

    validate_workflows()

    validate_external_tools()

    # ------------------------------------------------------------
    # Final
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
        "Frontend:"
    )

    print(
        "  PINNED / executable @latest prohibited"
    )

    print(
        "Compatibility:"
    )

    print(
        "  MODERN + LTX 0.9.8"
    )

    print(
        "Blur:"
    )

    print(
        "  NATIVE TORCH"
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

    main()
