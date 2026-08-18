#!/usr/bin/env python3
"""
LTX-13B PROJECT VALIDATOR

Validation layers:

1. Repository structure
2. Python AST/syntax
3. Compatibility lock schema
4. Single-source model-path authority
5. Workflow JSON structure
6. Workflow link integrity
7. Real graph -> API conversion through ComfyWorkflowAdapter
8. Legacy -> modern latent-upscaler conversion
9. Compatibility-builder contract
10. Production application wiring
11. Kaggle launcher/bootstrap wiring
12. CPU preflight presence and contract
13. Materialized ComfyUI runtime, when requested
14. Optional CUDA verification, when requested

Usage:

    python scripts/validate_project.py

For a fully materialized Kaggle runtime:

    python scripts/validate_project.py --require-runtime

For runtime + CUDA:

    python scripts/validate_project.py --require-runtime --require-cuda

IMPORTANT:

- kaggle/compatibility_lock.yaml is the SINGLE SOURCE OF TRUTH.
- kaggle/model_paths.yaml must not exist.
- kaggle/runtime_requirements.lock must not exist.
"""

from __future__ import annotations

import argparse
import ast
import importlib.metadata
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


# ======================================================================
# PATHS
# ======================================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

LOCK_FILE = (
    PROJECT_ROOT
    / "kaggle"
    / "compatibility_lock.yaml"
)

COMFYUI_DIR = (
    PROJECT_ROOT
    / "ComfyUI"
)

BASE_WORKFLOW = (
    PROJECT_ROOT
    / "workflows"
    / "baseline"
    / "ltxv-13b-dist-i2v-base.json"
)

DETAILER_WORKFLOW = (
    PROJECT_ROOT
    / "workflows"
    / "detailer"
    / "ltxv-13b-098-ic-lora-upscale.json"
)


# ======================================================================
# CANONICAL CONTRACTS
# ======================================================================

LEGACY_LATENT_LOADER = (
    "LTXVLatentUpsamplerModelLoader"
)

MODERN_LATENT_LOADER = (
    "LatentUpscaleModelLoader"
)

CPU_PREFLIGHT = (
    PROJECT_ROOT
    / "scripts"
    / "cpu_preflight.py"
)


REQUIRED_FILES = (
    # Planner
    "planner/config.py",
    "planner/qwen_loader.py",
    "planner/story_planner.py",
    "planner/character_detector.py",
    "planner/character_planner.py",
    "planner/scene_planner.py",
    "planner/shot_planner.py",

    # Pipeline
    "pipeline/__init__.py",
    "pipeline/continuity_manager.py",
    "pipeline/modes.py",
    "pipeline/production_manager.py",
    "pipeline/production_orchestrator.py",
    "pipeline/reference_manager.py",

    # Execution
    "execution/__init__.py",
    "execution/checkpoint_manager.py",
    "execution/comfy_client.py",
    "execution/comfy_workflow_adapter.py",
    "execution/shot_executor.py",
    "execution/assembly_manager.py",
    "execution/production_runner.py",

    # Scheduler
    "scheduler/__init__.py",
    "scheduler/gpu_scheduler.py",
    "scheduler/shot_queue.py",

    # Schemas
    "schemas/__init__.py",
    "schemas/character.py",
    "schemas/scene.py",
    "schemas/shot.py",
    "schemas/parser.py",

    # Kaggle
    "kaggle/compatibility_lock.yaml",
    "kaggle/bootstrap.py",
    "kaggle/config.py",
    "kaggle/launch.py",
    "kaggle/preflight_modern.py",
    "kaggle/start_comfyui.py",
    "kaggle/start_comfyui_tunnel.py",

    # Compatibility
    "compatibility/prepare_modern_ltx.py",

    # Scripts
    "scripts/cpu_preflight.py",
    "scripts/generate_video.py",
    "scripts/validate_project.py",

    # Workflows
    "workflows/baseline/ltxv-13b-dist-i2v-base.json",
    "workflows/detailer/ltxv-13b-098-ic-lora-upscale.json",
)


OBSOLETE_FILES = (
    PROJECT_ROOT
    / "kaggle"
    / "model_paths.yaml",

    PROJECT_ROOT
    / "kaggle"
    / "runtime_requirements.lock",
)


# ======================================================================
# GENERAL HELPERS
# ======================================================================

def fail(message: str) -> None:
    raise RuntimeError(message)


def require(
    condition: bool,
    message: str,
) -> None:
    if not condition:
        fail(message)


def read_text(
    path: Path,
) -> str:

    if not path.is_file():
        fail(
            "Required file is missing:\n"
            f"{path}"
        )

    try:
        return path.read_text(
            encoding="utf-8"
        )

    except OSError as error:
        raise RuntimeError(
            "Unable to read file:\n"
            f"{path}\n"
            f"{error}"
        ) from error


def parse_python(
    path: Path,
) -> ast.Module:

    source = read_text(
        path
    )

    try:
        return ast.parse(
            source,
            filename=str(path),
        )

    except SyntaxError as error:
        fail(
            "Python syntax error:\n"
            f"{path}\n"
            f"Line {error.lineno}: "
            f"{error.msg}"
        )


def parse_json(
    path: Path,
) -> dict[str, Any]:

    try:
        data = json.loads(
            read_text(path)
        )

    except json.JSONDecodeError as error:
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
            "JSON root must be an object:\n"
            f"{path}"
        )

    return data


def classes(
    module: ast.AST,
) -> dict[str, ast.ClassDef]:

    return {
        node.name: node
        for node in ast.walk(module)
        if isinstance(
            node,
            ast.ClassDef,
        )
    }


def methods(
    node: ast.ClassDef,
) -> set[str]:

    return {
        item.name
        for item in node.body
        if isinstance(
            item,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
            ),
        )
    }


def names(
    module: ast.AST,
) -> set[str]:

    result: set[str] = set()

    for node in ast.walk(
        module
    ):

        if isinstance(
            node,
            ast.Name,
        ):
            result.add(
                node.id
            )

        elif isinstance(
            node,
            ast.Attribute,
        ):
            result.add(
                node.attr
            )

        elif isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.ClassDef,
            ),
        ):
            result.add(
                node.name
            )

        elif isinstance(
            node,
            ast.alias,
        ):
            result.add(
                node.asname
                or node.name.rsplit(
                    ".",
                    1,
                )[-1]
            )

    return result


def string_constants(
    module: ast.AST,
) -> set[str]:

    return {
        node.value
        for node in ast.walk(
            module
        )
        if isinstance(
            node,
            ast.Constant,
        )
        and isinstance(
            node.value,
            str,
        )
    }


def require_subset(
    actual: set[str],
    required: set[str],
    label: str,
) -> None:

    missing = (
        required
        - actual
    )

    if missing:
        fail(
            f"{label} is missing:\n"
            + "\n".join(
                f"  - {item}"
                for item
                in sorted(missing)
            )
        )


def has_call(
    module: ast.AST,
    receiver: str,
    method: str,
) -> bool:

    for node in ast.walk(
        module
    ):

        if not isinstance(
            node,
            ast.Call,
        ):
            continue

        if not isinstance(
            node.func,
            ast.Attribute,
        ):
            continue

        if node.func.attr != method:
            continue

        value = node.func.value

        if (
            isinstance(
                value,
                ast.Name,
            )
            and value.id == receiver
        ):
            return True

        if (
            isinstance(
                value,
                ast.Attribute,
            )
            and value.attr == receiver
        ):
            return True

    return False


def find_function(
    module: ast.AST,
    function_name: str,
) -> ast.FunctionDef | ast.AsyncFunctionDef | None:

    for node in ast.walk(
        module
    ):
        if (
            isinstance(
                node,
                (
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                ),
            )
            and node.name == function_name
        ):
            return node

    return None


def extract_string_assignment(
    module: ast.AST,
    function_name: str,
    variable_name: str,
) -> str:

    function = find_function(
        module,
        function_name,
    )

    require(
        function is not None,
        f"Function missing: "
        f"{function_name}()",
    )

    for node in ast.walk(
        function
    ):

        if not isinstance(
            node,
            ast.Assign,
        ):
            continue

        target_matches = any(
            isinstance(
                target,
                ast.Name,
            )
            and target.id == variable_name
            for target
            in node.targets
        )

        if not target_matches:
            continue

        value = node.value

        # Support:
        #
        # replacement = """...""".lstrip()
        #
        if (
            isinstance(
                value,
                ast.Call,
            )
            and isinstance(
                value.func,
                ast.Attribute,
            )
            and value.func.attr
            == "lstrip"
            and isinstance(
                value.func.value,
                ast.Constant,
            )
            and isinstance(
                value.func.value.value,
                str,
            )
        ):
            return value.func.value

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
            return value.value

    fail(
        f"{function_name}() does not assign "
        f"a string to {variable_name!r}."
    )


def all_project_python_files() -> list[Path]:

    files: list[Path] = []

    for path in PROJECT_ROOT.rglob(
        "*.py"
    ):

        relative = path.relative_to(
            PROJECT_ROOT
        )

        if ".git" in relative.parts:
            continue

        if "ComfyUI" in relative.parts:
            continue

        if ".runtime_ltx098" in relative.parts:
            continue

        if "__pycache__" in relative.parts:
            continue

        files.append(path)

    return sorted(
        files
    )


# ======================================================================
# 1. REPOSITORY STRUCTURE
# ======================================================================

def validate_files() -> None:

    for relative in (
        REQUIRED_FILES
    ):

        path = (
            PROJECT_ROOT
            / relative
        )

        require(
            path.is_file(),
            "Required project file missing:\n"
            f"{path}",
        )

    for obsolete in (
        OBSOLETE_FILES
    ):

        require(
            not obsolete.exists(),
            "Obsolete duplicated configuration "
            "still exists:\n"
            f"{obsolete}\n\n"
            "compatibility_lock.yaml must remain "
            "the single source of truth.",
        )

    print(
        "OK   repository file structure"
    )


# ======================================================================
# 2. PYTHON SYNTAX
# ======================================================================

def validate_python() -> None:

    files = (
        all_project_python_files()
    )

    for path in files:
        parse_python(
            path
        )

    print(
        "OK   Python AST syntax:"
        f" {len(files)} files"
    )


# ======================================================================
# 3. COMPATIBILITY LOCK
# ======================================================================

def load_lock() -> dict[str, Any]:

    try:
        import yaml

    except ImportError as error:
        fail(
            "PyYAML is required to validate "
            "compatibility_lock.yaml:\n"
            f"{error}"
        )

    try:
        data = yaml.safe_load(
            read_text(
                LOCK_FILE
            )
        )

    except Exception as error:
        fail(
            "Invalid compatibility_lock.yaml:\n"
            f"{error}"
        )

    if not isinstance(
        data,
        dict,
    ):
        fail(
            "compatibility_lock.yaml "
            "must contain a mapping."
        )

    return data


def validate_lock(
    lock: dict[str, Any],
) -> None:

    required_top_level = {
        "comfyui",
        "python_runtime",
        "custom_nodes",
        "legacy_ltx_098_compat",
        "detailer_compat",
        "models",
        "validation",
    }

    require_subset(
        set(lock),
        required_top_level,
        "compatibility_lock.yaml",
    )

    comfy = lock[
        "comfyui"
    ]

    runtime = lock[
        "python_runtime"
    ]

    custom_nodes = lock[
        "custom_nodes"
    ]

    models = lock[
        "models"
    ]

    legacy = lock[
        "legacy_ltx_098_compat"
    ]

    detailer = lock[
        "detailer_compat"
    ]

    require(
        isinstance(
            comfy,
            dict,
        ),
        "comfyui lock section is invalid.",
    )

    require(
        isinstance(
            runtime,
            dict,
        ),
        "python_runtime lock section is invalid.",
    )

    require(
        isinstance(
            custom_nodes,
            dict,
        ),
        "custom_nodes lock section is invalid.",
    )

    require(
        isinstance(
            models,
            dict,
        ),
        "models lock section is invalid.",
    )

    require(
        comfy.get(
            "repository"
        ).endswith(
            "ComfyUI.git"
        ),
        "Invalid ComfyUI repository "
        "in compatibility lock.",
    )

    require(
        re.fullmatch(
            r"[0-9a-f]{40}",
            str(
                comfy.get(
                    "commit",
                    "",
                )
            ),
        )
        is not None,
        "ComfyUI commit must be "
        "a 40-character SHA.",
    )

    for package_key in (
        "frontend",
        "workflow_templates",
        "embedded_docs",
        "comfy_kitchen",
        "comfy_aimdo",
    ):

        package = comfy.get(
            package_key
        )

        require(
            isinstance(
                package,
                dict,
            ),
            f"Missing ComfyUI package lock: "
            f"{package_key}",
        )

        require(
            bool(
                package.get(
                    "package"
                )
            ),
            f"Missing package name: "
            f"{package_key}",
        )

        require(
            bool(
                package.get(
                    "version"
                )
            ),
            f"Missing package version: "
            f"{package_key}",
        )

    for package in (
        "torch",
        "torchvision",
        "torchsde",
        "spandrel",
        "av",
        "gguf",
    ):

        require(
            bool(
                runtime.get(
                    package
                )
            ),
            f"Missing locked runtime package: "
            f"{package}",
        )

    for name, spec in (
        custom_nodes.items()
    ):

        require(
            isinstance(
                spec,
                dict,
            ),
            f"Invalid custom node lock: "
            f"{name}",
        )

        require(
            bool(
                spec.get(
                    "repository"
                )
            ),
            f"Missing repository for "
            f"custom node: {name}",
        )

        require(
            re.fullmatch(
                r"[0-9a-f]{40}",
                str(
                    spec.get(
                        "commit",
                        "",
                    )
                ),
            )
            is not None,
            f"Invalid SHA for "
            f"custom node: {name}",
        )

    require(
        re.fullmatch(
            r"[0-9a-f]{40}",
            str(
                legacy.get(
                    "commit",
                    "",
                )
            ),
        )
        is not None,
        "Invalid legacy LTX compatibility commit.",
    )

    require(
        legacy.get(
            "runtime_package"
        )
        == "LTX098ModernCompat",
        "Unexpected compatibility "
        "runtime package name.",
    )

    required_models = {
        "ltx_q4",
        "t5_q4",
        "vae",
        "ic_lora",
        "spatial_upscaler",
    }

    require_subset(
        set(models),
        required_models,
        "Locked model definitions",
    )

    for name, spec in (
        models.items()
    ):

        require(
            isinstance(
                spec,
                dict,
            ),
            f"Invalid model lock: {name}",
        )

        dataset = spec.get(
            "dataset"
        )

        filename = spec.get(
            "filename"
        )

        target = spec.get(
            "target"
        )

        require(
            isinstance(
                dataset,
                str,
            )
            and dataset.startswith(
                "/kaggle/input/"
            ),
            f"Invalid dataset path "
            f"for {name}: {dataset}",
        )

        require(
            isinstance(
                filename,
                str,
            )
            and bool(
                filename.strip()
            ),
            f"Missing model filename "
            f"for {name}",
        )

        require(
            isinstance(
                target,
                str,
            )
            and target.startswith(
                "models/"
            ),
            f"Invalid model target "
            f"for {name}: {target}",
        )

    require(
        detailer.get(
            "legacy_loader"
        )
        == LEGACY_LATENT_LOADER,
        "Lock/detailer legacy loader "
        "is inconsistent.",
    )

    require(
        detailer.get(
            "modern_loader"
        )
        == MODERN_LATENT_LOADER,
        "Lock/detailer modern loader "
        "is inconsistent.",
    )

    print(
        "OK   compatibility lock schema"
    )


# ======================================================================
# 4. WORKFLOW GRAPH VALIDATION
# ======================================================================

def validate_workflow_graph(
    path: Path,
    required_nodes: set[str],
) -> dict[str, Any]:

    workflow = parse_json(
        path
    )

    nodes = workflow.get(
        "nodes"
    )

    links = workflow.get(
        "links"
    )

    require(
        isinstance(
            nodes,
            list,
        )
        and nodes,
        "Workflow contains no nodes:\n"
        f"{path}",
    )

    require(
        isinstance(
            links,
            list,
        ),
        "Workflow links are not a list:\n"
        f"{path}",
    )

    node_map: dict[
        str,
        dict[str, Any],
    ] = {}

    for node in nodes:

        require(
            isinstance(
                node,
                dict,
            ),
            "Workflow contains invalid node:\n"
            f"{path}",
        )

        node_id = node.get(
            "id"
        )

        node_type = node.get(
            "type"
        )

        require(
            node_id is not None,
            "Workflow node has no ID:\n"
            f"{path}",
        )

        key = str(
            node_id
        )

        require(
            key not in node_map,
            f"Duplicate workflow node ID "
            f"{key}:\n{path}",
        )

        require(
            isinstance(
                node_type,
                str,
            )
            and node_type.strip(),
            f"Workflow node "
            f"{key} has no type:\n{path}",
        )

        node_map[
            key
        ] = node

    node_types = {
        str(
            node.get(
                "type"
            )
        )
        for node
        in nodes
    }

    require_subset(
        node_types,
        required_nodes,
        str(path),
    )

    link_map: dict[
        str,
        list[Any],
    ] = {}

    for link in links:

        require(
            isinstance(
                link,
                list,
            )
            and len(link) >= 5,
            "Malformed workflow link:\n"
            f"{path}\n"
            f"{link}",
        )

        link_id = str(
            link[0]
        )

        origin_id = str(
            link[1]
        )

        target_id = str(
            link[3]
        )

        require(
            link_id not in link_map,
            f"Duplicate workflow link "
            f"{link_id}:\n{path}",
        )

        require(
            origin_id in node_map,
            f"Link {link_id} references "
            f"missing origin node "
            f"{origin_id}:\n{path}",
        )

        require(
            target_id in node_map,
            f"Link {link_id} references "
            f"missing target node "
            f"{target_id}:\n{path}",
        )

        link_map[
            link_id
        ] = link

    # Every declared input link must resolve.
    for node in nodes:

        inputs = node.get(
            "inputs",
            [],
        )

        if not isinstance(
            inputs,
            list,
        ):
            continue

        for input_def in inputs:

            if not isinstance(
                input_def,
                dict,
            ):
                continue

            link_id = input_def.get(
                "link"
            )

            if link_id is None:
                continue

            require(
                str(link_id)
                in link_map,
                f"Node "
                f"{node['id']} input "
                f"{input_def.get('name')} "
                f"references missing "
                f"link {link_id}:\n"
                f"{path}",
            )

    require(
        "VHS_VideoCombine"
        in node_types,
        "Workflow has no "
        "VHS_VideoCombine output:\n"
        f"{path}",
    )

    print(
        "OK   workflow graph:"
        f" {path.relative_to(PROJECT_ROOT)}"
        f" ({len(nodes)} nodes, "
        f"{len(links)} links)"
    )

    return workflow


# ======================================================================
# 5. REAL WORKFLOW -> API CONVERSION
# ======================================================================

def validate_workflow_conversion(
    path: Path,
    workflow: dict[str, Any],
    detailer: bool,
) -> None:

    try:

        from execution.comfy_workflow_adapter import (
            ComfyWorkflowAdapter,
        )

        adapter = (
            ComfyWorkflowAdapter(
                path
            )
        )

        api = (
            adapter.to_api_workflow()
        )

    except Exception as error:
        fail(
            "Workflow conversion failed:\n"
            f"{path}\n"
            f"{type(error).__name__}: "
            f"{error}"
        )

    require(
        isinstance(
            api,
            dict,
        )
        and api,
        "Empty API workflow:\n"
        f"{path}",
    )

    class_types = {
        node.get(
            "class_type"
        )
        for node
        in api.values()
        if isinstance(
            node,
            dict,
        )
    }

    require(
        "VHS_VideoCombine"
        in class_types,
        "Converted workflow has no "
        "VHS_VideoCombine:\n"
        f"{path}",
    )

    if detailer:

        require(
            LEGACY_LATENT_LOADER
            not in class_types,
            "Legacy latent-upscaler loader "
            "survived API conversion.",
        )

        require(
            MODERN_LATENT_LOADER
            in class_types,
            "Modern LatentUpscaleModelLoader "
            "is missing after API conversion.",
        )

        try:

            adapter.validate_modern_detailer(
                api
            )

        except Exception as error:

            fail(
                "Modern detailer validation failed:\n"
                f"{type(error).__name__}: "
                f"{error}"
            )

    # Validate API node references.
    for node_id, node in (
        api.items()
    ):

        require(
            isinstance(
                node,
                dict,
            ),
            f"Converted node "
            f"{node_id} is invalid.",
        )

        inputs = node.get(
            "inputs",
            {},
        )

        require(
            isinstance(
                inputs,
                dict,
            ),
            f"Converted node "
            f"{node_id} has invalid inputs.",
        )

        for input_name, value in (
            inputs.items()
        ):

            if (
                isinstance(
                    value,
                    list,
                )
                and len(value) == 2
                and isinstance(
                    value[0],
                    str,
                )
            ):

                require(
                    value[0] in api,
                    f"Converted input "
                    f"{node_id}.{input_name} "
                    f"references missing "
                    f"API node {value[0]}.",
                )

    print(
        "OK   workflow API conversion:"
        f" {path.relative_to(PROJECT_ROOT)}"
    )


# ======================================================================
# 6. COMPATIBILITY BUILDER
# ======================================================================

def validate_compatibility_builder() -> None:

    path = (
        PROJECT_ROOT
        / "compatibility"
        / "prepare_modern_ltx.py"
    )

    module = parse_python(
        path
    )

    module_names = names(
        module
    )

    module_constants = string_constants(
        module
    )

    require_subset(
        module_names
        | module_constants,
        {
            "load_lock",
            "get_legacy_commit",
            "patch_blur",
            "write_curated_init",
            "build_compat_package",
            "LTX098ModernCompat",
            "compatibility_lock.yaml",
        },
        "Compatibility builder",
    )

    # --------------------------------------------------------------
    # Generated legacy initializer
    # --------------------------------------------------------------

    init_code = extract_string_assignment(
        module,
        "write_curated_init",
        "init_code",
    )

    try:
        initializer_ast = ast.parse(
            init_code,
            filename="generated_compat_init",
        )

    except SyntaxError as error:
        fail(
            "Generated compatibility initializer "
            "is syntactically invalid:\n"
            f"{error}"
        )

    initializer_constants = (
        string_constants(
            initializer_ast
        )
    )

    required_legacy_nodes = {
        "LTXVBaseSampler",
        "LTXVLoopingSampler",
        "LTXVTiledSampler",
        "LTXVTiledVAEDecode",
        "LTXVLatentUpsampler",
        "LTXVLatentUpsamplerModelLoader",
        "LTXVFilmGrain",
        "STGGuiderAdvanced",
        "Set VAE Decoder Noise",
    }

    require_subset(
        initializer_constants,
        required_legacy_nodes,
        "Generated compatibility initializer",
    )

    # --------------------------------------------------------------
    # Generated blur replacement
    # --------------------------------------------------------------

    replacement = extract_string_assignment(
        module,
        "patch_blur",
        "replacement",
    )

    try:

        replacement_ast = ast.parse(
            replacement,
            filename="generated_blur_replacement",
        )

    except SyntaxError as error:
        fail(
            "Generated blur replacement "
            "is syntactically invalid:\n"
            f"{error}"
        )

    blur_function = next(
        (
            node
            for node
            in ast.walk(
                replacement_ast
            )
            if isinstance(
                node,
                (
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                ),
            )
            and node.name
            == "blur_internal"
        ),
        None,
    )

    require(
        blur_function is not None,
        "Generated blur replacement "
        "does not define blur_internal().",
    )

    parameter_names = [
        arg.arg
        for arg
        in blur_function.args.args
    ]

    require(
        parameter_names
        == [
            "image",
            "blur_radius",
        ],
        "Generated blur_internal() "
        "signature is wrong.\n"
        f"Actual: {parameter_names}\n"
        "Expected: "
        "['image', 'blur_radius']",
    )

    conv2d_found = any(
        isinstance(
            node,
            ast.Call,
        )
        and isinstance(
            node.func,
            ast.Attribute,
        )
        and node.func.attr
        == "conv2d"
        for node
        in ast.walk(
            blur_function
        )
    )

    require(
        conv2d_found,
        "Generated blur_internal() "
        "does not use conv2d().",
    )

    print(
        "OK   LTX 0.9.8 compatibility builder"
    )

    print(
        "OK   generated legacy initializer"
    )

    print(
        "OK   generated blur implementation"
    )


# ======================================================================
# 7. APPLICATION WIRING
# ======================================================================

def validate_application_wiring() -> None:

    adapter = parse_python(
        PROJECT_ROOT
        / "execution"
        / "comfy_workflow_adapter.py"
    )

    adapter_class = (
        classes(adapter)
        .get(
            "ComfyWorkflowAdapter"
        )
    )

    require(
        adapter_class is not None,
        "ComfyWorkflowAdapter class "
        "is missing.",
    )

    require_subset(
        methods(adapter_class),
        {
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
        },
        "ComfyWorkflowAdapter",
    )

    require_subset(
        string_constants(adapter),
        {
            LEGACY_LATENT_LOADER,
            MODERN_LATENT_LOADER,
            "VHS_LoadVideo",
            "VHS_VideoCombine",
            "CLIPTextEncode",
            "LoadImage",
        },
        "ComfyWorkflowAdapter node contract",
    )

    contracts = {
        (
            "execution/comfy_client.py",
            "ComfyClient",
            {
                "__init__",
                "health_check",
                "queue_prompt",
                "get_history",
                "wait_for_prompt",
                "download_file",
                "find_video_outputs",
            },
        ),
        (
            "execution/shot_executor.py",
            "ShotExecutor",
            {
                "__init__",
                "execute_raw",
                "execute_detailer",
                "execute_shot",
            },
        ),
        (
            "execution/production_runner.py",
            "ProductionRunner",
            {
                "__init__",
                "prepare",
                "run",
                "_run_one_shot",
                "_dict_to_shot",
            },
        ),
        (
            "scheduler/gpu_scheduler.py",
            "GPUScheduler",
            {
                "run",
            },
        ),
        (
            "execution/checkpoint_manager.py",
            "CheckpointManager",
            {
                "initialize_shot",
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
            },
        ),
        (
            "pipeline/production_orchestrator.py",
            "ProductionOrchestrator",
            {
                "create_production_plan",
                "unload_models",
            },
        ),
        (
            "pipeline/production_manager.py",
            "ProductionManager",
            {
                "get_pipeline",
            },
        ),
        (
            "execution/assembly_manager.py",
            "AssemblyManager",
            {
                "assemble",
                "check_ffmpeg",
                "create_concat_file",
            },
        ),
    }

    for (
        relative,
        class_name,
        required_methods,
    ) in contracts:

        module = parse_python(
            PROJECT_ROOT
            / relative
        )

        class_node = (
            classes(module)
            .get(
                class_name
            )
        )

        require(
            class_node is not None,
            f"{class_name} missing in "
            f"{relative}",
        )

        require_subset(
            methods(class_node),
            required_methods,
            class_name,
        )

    # --------------------------------------------------------------
    # ShotExecutor -> adapters -> ComfyClient
    # --------------------------------------------------------------

    executor = parse_python(
        PROJECT_ROOT
        / "execution"
        / "shot_executor.py"
    )

    require(
        has_call(
            executor,
            "workflow_adapter",
            "to_api_workflow",
        ),
        "ShotExecutor does not convert "
        "the BASE workflow.",
    )

    require(
        has_call(
            executor,
            "detailer_workflow_adapter",
            "to_api_workflow",
        ),
        "ShotExecutor does not convert "
        "the DETAILER workflow.",
    )

    require(
        has_call(
            executor,
            "client",
            "queue_prompt",
        ),
        "ShotExecutor does not queue "
        "ComfyUI prompts.",
    )

    require(
        has_call(
            executor,
            "client",
            "wait_for_prompt",
        ),
        "ShotExecutor does not wait for "
        "ComfyUI completion.",
    )

    require(
        has_call(
            executor,
            "client",
            "download_file",
        ),
        "ShotExecutor does not download "
        "ComfyUI outputs.",
    )

    require(
        "ltx_raw/"
        in string_constants(
            executor
        ),
        "Raw LTX output path is missing.",
    )

    require(
        "ltx_master/"
        in string_constants(
            executor
        ),
        "Final LTX master output path is missing.",
    )

    # --------------------------------------------------------------
    # ProductionRunner -> Scheduler
    # --------------------------------------------------------------

    runner = parse_python(
        PROJECT_ROOT
        / "execution"
        / "production_runner.py"
    )

    require(
        has_call(
            runner,
            "scheduler",
            "run",
        ),
        "ProductionRunner does not call "
        "the GPU scheduler.",
    )

    # --------------------------------------------------------------
    # generate_video.py
    # --------------------------------------------------------------

    entry = parse_python(
        PROJECT_ROOT
        / "scripts"
        / "generate_video.py"
    )

    require_subset(
        names(entry),
        {
            "ProductionOrchestrator",
            "ProductionRunner",
            "ProductionManager",
        },
        "generate_video.py",
    )

    require_subset(
        string_constants(entry),
        {
            "--story",
            "--mode",
            "--gpu-url",
        },
        "generate_video.py CLI",
    )

    require(
        has_call(
            entry,
            "orchestrator",
            "create_production_plan",
        ),
        "generate_video.py does not call "
        "create_production_plan().",
    )

    require(
        has_call(
            entry,
            "orchestrator",
            "unload_models",
        ),
        "generate_video.py does not unload "
        "the planning model before "
        "LTX execution.",
    )

    require(
        has_call(
            entry,
            "runner",
            "run",
        ),
        "generate_video.py does not call "
        "ProductionRunner.run().",
    )

    print(
        "OK   application wiring contracts"
    )


# ======================================================================
# 8. CPU PREFLIGHT CONTRACT
# ======================================================================

def validate_cpu_preflight() -> None:

    module = parse_python(
        CPU_PREFLIGHT
    )

    constants = string_constants(
        module
    )

    source = read_text(
        CPU_PREFLIGHT
    )

    lower_source = (
        source.lower()
    )

    required_markers = {
        "LTX-13B CPU PREFLIGHT",
        "Python AST",
        "ComfyWorkflowAdapter",
        "LTXVLatentUpsamplerModelLoader",
        "LatentUpscaleModelLoader",
    }

    require_subset(
        constants,
        required_markers,
        "CPU preflight",
    )

    forbidden_runtime_actions = (
        "torch.cuda.is_available",
        "torch.cuda.device_count",
        "start_comfyui",
        "subprocess.Popen",
    )

    for marker in (
        forbidden_runtime_actions
    ):

        require(
            marker not in lower_source,
            "CPU preflight must remain CPU-only; "
            f"forbidden runtime marker found: "
            f"{marker}",
        )

    require(
        "import ast"
        in source,
        "CPU preflight must use AST-based "
        "repository checks.",
    )

    require(
        "to_api_workflow"
        in source,
        "CPU preflight must validate "
        "workflow conversion.",
    )

    print(
        "OK   CPU preflight contract"
    )


# ======================================================================
# 9. KAGGLE LAUNCHER / BOOTSTRAP WIRING
# ======================================================================

def validate_kaggle_wiring(
    lock: dict[str, Any],
) -> None:

    launch = parse_python(
        PROJECT_ROOT
        / "kaggle"
        / "launch.py"
    )

    launch_constants = (
        string_constants(
            launch
        )
    )

    require_subset(
        launch_constants,
        {
            "LTX-13B MODERN ONE-CELL STARTUP",
            "preflight_modern.py",
            "bootstrap.py",
            "start_comfyui_tunnel.py",
        },
        "kaggle/launch.py",
    )

    bootstrap_path = (
        PROJECT_ROOT
        / "kaggle"
        / "bootstrap.py"
    )

    bootstrap = parse_python(
        bootstrap_path
    )

    bootstrap_source = read_text(
        bootstrap_path
    )

    bootstrap_names = names(
        bootstrap
    )

    bootstrap_constants = (
        string_constants(
            bootstrap
        )
    )

    # --------------------------------------------------------------
    # Lock must be consumed dynamically.
    # --------------------------------------------------------------

    require(
        "compatibility_lock.yaml"
        in bootstrap_source,
        "bootstrap.py does not reference "
        "compatibility_lock.yaml.",
    )

    require(
        "lock"
        in bootstrap_names
        or "LOCK"
        in bootstrap_names,
        "bootstrap.py does not appear to "
        "load the compatibility lock.",
    )

    require(
        "models"
        in bootstrap_source,
        "bootstrap.py does not contain "
        "dynamic model handling.",
    )

    require(
        "filename"
        in bootstrap_source,
        "bootstrap.py does not read locked "
        "model filenames dynamically.",
    )

    require(
        "target"
        in bootstrap_source,
        "bootstrap.py does not read locked "
        "model targets dynamically.",
    )

    # DO NOT require the literal model filenames here.
    #
    # The whole point of the architecture is:
    #
    # compatibility_lock.yaml
    #          ↓
    # bootstrap.py
    #
    # not duplicated literals in bootstrap.py.

    # --------------------------------------------------------------
    # Frontend policy
    # --------------------------------------------------------------

    require(
        (
            "Frontend override: NONE"
            in bootstrap_constants
        )
        or (
            "frontend"
            in bootstrap_source
        ),
        "bootstrap.py does not contain the "
        "locked frontend handling.",
    )

    # --------------------------------------------------------------
    # Preflight
    # --------------------------------------------------------------

    preflight_path = (
        PROJECT_ROOT
        / "kaggle"
        / "preflight_modern.py"
    )

    preflight = parse_python(
        preflight_path
    )

    preflight_source = read_text(
        preflight_path
    )

    require(
        "compatibility_lock.yaml"
        in preflight_source,
        "preflight_modern.py does not consume "
        "compatibility_lock.yaml.",
    )

    require(
        "torch.cuda.is_available"
        in preflight_source,
        "preflight_modern.py does not validate "
        "CUDA availability.",
    )

    require(
        "torchvision"
        in preflight_source,
        "preflight_modern.py does not validate "
        "torchvision.",
    )

    # --------------------------------------------------------------
    # Tunnel / ComfyUI startup
    # --------------------------------------------------------------

    tunnel_path = (
        PROJECT_ROOT
        / "kaggle"
        / "start_comfyui_tunnel.py"
    )

    tunnel = parse_python(
        tunnel_path
    )

    tunnel_source = read_text(
        tunnel_path
    )

    require(
        "COMFYUI_PORT"
        in names(tunnel)
        or "COMFYUI_PORT"
        in tunnel_source,
        "ComfyUI port configuration "
        "is missing.",
    )

    require(
        "8188"
        in string_constants(
            tunnel
        )
        or "8188"
        in tunnel_source,
        "Production launcher no longer "
        "contains the expected ComfyUI port 8188.",
    )

    require(
        "main.py"
        in tunnel_source,
        "start_comfyui_tunnel.py does not "
        "start ComfyUI.",
    )

    require(
        "cloudflared"
        in tunnel_source.lower(),
        "start_comfyui_tunnel.py does not "
        "configure the Cloudflare tunnel.",
    )

    # --------------------------------------------------------------
    # Removed duplicate configuration must not be referenced.
    # --------------------------------------------------------------

    for relative in (
        "kaggle/bootstrap.py",
        "kaggle/config.py",
        "kaggle/preflight_modern.py",
        "kaggle/launch.py",
        "kaggle/start_comfyui.py",
        "kaggle/start_comfyui_tunnel.py",
    ):

        text = read_text(
            PROJECT_ROOT
            / relative
        )

        require(
            "model_paths.yaml"
            not in text,
            f"Removed model_paths.yaml is still "
            f"referenced by {relative}.",
        )

    # --------------------------------------------------------------
    # Model source consistency.
    # Validate the lock itself rather than looking for duplicated
    # strings in bootstrap.py.
    # --------------------------------------------------------------

    for name, spec in (
        lock["models"].items()
    ):

        require(
            spec["dataset"].startswith(
                "/kaggle/input/"
            ),
            f"Model dataset path invalid "
            f"for {name}.",
        )

        require(
            spec["target"].startswith(
                "models/"
            ),
            f"Model target invalid "
            f"for {name}.",
        )

    print(
        "OK   Kaggle launcher/bootstrap wiring"
    )


# ======================================================================
# 10. MATERIALIZED RUNTIME
# ======================================================================

def validate_materialized_runtime(
    lock: dict[str, Any],
    require_runtime: bool,
    require_cuda: bool,
) -> None:

    exists = (
        COMFYUI_DIR.exists()
    )

    if not exists:

        if require_runtime:
            fail(
                "ComfyUI is not materialized, "
                "but --require-runtime was supplied."
            )

        print(
            "SKIP runtime: ComfyUI/ "
            "is not materialized."
        )

        return

    try:

        import torch
        import torchvision

    except Exception as error:

        fail(
            "Materialized runtime cannot import "
            "torch/torchvision:\n"
            f"{type(error).__name__}: "
            f"{error}"
        )

    runtime = lock[
        "python_runtime"
    ]

    require(
        torch.__version__
        == runtime["torch"],
        "Torch version mismatch:\n"
        f"Expected: {runtime['torch']}\n"
        f"Actual:   {torch.__version__}",
    )

    require(
        torchvision.__version__
        == runtime["torchvision"],
        "Torchvision version mismatch:\n"
        f"Expected: "
        f"{runtime['torchvision']}\n"
        f"Actual: "
        f"{torchvision.__version__}",
    )

    if require_cuda:

        require(
            torch.cuda.is_available(),
            "CUDA is unavailable but "
            "--require-cuda was supplied.",
        )

        require(
            torch.cuda.device_count()
            > 0,
            "No CUDA GPU detected.",
        )

        print(
            "OK   CUDA:"
            f" {torch.cuda.device_count()} GPU(s)"
        )

        for index in range(
            torch.cuda.device_count()
        ):

            print(
                f"     GPU {index}: "
                f"{torch.cuda.get_device_name(index)}"
            )

    packages = {
        lock["comfyui"]["frontend"]["package"]:
            lock["comfyui"]["frontend"]["version"],

        lock["comfyui"]["workflow_templates"]["package"]:
            lock["comfyui"]["workflow_templates"]["version"],

        lock["comfyui"]["embedded_docs"]["package"]:
            lock["comfyui"]["embedded_docs"]["version"],

        lock["comfyui"]["comfy_kitchen"]["package"]:
            lock["comfyui"]["comfy_kitchen"]["version"],

        lock["comfyui"]["comfy_aimdo"]["package"]:
            lock["comfyui"]["comfy_aimdo"]["version"],

        "torchsde":
            runtime["torchsde"],

        "spandrel":
            runtime["spandrel"],

        "av":
            runtime["av"],

        "gguf":
            runtime["gguf"],
    }

    for package, expected in (
        packages.items()
    ):

        try:

            actual = (
                importlib.metadata
                .version(
                    package
                )
            )

        except (
            importlib.metadata
            .PackageNotFoundError
        ):

            fail(
                "Missing locked runtime package:\n"
                f"{package}=={expected}"
            )

        require(
            actual == expected,
            "Package version mismatch:\n"
            f"{package}\n"
            f"Expected: {expected}\n"
            f"Actual:   {actual}",
        )

    print(
        "OK   locked runtime packages"
    )

    # --------------------------------------------------------------
    # ComfyUI exact revision
    # --------------------------------------------------------------

    expected_comfy = (
        lock["comfyui"]["commit"]
    )

    actual_comfy = (
        subprocess.check_output(
            [
                "git",
                "-C",
                str(COMFYUI_DIR),
                "rev-parse",
                "HEAD",
            ],
            text=True,
        )
        .strip()
    )

    require(
        actual_comfy
        == expected_comfy,
        "Materialized ComfyUI revision mismatch:\n"
        f"Expected: {expected_comfy}\n"
        f"Actual:   {actual_comfy}",
    )

    print(
        "OK   ComfyUI locked commit"
    )

    # --------------------------------------------------------------
    # Custom nodes exact revisions
    # --------------------------------------------------------------

    custom_root = (
        COMFYUI_DIR
        / "custom_nodes"
    )

    for name, spec in (
        lock["custom_nodes"].items()
    ):

        node_path = (
            custom_root
            / name
        )

        require(
            node_path.is_dir(),
            f"Materialized custom node missing:\n"
            f"{name}",
        )

        actual = (
            subprocess.check_output(
                [
                    "git",
                    "-C",
                    str(node_path),
                    "rev-parse",
                    "HEAD",
                ],
                text=True,
            )
            .strip()
        )

        require(
            actual
            == spec["commit"],
            "Custom node revision mismatch:\n"
            f"{name}\n"
            f"Expected: "
            f"{spec['commit']}\n"
            f"Actual: {actual}",
        )

    print(
        "OK   custom-node locked commits"
    )

    # --------------------------------------------------------------
    # Model targets
    # --------------------------------------------------------------

    for name, spec in (
        lock["models"].items()
    ):

        target = (
            COMFYUI_DIR
            / spec["target"]
        )

        require(
            target.exists(),
            "Locked model target missing:\n"
            f"{name}\n"
            f"{target}",
        )

    print(
        "OK   locked model targets"
    )


# ======================================================================
# 11. OPTIONAL LIVE COMFYUI CHECK
# ======================================================================

def validate_live_comfyui(
    require_live: bool,
    base_url: str,
) -> None:

    if not require_live:

        return

    from urllib.request import (
        urlopen,
    )
    from urllib.error import (
        URLError,
        HTTPError,
    )

    url = (
        base_url.rstrip("/")
        + "/system_stats"
    )

    try:

        with urlopen(
            url,
            timeout=10,
        ) as response:

            status = response.status

    except (
        HTTPError,
        URLError,
        OSError,
    ) as error:

        fail(
            "Live ComfyUI health check failed:\n"
            f"{base_url}\n"
            f"{error}"
        )

    require(
        status == 200,
        "ComfyUI system_stats returned "
        f"HTTP {status}.",
    )

    print(
        "OK   live ComfyUI API:"
        f" {base_url}"
    )


# ======================================================================
# 12. CLI
# ======================================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Validate the complete "
            "LTX-13B repository."
        )
    )

    parser.add_argument(
        "--require-runtime",
        action="store_true",
        help=(
            "Fail when ComfyUI is not "
            "materialized."
        ),
    )

    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help=(
            "Require CUDA/GPU availability. "
            "Implies --require-runtime."
        ),
    )

    parser.add_argument(
        "--require-live",
        action="store_true",
        help=(
            "Require a live ComfyUI API "
            "at --comfy-url."
        ),
    )

    parser.add_argument(
        "--comfy-url",
        default="http://127.0.0.1:8188",
        help=(
            "ComfyUI base URL for "
            "--require-live."
        ),
    )

    return parser.parse_args()


# ======================================================================
# MAIN
# ======================================================================

def main() -> int:

    args = parse_args()

    if args.require_cuda:
        args.require_runtime = True

    print(
        "=" * 88
    )
    print(
        "LTX-13B PROJECT VALIDATOR"
    )
    print(
        "=" * 88
    )

    print(
        f"Project: {PROJECT_ROOT}"
    )

    lock = load_lock()

    # --------------------------------------------------------------
    # Repository
    # --------------------------------------------------------------

    validate_files()
    validate_python()

    # --------------------------------------------------------------
    # Compatibility
    # --------------------------------------------------------------

    validate_lock(
        lock
    )

    # --------------------------------------------------------------
    # BASE workflow
    # --------------------------------------------------------------

    base_workflow = (
        validate_workflow_graph(
            BASE_WORKFLOW,
            {
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
            },
        )
    )

    validate_workflow_conversion(
        BASE_WORKFLOW,
        base_workflow,
        detailer=False,
    )

    # --------------------------------------------------------------
    # DETAILER workflow
    # --------------------------------------------------------------

    detailer_workflow = (
        validate_workflow_graph(
            DETAILER_WORKFLOW,
            {
                "VHS_LoadVideo",
                "LTXVLoopingSampler",
                LEGACY_LATENT_LOADER,
                "LTXVLatentUpsampler",
                "LTXVTiledVAEDecode",
                "LTXVFilmGrain",
                "LoraLoaderModelOnly",
                "VHS_VideoCombine",
            },
        )
    )

    validate_workflow_conversion(
        DETAILER_WORKFLOW,
        detailer_workflow,
        detailer=True,
    )

    # --------------------------------------------------------------
    # Compatibility builder
    # --------------------------------------------------------------

    validate_compatibility_builder()

    # --------------------------------------------------------------
    # Application
    # --------------------------------------------------------------

    validate_application_wiring()

    # --------------------------------------------------------------
    # CPU preflight
    # --------------------------------------------------------------

    validate_cpu_preflight()

    # --------------------------------------------------------------
    # Kaggle
    # --------------------------------------------------------------

    validate_kaggle_wiring(
        lock
    )

    # --------------------------------------------------------------
    # Runtime
    # --------------------------------------------------------------

    validate_materialized_runtime(
        lock,
        require_runtime=(
            args.require_runtime
        ),
        require_cuda=(
            args.require_cuda
        ),
    )

    # --------------------------------------------------------------
    # Optional live ComfyUI API
    # --------------------------------------------------------------

    validate_live_comfyui(
        require_live=(
            args.require_live
        ),
        base_url=(
            args.comfy_url
        ),
    )

    # --------------------------------------------------------------
    # FFmpeg
    # --------------------------------------------------------------

    if shutil.which(
        "ffmpeg"
    ) is None:

        print(
            "WARNING ffmpeg is not "
            "available in the current environment."
        )

    print()
    print(
        "=" * 88
    )
    print(
        "✅ LTX-13B VALIDATION PASSED"
    )
    print(
        "=" * 88
    )

    print(
        "Verified:"
    )

    print(
        "  ✅ repository structure"
    )

    print(
        "  ✅ Python syntax"
    )

    print(
        "  ✅ single compatibility lock"
    )

    print(
        "  ✅ no obsolete model_paths.yaml"
    )

    print(
        "  ✅ BASE workflow graph"
    )

    print(
        "  ✅ DETAILER workflow graph"
    )

    print(
        "  ✅ workflow → API conversion"
    )

    print(
        "  ✅ legacy → modern loader conversion"
    )

    print(
        "  ✅ LTX compatibility builder"
    )

    print(
        "  ✅ application wiring"
    )

    print(
        "  ✅ CPU preflight contract"
    )

    print(
        "  ✅ Kaggle bootstrap wiring"
    )

    if args.require_runtime:

        print(
            "  ✅ materialized runtime"
        )

    if args.require_cuda:

        print(
            "  ✅ CUDA"
        )

    if args.require_live:

        print(
            "  ✅ live ComfyUI API"
        )

    return 0


if __name__ == "__main__":

    try:

        raise SystemExit(
            main()
        )

    except Exception as error:

        print()
        print(
            "=" * 88
        )
        print(
            "❌ LTX-13B VALIDATION FAILED"
        )
        print(
            "=" * 88
        )
        print(
            f"{type(error).__name__}: "
            f"{error}"
        )

        raise
