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
12. Materialized ComfyUI runtime, when requested
13. Optional CUDA verification, when requested

Usage:

    python scripts/validate_project.py

For a fully materialized Kaggle runtime:

    python scripts/validate_project.py --require-runtime

For runtime + CUDA:

    python scripts/validate_project.py --require-runtime --require-cuda
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]

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


REQUIRED_FILES = (
    "planner/config.py",
    "planner/qwen_loader.py",
    "planner/story_planner.py",
    "planner/character_detector.py",
    "planner/character_planner.py",
    "planner/scene_planner.py",
    "planner/shot_planner.py",

    "pipeline/__init__.py",
    "pipeline/continuity_manager.py",
    "pipeline/modes.py",
    "pipeline/production_manager.py",
    "pipeline/production_orchestrator.py",
    "pipeline/reference_manager.py",

    "execution/__init__.py",
    "execution/checkpoint_manager.py",
    "execution/comfy_client.py",
    "execution/comfy_workflow_adapter.py",
    "execution/shot_executor.py",
    "execution/assembly_manager.py",
    "execution/production_runner.py",

    "scheduler/__init__.py",
    "scheduler/gpu_scheduler.py",
    "scheduler/shot_queue.py",

    "schemas/__init__.py",
    "schemas/character.py",
    "schemas/scene.py",
    "schemas/shot.py",
    "schemas/parser.py",

    "kaggle/compatibility_lock.yaml",
    "kaggle/bootstrap.py",
    "kaggle/config.py",
    "kaggle/launch.py",
    "kaggle/preflight_modern.py",
    "kaggle/start_comfyui.py",
    "kaggle/start_comfyui_tunnel.py",

    "compatibility/prepare_modern_ltx.py",

    "scripts/generate_video.py",
    "scripts/validate_project.py",

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


def read_text(path: Path) -> str:
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

    if not isinstance(data, dict):
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

    for node in ast.walk(module):

        if isinstance(
            node,
            ast.Name,
        ):
            result.add(node.id)

        elif isinstance(
            node,
            ast.Attribute,
        ):
            result.add(node.attr)

        elif isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.ClassDef,
            ),
        ):
            result.add(node.name)

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
        for node in ast.walk(module)
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

    for node in ast.walk(module):

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

    return sorted(files)


# ======================================================================
# 1. REPOSITORY STRUCTURE
# ======================================================================

def validate_files() -> None:

    for relative in REQUIRED_FILES:

        path = (
            PROJECT_ROOT
            / relative
        )

        require(
            path.is_file(),
            "Required project file missing:\n"
            f"{path}",
        )

    for obsolete in OBSOLETE_FILES:

        require(
            not obsolete.exists(),
            "Obsolete duplicated configuration still exists:\n"
            f"{obsolete}\n\n"
            "compatibility_lock.yaml must remain the "
            "single source of truth.",
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
        parse_python(path)

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
            read_text(LOCK_FILE)
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

    comfy = lock["comfyui"]

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
        comfy["repository"].endswith(
            "ComfyUI.git"
        ),
        "Invalid ComfyUI repository "
        "in compatibility lock.",
    )

    require(
        re.fullmatch(
            r"[0-9a-f]{40}",
            comfy["commit"],
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
            package.get("package"),
            f"Missing package name: "
            f"{package_key}",
        )

        require(
            package.get("version"),
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
            runtime.get(package),
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
            spec.get("repository"),
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
        detailer[
            "legacy_loader"
        ]
        == LEGACY_LATENT_LOADER,
        "Lock/detailer legacy loader "
        "is inconsistent.",
    )

    require(
        detailer[
            "modern_loader"
        ]
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

    workflow = parse_json(path)

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
        f"Workflow contains no nodes:\n"
        f"{path}",
    )

    require(
        isinstance(
            links,
            list,
        ),
        f"Workflow links are not a list:\n"
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
            f"Workflow contains invalid node:\n"
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
            f"Workflow node has no ID:\n"
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

        node_map[key] = node

    node_types = {
        str(
            node.get("type")
        )
        for node in nodes
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
            f"Malformed workflow link:\n"
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

        for input_def in node.get(
            "inputs",
            [],
        ):

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
                f"references missing link "
                f"{link_id}:\n{path}",
            )

    # A real executable workflow needs a video output.
    require(
        "VHS_VideoCombine"
        in node_types,
        f"Workflow has no "
        f"VHS_VideoCombine output:\n"
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
        f"Empty API workflow:\n"
        f"{path}",
    )

    class_types = {
        node.get(
            "class_type"
        )
        for node in api.values()
        if isinstance(
            node,
            dict,
        )
    }

    require(
        "VHS_VideoCombine"
        in class_types,
        f"Converted workflow has no "
        f"VHS_VideoCombine:\n{path}",
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

    # Validate all API graph references.
    for node_id, node in (
        api.items()
    ):

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

    if detailer:

        source_types = {
            node.get(
                "type"
            )
            for node
            in workflow.get(
                "nodes",
                [],
            )
            if isinstance(
                node,
                dict,
            )
        }

        require(
            LEGACY_LATENT_LOADER
            in source_types,
            "Detailer source no longer contains "
            "the expected legacy loader.",
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

    constants = (
        string_constants(module)
    )

    require_subset(
        names(module)
        | constants,
        {
            "load_lock",
            "get_legacy_commit",
            "patch_blur",
            "write_curated_init",
            "build_compat_package",
            "compatibility_lock.yaml",
            "LTX098ModernCompat",
        },
        "Compatibility builder",
    )

    # The generated legacy initializer must expose the
    # legacy nodes required by the lock.
    initializer_function = next(
        (
            node
            for node
            in ast.walk(module)
            if isinstance(
                node,
                (
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                ),
            )
            and node.name
            == "write_curated_init"
        ),
        None,
    )

    require(
        initializer_function is not None,
        "write_curated_init() is missing.",
    )

    initializer_strings: set[str] = set()

    for node in ast.walk(
        initializer_function
    ):

        if (
            isinstance(
                node,
                ast.Constant,
            )
            and isinstance(
                node.value,
                str,
            )
        ):

            initializer_strings.add(
                node.value
            )

    require_subset(
        initializer_strings,
        {
            "LTXVBaseSampler",
            "LTXVLoopingSampler",
            "LTXVTiledSampler",
            "LTXVTiledVAEDecode",
            "LTXVLatentUpsampler",
            "LTXVLatentUpsamplerModelLoader",
            "LTXVFilmGrain",
            "STGGuiderAdvanced",
            "Set VAE Decoder Noise",
        },
        "Generated compatibility initializer",
    )

    # Verify the blur replacement contains the required function
    # and native convolution.
    patch_function = next(
        (
            node
            for node
            in ast.walk(module)
            if isinstance(
                node,
                (
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                ),
            )
            and node.name
            == "patch_blur"
        ),
        None,
    )

    require(
        patch_function is not None,
        "patch_blur() is missing.",
    )

    blur_source_found = False
    conv2d_found = False

    for node in ast.walk(
        patch_function
    ):

        if (
            isinstance(
                node,
                ast.Constant,
            )
            and node.value
            == "blur_internal(image, blur_radius):"
        ):
            blur_source_found = True

        if (
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
        ):
            conv2d_found = True

    require(
        blur_source_found
        or "blur_internal"
        in constants,
        "Compatibility builder does not contain "
        "blur_internal implementation.",
    )

    require(
        conv2d_found,
        "Compatibility builder does not use native "
        "torch conv2d for blur.",
    )

    print(
        "OK   LTX 0.9.8 compatibility builder"
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
        "execution/comfy_client.py": (
            "ComfyClient",
            {
                "health_check",
                "queue_prompt",
                "get_history",
                "wait_for_prompt",
                "download_file",
                "find_video_outputs",
            },
        ),
        "execution/shot_executor.py": (
            "ShotExecutor",
            {
                "__init__",
                "execute_raw",
                "execute_detailer",
                "execute_shot",
            },
        ),
        "execution/production_runner.py": (
            "ProductionRunner",
            {
                "__init__",
                "prepare",
                "run",
                "_run_one_shot",
                "_dict_to_shot",
            },
        ),
        "scheduler/gpu_scheduler.py": (
            "GPUScheduler",
            {"run"},
        ),
        "execution/checkpoint_manager.py": (
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
        "pipeline/production_orchestrator.py": (
            "ProductionOrchestrator",
            {
                "create_production_plan",
                "unload_models",
            },
        ),
        "pipeline/production_manager.py": (
            "ProductionManager",
            {"get_pipeline"},
        ),
        "execution/assembly_manager.py": (
            "AssemblyManager",
            {
                "assemble",
                "check_ffmpeg",
                "create_concat_file",
            },
        ),
    }

    for relative, (
        class_name,
        required_methods,
    ) in contracts.items():

        module = parse_python(
            PROJECT_ROOT
            / relative
        )

        class_node = (
            classes(module)
            .get(class_name)
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
    # ShotExecutor -> workflow adapters -> ComfyClient
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
        in string_constants(executor),
        "Raw LTX output path is missing.",
    )

    require(
        "ltx_master/"
        in string_constants(executor),
        "Final LTX master output path is missing.",
    )

    # --------------------------------------------------------------
    # generate_video.py -> planning -> execution
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
        "the planning model before LTX execution.",
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
# 8. KAGGLE LAUNCHER WIRING
# ======================================================================

def validate_kaggle_wiring(
    lock: dict[str, Any],
) -> None:

    launch = parse_python(
        PROJECT_ROOT
        / "kaggle"
        / "launch.py"
    )

    launch_strings = (
        string_constants(launch)
    )

    require_subset(
        launch_strings,
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

    bootstrap_strings = (
        string_constants(bootstrap)
    )

    require(
        "compatibility_lock.yaml"
        in bootstrap_strings,
        "bootstrap.py does not consume "
        "compatibility_lock.yaml.",
    )

    require(
        "Frontend override: NONE"
        in bootstrap_strings,
        "bootstrap.py no longer confirms "
        "the pinned frontend policy.",
    )

    require(
        "models"
        in bootstrap_strings,
        "bootstrap.py does not contain "
        "locked model handling.",
    )

    # Every locked model filename and target must be present in bootstrap.
    for name, spec in (
        lock["models"].items()
    ):

        require(
            spec["filename"]
            in bootstrap_strings,
            f"bootstrap.py does not consume "
            f"locked filename for {name}.",
        )

        require(
            spec["target"]
            in bootstrap_strings,
            f"bootstrap.py does not consume "
            f"locked target for {name}.",
        )

    preflight = parse_python(
        PROJECT_ROOT
        / "kaggle"
        / "preflight_modern.py"
    )

    require(
        "compatibility_lock.yaml"
        in string_constants(preflight),
        "preflight_modern.py does not consume "
        "compatibility_lock.yaml.",
    )

    tunnel = parse_python(
        PROJECT_ROOT
        / "kaggle"
        / "start_comfyui_tunnel.py"
    )

    tunnel_strings = (
        string_constants(tunnel)
    )

    require(
        "trycloudflare.com"
        in "\n".join(
            sorted(tunnel_strings)
        ),
        "Cloudflare quick-tunnel contract "
        "is missing.",
    )

    require(
        "--listen"
        in tunnel_strings
        or "COMFYUI_HOST"
        in names(tunnel),
        "ComfyUI host/listen configuration "
        "is missing.",
    )

    require(
        "--port"
        in tunnel_strings
        or "COMFYUI_PORT"
        in names(tunnel),
        "ComfyUI port configuration "
        "is missing.",
    )

    # The removed model-path file must not be referenced.
    for relative in (
        "kaggle/bootstrap.py",
        "kaggle/config.py",
        "kaggle/preflight_modern.py",
        "kaggle/launch.py",
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

    print(
        "OK   Kaggle launcher/bootstrap wiring"
    )


# ======================================================================
# 9. MATERIALIZED RUNTIME
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
            "SKIP runtime: ComfyUI/"
            " is not materialized."
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
                .version(package)
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
            f"Package version mismatch:\n"
            f"{package}\n"
            f"Expected: {expected}\n"
            f"Actual:   {actual}",
        )

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
            f"Custom node revision mismatch:\n"
            f"{name}\n"
            f"Expected: "
            f"{spec['commit']}\n"
            f"Actual: {actual}",
        )

    for name, spec in (
        lock["models"].items()
    ):

        target = (
            COMFYUI_DIR
            / spec["target"]
        )

        require(
            target.exists(),
            f"Locked model target missing:\n"
            f"{name}\n"
            f"{target}",
        )

    print(
        "OK   materialized ComfyUI runtime"
    )


# ======================================================================
# 10. MAIN
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

    return parser.parse_args()


def main() -> int:

    args = parse_args()

    if args.require_cuda:
        args.require_runtime = True

    print("=" * 88)
    print("LTX-13B PROJECT VALIDATOR")
    print("=" * 88)

    lock = load_lock()

    validate_files()
    validate_python()
    validate_lock(lock)

    # --------------------------------------------------------------
    # BASE WORKFLOW
    # --------------------------------------------------------------

    base = validate_workflow_graph(
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

    validate_workflow_conversion(
        BASE_WORKFLOW,
        base,
        detailer=False,
    )

    # --------------------------------------------------------------
    # DETAILER WORKFLOW
    # --------------------------------------------------------------

    detailer = validate_workflow_graph(
        DETAILER_WORKFLOW,
        {
            "VHS_LoadVideo",
            "LTXVLoopingSampler",
            LEGACY_LATENT_LOADER,
            "LTXVTiledVAEDecode",
            "LTXVFilmGrain",
            "LoraLoaderModelOnly",
            "VHS_VideoCombine",
        },
    )

    validate_workflow_conversion(
        DETAILER_WORKFLOW,
        detailer,
        detailer=True,
    )

    # --------------------------------------------------------------
    # COMPATIBILITY
    # --------------------------------------------------------------

    validate_compatibility_builder()

    # --------------------------------------------------------------
    # APPLICATION
    # --------------------------------------------------------------

    validate_application_wiring()

    # --------------------------------------------------------------
    # KAGGLE
    # --------------------------------------------------------------

    validate_kaggle_wiring(
        lock
    )

    # --------------------------------------------------------------
    # RUNTIME
    # --------------------------------------------------------------

    validate_materialized_runtime(
        lock,
        require_runtime=args.require_runtime,
        require_cuda=args.require_cuda,
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

    print("=" * 88)
    print("✅ LTX-13B VALIDATION PASSED")
    print("=" * 88)

    print()
    print(
        "Verified:"
    )
    print(
        "  • repository structure"
    )
    print(
        "  • Python syntax"
    )
    print(
        "  • compatibility lock"
    )
    print(
        "  • single model-path authority"
    )
    print(
        "  • BASE workflow graph"
    )
    print(
        "  • DETAILER workflow graph"
    )
    print(
        "  • real workflow → API conversion"
    )
    print(
        "  • legacy → modern latent loader conversion"
    )
    print(
        "  • LTX compatibility builder"
    )
    print(
        "  • production application wiring"
    )
    print(
        "  • Kaggle launcher/bootstrap wiring"
    )
    print(
        "  • materialized runtime when requested"
    )

    return 0


if __name__ == "__main__":

    try:

        raise SystemExit(
            main()
        )

    except Exception as error:

        print()
        print("=" * 88)
        print("❌ LTX-13B VALIDATION FAILED")
        print("=" * 88)
        print(
            f"{type(error).__name__}: "
            f"{error}"
        )

        raise
