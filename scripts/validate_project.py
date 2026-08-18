#!/usr/bin/env python3

"""
LTX-13B PROJECT VALIDATOR

This is the repository-level final validation gate.

Validation layers:

1. Repository structure
2. Obsolete configuration detection
3. Python syntax
4. Compatibility lock
5. Model lock contract
6. Workflow JSON
7. BASE workflow contract
8. DETAILER workflow contract
9. Workflow adapter contract
10. Real DETAILER legacy -> modern conversion
11. Compatibility builder contract
12. Production application wiring
13. Kaggle bootstrap wiring
14. CPU preflight contract
15. Canonical ComfyUI port contract
16. Live runtime verifier contract

Normal repository validation:

    python scripts/validate_project.py

This does NOT require GPU or a running ComfyUI server.

Live runtime validation:

    python scripts/validate_project.py --require-runtime

Live runtime + CUDA:

    python scripts/validate_project.py \
        --require-runtime \
        --require-cuda

Single source of truth:

    kaggle/compatibility_lock.yaml

Obsolete:

    kaggle/model_paths.yaml
    kaggle/runtime_requirements.lock

Canonical ComfyUI local port:

    8188
"""

from __future__ import annotations

import argparse
import ast
import importlib
import json
import subprocess
import sys
import urllib.error
import urllib.request

from pathlib import Path
from typing import Any


# ======================================================================
# PATHS
# ======================================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

KAGGLE_DIR = (
    PROJECT_ROOT
    / "kaggle"
)

SCRIPTS_DIR = (
    PROJECT_ROOT
    / "scripts"
)

WORKFLOWS_DIR = (
    PROJECT_ROOT
    / "workflows"
)

LOCK_FILE = (
    KAGGLE_DIR
    / "compatibility_lock.yaml"
)

CPU_PREFLIGHT = (
    SCRIPTS_DIR
    / "cpu_preflight.py"
)

LIVE_RUNTIME = (
    KAGGLE_DIR
    / "verify_live_runtime.py"
)

BOOTSTRAP = (
    KAGGLE_DIR
    / "bootstrap.py"
)

LAUNCH = (
    KAGGLE_DIR
    / "launch.py"
)

PREFLIGHT_MODERN = (
    KAGGLE_DIR
    / "preflight_modern.py"
)

TUNNEL = (
    KAGGLE_DIR
    / "start_comfyui_tunnel.py"
)

GENERATE_VIDEO = (
    SCRIPTS_DIR
    / "generate_video.py"
)

ADAPTER = (
    PROJECT_ROOT
    / "execution"
    / "comfy_workflow_adapter.py"
)

COMPATIBILITY_BUILDER = (
    PROJECT_ROOT
    / "compatibility"
    / "prepare_modern_ltx.py"
)

BASE_WORKFLOW = (
    WORKFLOWS_DIR
    / "baseline"
    / "ltxv-13b-dist-i2v-base.json"
)

DETAILER_WORKFLOW = (
    WORKFLOWS_DIR
    / "detailer"
    / "ltxv-13b-098-ic-lora-upscale.json"
)


# ======================================================================
# CONSTANTS
# ======================================================================

CANONICAL_COMFYUI_PORT = 8188
STALE_COMFYUI_PORT = 8219

LEGACY_LATENT_LOADER = (
    "LTXVLatentUpsamplerModelLoader"
)

MODERN_LATENT_LOADER = (
    "LatentUpscaleModelLoader"
)


OBSOLETE_FILES = (
    KAGGLE_DIR / "model_paths.yaml",
    KAGGLE_DIR / "runtime_requirements.lock",
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
    "kaggle/verify_live_runtime.py",

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


# ======================================================================
# HELPERS
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

    require(
        path.is_file(),
        f"Required file missing:\n{path}",
    )

    try:

        return path.read_text(
            encoding="utf-8"
        )

    except OSError as error:

        fail(
            f"Could not read file:\n"
            f"{path}\n"
            f"{error}"
        )


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

    require(
        isinstance(data, dict),
        f"JSON root must be an object:\n{path}",
    )

    return data


def all_python_files() -> list[Path]:

    result = []

    for path in PROJECT_ROOT.rglob(
        "*.py"
    ):

        relative = (
            path.relative_to(
                PROJECT_ROOT
            )
        )

        if ".git" in relative.parts:
            continue

        if "ComfyUI" in relative.parts:
            continue

        if ".runtime_ltx098" in relative.parts:
            continue

        if "__pycache__" in relative.parts:
            continue

        result.append(path)

    return sorted(result)


def node_types(
    workflow: dict[str, Any],
) -> set[str]:

    nodes = workflow.get(
        "nodes",
        [],
    )

    require(
        isinstance(nodes, list),
        "Workflow nodes must be a list.",
    )

    return {
        node.get("type")
        for node in nodes
        if isinstance(node, dict)
        and node.get("type")
    }


# ======================================================================
# 1. FILE STRUCTURE
# ======================================================================

def validate_files() -> None:

    for relative in REQUIRED_FILES:

        path = (
            PROJECT_ROOT
            / relative
        )

        require(
            path.is_file(),
            f"Required project file missing:\n{path}",
        )

    for obsolete in OBSOLETE_FILES:

        require(
            not obsolete.exists(),
            "OBSOLETE FILE DETECTED:\n"
            f"{obsolete}\n\n"
            "Remove it.\n"
            "Do NOT recreate it.\n"
            "compatibility_lock.yaml is the single "
            "source of truth.",
        )

    print(
        "OK   repository file structure"
    )


# ======================================================================
# 2. PYTHON SYNTAX
# ======================================================================

def validate_python() -> None:

    files = all_python_files()

    for path in files:

        parse_python(
            path
        )

    print(
        f"OK   Python AST syntax: "
        f"{len(files)} files"
    )


# ======================================================================
# 3. LOCK FILE
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

    require(
        isinstance(data, dict),
        "compatibility_lock.yaml must contain "
        "a mapping.",
    )

    return data


def validate_lock(
    lock: dict[str, Any],
) -> None:

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
        - set(lock)
    )

    require(
        not missing,
        "compatibility_lock.yaml missing sections:\n"
        + "\n".join(
            sorted(missing)
        ),
    )

    comfy = lock[
        "comfyui"
    ]

    models = lock[
        "models"
    ]

    require(
        isinstance(
            comfy,
            dict,
        ),
        "lock.comfyui must be a mapping.",
    )

    require(
        isinstance(
            models,
            dict,
        ),
        "lock.models must be a mapping.",
    )

    commit = str(
        comfy.get(
            "commit",
            "",
        )
    )

    require(
        len(commit) == 40
        and all(
            c in "0123456789abcdef"
            for c in commit.lower()
        ),
        "ComfyUI commit must be "
        "a 40-character hexadecimal SHA.",
    )

    required_models = {
        "ltx_q4",
        "t5_q4",
        "vae",
        "ic_lora",
        "spatial_upscaler",
    }

    missing_models = (
        required_models
        - set(models)
    )

    require(
        not missing_models,
        "compatibility_lock.yaml missing models:\n"
        + "\n".join(
            sorted(missing_models)
        ),
    )

    for name in sorted(
        required_models
    ):

        spec = models[name]

        require(
            isinstance(
                spec,
                dict,
            ),
            f"models.{name} must be a mapping.",
        )

        for field in (
            "dataset",
            "filename",
            "target",
        ):

            value = spec.get(
                field
            )

            require(
                isinstance(value, str)
                and value.strip(),
                f"models.{name}.{field} is missing.",
            )

    print(
        "OK   compatibility lock"
    )


# ======================================================================
# 4. MODEL PATH ARCHITECTURE
# ======================================================================

def validate_model_path_architecture(
    lock: dict[str, Any],
) -> None:

    bootstrap_source = read_text(
        BOOTSTRAP
    )

    require(
        "compatibility_lock.yaml"
        in bootstrap_source,
        "bootstrap.py does not reference "
        "compatibility_lock.yaml.",
    )

    require(
        "model_paths.yaml"
        not in bootstrap_source,
        "bootstrap.py still references "
        "obsolete model_paths.yaml.",
    )

    models = lock[
        "models"
    ]

    for name, spec in models.items():

        if not isinstance(
            spec,
            dict,
        ):
            continue

        for field in (
            "dataset",
            "filename",
            "target",
        ):

            require(
                isinstance(
                    spec.get(field),
                    str,
                ),
                f"Model lock field invalid: "
                f"{name}.{field}",
            )

    print(
        "OK   single-source model architecture"
    )


# ======================================================================
# 5. WORKFLOWS
# ======================================================================

def validate_workflows() -> None:

    base = parse_json(
        BASE_WORKFLOW
    )

    detailer = parse_json(
        DETAILER_WORKFLOW
    )

    base_types = node_types(
        base
    )

    detailer_types = node_types(
        detailer
    )

    base_required = {
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

    missing = (
        base_required
        - base_types
    )

    require(
        not missing,
        "BASE workflow missing nodes:\n"
        + "\n".join(
            sorted(missing)
        ),
    )

    detailer_required = {
        "VHS_LoadVideo",
        "LTXVConditioning",
        "STGGuiderAdvanced",
        "FloatToSigmas",
        "StringToFloatList",
        "LTXVLoopingSampler",
        "LTXVLatentUpsampler",
        "LTXVLatentUpsamplerModelLoader",
        "LTXVTiledVAEDecode",
        "LTXVFilmGrain",
        "LoraLoaderModelOnly",
        "VHS_VideoCombine",
    }

    missing = (
        detailer_required
        - detailer_types
    )

    require(
        not missing,
        "DETAILER source workflow missing nodes:\n"
        + "\n".join(
            sorted(missing)
        ),
    )

    require(
        MODERN_LATENT_LOADER
        not in detailer_types,
        "DETAILER source workflow already contains "
        "modern LatentUpscaleModelLoader.\n"
        "The source workflow must retain the legacy "
        "loader and the adapter performs the conversion.",
    )

    print(
        "OK   BASE workflow"
    )

    print(
        "OK   DETAILER source workflow"
    )


# ======================================================================
# 6. ADAPTER
# ======================================================================

def validate_adapter() -> None:

    module = parse_python(
        ADAPTER
    )

    classes = {
        node.name
        for node in ast.walk(
            module
        )
        if isinstance(
            node,
            ast.ClassDef,
        )
    }

    require(
        "ComfyWorkflowAdapter"
        in classes,
        "ComfyWorkflowAdapter class missing.",
    )

    source = read_text(
        ADAPTER
    )

    required_strings = {
        LEGACY_LATENT_LOADER,
        MODERN_LATENT_LOADER,
        "apply_modern_compatibility",
        "validate_modern_detailer",
    }

    for text in required_strings:

        require(
            text in source,
            "ComfyWorkflowAdapter missing contract:\n"
            f"{text}",
        )

    print(
        "OK   ComfyWorkflowAdapter contract"
    )


# ======================================================================
# 7. REAL DETAILER CONVERSION
# ======================================================================

def validate_real_detailer_conversion() -> None:

    previous_path = list(
        sys.path
    )

    try:

        root = str(
            PROJECT_ROOT
        )

        if root not in sys.path:

            sys.path.insert(
                0,
                root,
            )

        module = importlib.import_module(
            "execution.comfy_workflow_adapter"
        )

        adapter_class = getattr(
            module,
            "ComfyWorkflowAdapter",
            None,
        )

        require(
            adapter_class is not None,
            "ComfyWorkflowAdapter import failed.",
        )

        adapter = adapter_class(
            DETAILER_WORKFLOW
        )

        api_workflow = (
            adapter.to_api_workflow()
        )

        require(
            isinstance(
                api_workflow,
                dict,
            )
            and api_workflow,
            "DETAILER adapter produced "
            "an empty API workflow.",
        )

        adapter.validate_modern_detailer(
            api_workflow
        )

        api_types = {
            node.get("class_type")
            for node in api_workflow.values()
            if isinstance(
                node,
                dict,
            )
        }

        require(
            MODERN_LATENT_LOADER
            in api_types,
            "DETAILER conversion did not produce "
            "LatentUpscaleModelLoader.",
        )

        require(
            LEGACY_LATENT_LOADER
            not in api_types,
            "Legacy latent loader survived "
            "DETAILER conversion.",
        )

    except RuntimeError:

        raise

    except Exception as error:

        fail(
            "DETAILER conversion failed:\n"
            f"{type(error).__name__}: {error}"
        )

    finally:

        sys.path[:] = previous_path

    print(
        "OK   DETAILER graph → API conversion"
    )

    print(
        "OK   legacy latent loader removed"
    )

    print(
        "OK   modern latent loader produced"
    )


# ======================================================================
# 8. COMPATIBILITY BUILDER
# ======================================================================

def validate_compatibility_builder() -> None:

    source = read_text(
        COMPATIBILITY_BUILDER
    )

    required = {
        "compatibility_lock.yaml",
        "legacy_ltx_098_compat",
        "runtime_package",
        "LTX098ModernCompat",
        "blur_internal",
    }

    for text in required:

        require(
            text in source,
            "Compatibility builder missing contract:\n"
            f"{text}",
        )

    print(
        "OK   compatibility builder contract"
    )


# ======================================================================
# 9. PRODUCTION APPLICATION
# ======================================================================

def validate_production_application() -> None:

    source = read_text(
        GENERATE_VIDEO
    )

    required = {
        "ProductionRunner",
        "ProductionOrchestrator",
        "ProductionManager",
        "BASE_WORKFLOW",
        "DETAILER_WORKFLOW",
        "gpu-url",
    }

    for text in required:

        require(
            text in source,
            "generate_video.py missing contract:\n"
            f"{text}",
        )

    require(
        str(STALE_COMFYUI_PORT)
        not in source,
        "generate_video.py contains stale "
        "ComfyUI port 8219.",
    )

    require(
        str(CANONICAL_COMFYUI_PORT)
        in source,
        "generate_video.py does not declare "
        "canonical ComfyUI port 8188.",
    )

    print(
        "OK   production application wiring"
    )


# ======================================================================
# 10. KAGGLE WIRING
# ======================================================================

def validate_kaggle_wiring() -> None:

    launch = read_text(
        LAUNCH
    )

    bootstrap = read_text(
        BOOTSTRAP
    )

    preflight = read_text(
        PREFLIGHT_MODERN
    )

    tunnel = read_text(
        TUNNEL
    )

    require(
        "preflight_modern.py"
        in launch,
        "launch.py is not wired to "
        "preflight_modern.py.",
    )

    require(
        "bootstrap.py"
        in launch,
        "launch.py is not wired to "
        "bootstrap.py.",
    )

    require(
        "start_comfyui_tunnel.py"
        in launch,
        "launch.py is not wired to "
        "start_comfyui_tunnel.py.",
    )

    require(
        "compatibility_lock.yaml"
        in bootstrap,
        "bootstra
