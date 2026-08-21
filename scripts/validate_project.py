#!/usr/bin/env python3

"""
LTX-13B COMPLETE REPOSITORY + RUNTIME VALIDATION

Purpose
-------
Validate the complete LTX-13B project before production generation.

This validator checks:

1. Repository file structure
2. Python AST syntax
3. compatibility_lock.yaml structure
4. Lock-driven bootstrap architecture
5. BASE workflow structure
6. DETAILER source workflow structure
7. Legacy -> modern DETAILER adapter conversion
8. Compatibility builder contract
9. Production application wiring
10. Kaggle startup wiring
11. CPU preflight contract
12. Live-runtime verifier contract
13. Canonical ComfyUI port contract
14. Runtime/network contract
15. Optional live ComfyUI runtime
16. Optional CUDA availability

Important
---------
This file does NOT:

- install packages
- modify the environment
- start ComfyUI
- generate video
- download models
- require CUDA unless --require-cuda is supplied

Single source of truth:

    kaggle/compatibility_lock.yaml

Canonical local ComfyUI port:

    8188

Known stale local ComfyUI port:

    8219

Multi-GPU port strategy:

    base_port + GPU ID

Therefore:

    GPU 0 -> 8188
    GPU 1 -> 8189
    GPU 2 -> 8190

The source DETAILER workflow intentionally contains the legacy:

    LTXVLatentUpsamplerModelLoader

The runtime/API workflow must contain:

    LatentUpscaleModelLoader

That conversion is performed by:

    execution/comfy_workflow_adapter.py
"""

from __future__ import annotations

import argparse
import ast
import importlib
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


# ============================================================================
# PROJECT PATHS
# ============================================================================

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

EXECUTION_DIR = (
    PROJECT_ROOT
    / "execution"
)

COMPATIBILITY_DIR = (
    PROJECT_ROOT
    / "compatibility"
)


# ============================================================================
# IMPORTANT FILES
# ============================================================================

LOCK_FILE = (
    KAGGLE_DIR
    / "compatibility_lock.yaml"
)

BOOTSTRAP = (
    KAGGLE_DIR
    / "bootstrap.py"
)

KAGGLE_CONFIG = (
    KAGGLE_DIR
    / "config.py"
)

LAUNCH = (
    KAGGLE_DIR
    / "launch.py"
)

PREFLIGHT_MODERN = (
    KAGGLE_DIR
    / "preflight_modern.py"
)

START_COMFYUI = (
    KAGGLE_DIR
    / "start_comfyui.py"
)

LIVE_RUNTIME = (
    KAGGLE_DIR
    / "verify_live_runtime.py"
)

CPU_PREFLIGHT = (
    SCRIPTS_DIR
    / "cpu_preflight.py"
)

GENERATE_VIDEO = (
    SCRIPTS_DIR
    / "generate_video.py"
)

VALIDATOR = (
    SCRIPTS_DIR
    / "validate_project.py"
)

SHOT_EXECUTOR = (
    EXECUTION_DIR
    / "shot_executor.py"
)

REFERENCE_IMAGE_GENERATOR = (
    EXECUTION_DIR
    / "reference_image_generator.py"
)

CHARACTER_REFERENCE_PROCESSOR = (
    EXECUTION_DIR
    / "character_reference_processor.py"
)

CHARACTER_SCHEMA = (
    PROJECT_ROOT
    / "schemas"
    / "character.py"
)

SHOT_SCHEMA = (
    PROJECT_ROOT
    / "schemas"
    / "shot.py"
)

ADAPTER = (
    EXECUTION_DIR
    / "comfy_workflow_adapter.py"
)

COMPATIBILITY_BUILDER = (
    COMPATIBILITY_DIR
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


# ============================================================================
# CONSTANTS
# ============================================================================

CANONICAL_COMFYUI_PORT = 8188

STALE_COMFYUI_PORT = 8219

CANONICAL_COMFYUI_HOST = "127.0.0.1"

COMFYUI_PORT_STRATEGY = (
    "base_port_plus_gpu_id"
)

COMFYUI_PROTOCOL = "http"

LEGACY_LATENT_LOADER = (
    "LTXVLatentUpsamplerModelLoader"
)

MODERN_LATENT_LOADER = (
    "LatentUpscaleModelLoader"
)


# ============================================================================
# REQUIRED PROJECT FILES
# ============================================================================

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
    "execution/character_reference_processor.py",
    "execution/checkpoint_manager.py",
    "execution/comfy_client.py",
    "execution/comfy_workflow_adapter.py",
    "execution/reference_image_generator.py",
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


# ============================================================================
# ERROR / ASSERTION HELPERS
# ============================================================================

def fail(message: str) -> None:

    raise RuntimeError(
        message
    )


def require(
    condition: bool,
    message: str,
) -> None:

    if not condition:

        fail(
            message
        )


# ============================================================================
# FILE HELPERS
# ============================================================================

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
            "Could not read file:\n"
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

        value = json.loads(
            read_text(path)
        )

    except json.JSONDecodeError as error:

        fail(
            "Invalid JSON:\n"
            f"{path}\n"
            f"{error}"
        )

    require(
        isinstance(value, dict),
        "JSON root must be an object:\n"
        f"{path}",
    )

    return value

def validate_identity_reference_contract() -> None:

    processor = read_text(
        CHARACTER_REFERENCE_PROCESSOR
    )

    character_schema = read_text(
        CHARACTER_SCHEMA
    )

    shot_schema = read_text(
        SHOT_SCHEMA
    )

    adapter = read_text(
        ADAPTER
    )

    executor = read_text(
        SHOT_EXECUTOR
    )

    manager = read_text(
        PROJECT_ROOT
        / "pipeline"
        / "reference_manager.py"
    )

    planner = read_text(
        PROJECT_ROOT
        / "planner"
        / "character_planner.py"
    )

    shot_planner = read_text(
        PROJECT_ROOT
        / "planner"
        / "shot_planner.py"
    )

    # Processor

    for text in (
        "CharacterReferenceProcessor",
        "identity_reference.png",
        "identity_mask.png",
        "mask_path",
        "768",
        "432",
    ):

        require(
            text in processor,
            "Character reference processor "
            f"is missing contract: {text}",
        )

    # Character schema

    require(
        "reference_mask_path"
        in character_schema,
        "schemas/character.py must contain "
        "reference_mask_path.",
    )

    # Shot schema

    require(
        "reference_masks"
        in shot_schema,
        "schemas/shot.py must contain "
        "reference_masks.",
    )

    # Reference manager

    require(
        "mask_path"
        in manager,
        "ReferenceManager must propagate "
        "identity mask paths.",
    )

    # Character planner

    require(
        "reference_mask_path"
        in planner,
        "CharacterPlanner must propagate "
        "reference_mask_path.",
    )

    # Shot planner

    require(
        "reference_masks"
        in shot_planner,
        "ShotPlanner must propagate "
        "reference_masks.",
    )

    # Adapter

    for text in (
        "set_identity_reference_image",
        "set_identity_mask_image",
    ):

        require(
            text in adapter,
            "Workflow adapter missing "
            f"{text}.",
        )

    # Executor

    for text in (
        "reference_masks",
        "_copy_reference_mask",
        "_compose_reference_masks",
    ):

        require(
            text in executor,
            "ShotExecutor missing "
            f"{text}.",
        )

    print(
        "OK   identity reference contract"
    )


# ============================================================================
# PYTHON FILE DISCOVERY
# ============================================================================

def all_python_files() -> list[Path]:

    result: list[Path] = []

    for path in PROJECT_ROOT.rglob(
        "*.py"
    ):

        relative = (
            path.relative_to(
                PROJECT_ROOT
            )
        )

        excluded_parts = {
            ".git",
            "ComfyUI",
            ".runtime_ltx098",
            "__pycache__",
        }

        if any(
            part in excluded_parts
            for part in relative.parts
        ):

            continue

        result.append(
            path
        )

    return sorted(
        result
    )


# ============================================================================
# WORKFLOW HELPERS
# ============================================================================

def node_types(
    workflow: dict[str, Any],
) -> set[str]:

    nodes = workflow.get(
        "nodes",
        [],
    )

    require(
        isinstance(nodes, list),
        "Workflow 'nodes' must be a list.",
    )

    return {
        node.get("type")
        for node in nodes
        if isinstance(node, dict)
        and node.get("type")
    }


# ============================================================================
# LOCK FILE
# ============================================================================

def load_lock() -> dict[str, Any]:

    try:

        import yaml

    except ImportError as error:

        fail(
            "PyYAML is required:\n"
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
        "compatibility_lock.yaml "
        "must contain a mapping.",
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
        "runtime",
    }

    missing = (
        required_sections
        - set(lock)
    )

    require(
        not missing,
        "compatibility_lock.yaml "
        "missing sections:\n"
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

    custom_nodes = (
        lock["custom_nodes"]
    )

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

    require(
        isinstance(
            custom_nodes,
            dict,
        ),
        "lock.custom_nodes must be a mapping.",
    )

    # ------------------------------------------------------------------------
    # ComfyUI commit
    # ------------------------------------------------------------------------

    commit = str(
        comfy.get(
            "commit",
            "",
        )
    )

    require(
        len(commit) == 40
        and all(
            character
            in "0123456789abcdef"
            for character in commit.lower()
        ),
        "ComfyUI commit must be "
        "a 40-character hexadecimal SHA.",
    )

    # ------------------------------------------------------------------------
    # Required model entries
    # ------------------------------------------------------------------------

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
        "compatibility_lock.yaml "
        "missing models:\n"
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
            f"models.{name} "
            "must be a mapping.",
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
                isinstance(
                    value,
                    str,
                )
                and value.strip(),
                f"models.{name}."
                f"{field} is missing.",
            )

    # ------------------------------------------------------------------------
    # Python runtime section
    # ------------------------------------------------------------------------

    runtime = lock[
        "python_runtime"
    ]

    require(
        isinstance(
            runtime,
            dict,
        ),
        "lock.python_runtime "
        "must be a mapping.",
    )

    for field in (
        "torch",
        "torchvision",
        "torchsde",
        "spandrel",
        "av",
        "gguf",
    ):

        require(
            isinstance(
                runtime.get(field),
                str,
            )
            and runtime[field].strip(),
            "lock.python_runtime."
            f"{field} is missing.",
        )

    # ------------------------------------------------------------------------
    # Runtime/network contract
    # ------------------------------------------------------------------------

    runtime_contract = lock[
        "runtime"
    ]

    require(
        isinstance(
            runtime_contract,
            dict,
        ),
        "lock.runtime must be a mapping.",
    )

    network = runtime_contract.get(
        "network"
    )

    require(
        isinstance(
            network,
            dict,
        ),
        "lock.runtime.network "
        "must be a mapping.",
    )

    require(
        network.get("host")
        == CANONICAL_COMFYUI_HOST,
        "lock.runtime.network.host "
        "must be "
        f"{CANONICAL_COMFYUI_HOST}.",
    )

    require(
        network.get("base_port")
        == CANONICAL_COMFYUI_PORT,
        "lock.runtime.network.base_port "
        "must be "
        f"{CANONICAL_COMFYUI_PORT}.",
    )

    require(
        network.get("port_strategy")
        == COMFYUI_PORT_STRATEGY,
        "lock.runtime.network."
        "port_strategy must be "
        f"{COMFYUI_PORT_STRATEGY}.",
    )

    require(
        network.get("protocol")
        == COMFYUI_PROTOCOL,
        "lock.runtime.network.protocol "
        "must be "
        f"{COMFYUI_PROTOCOL}.",
    )

    stale_ports = network.get(
        "stale_ports"
    )

    require(
        isinstance(
            stale_ports,
            list,
        ),
        "lock.runtime.network."
        "stale_ports must be a list.",
    )

    require(
        STALE_COMFYUI_PORT
        in stale_ports,
        "lock.runtime.network."
        "stale_ports must contain "
        f"{STALE_COMFYUI_PORT}.",
    )

    print(
        "OK   compatibility lock"
    )

    print(
        "OK   runtime/network contract"
    )


# ============================================================================
# FILE STRUCTURE
# ============================================================================

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

    print(
        "OK   repository file structure"
    )


# ============================================================================
# PYTHON SYNTAX
# ============================================================================

def validate_python() -> None:

    files = all_python_files()

    require(
        bool(files),
        "No Python files found.",
    )

    for path in files:

        parse_python(
            path
        )

    print(
        "OK   Python AST syntax: "
        f"{len(files)} files"
    )


# ============================================================================
# BOOTSTRAP ARCHITECTURE
# ============================================================================

def validate_model_architecture() -> None:

    bootstrap = read_text(
        BOOTSTRAP
    )

    preflight = read_text(
        PREFLIGHT_MODERN
    )

    cpu_preflight = read_text(
        CPU_PREFLIGHT
    )

    # ------------------------------------------------------------------------
    # Bootstrap must use lock
    # ------------------------------------------------------------------------

    require(
        "compatibility_lock.yaml"
        in bootstrap,
        "bootstrap.py is not "
        "lock-driven.",
    )

    require(
        "compatibility_lock.yaml"
        in preflight,
        "preflight_modern.py is not "
        "lock-driven.",
    )

    # ------------------------------------------------------------------------
    # CPU preflight must use lock
    # ------------------------------------------------------------------------

    require(
        "compatibility_lock.yaml"
        in cpu_preflight,
        "cpu_preflight.py is not "
        "lock-driven.",
    )

    print(
        "OK   single-source model architecture"
    )


# ============================================================================
# WORKFLOWS
# ============================================================================

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

    require(
       "LTXICLoRALoaderModelOnly"
       in base_types,
       "BASE workflow is missing "
       "LTXICLoRALoaderModelOnly.",
    )

    require(
       "LTXAddVideoICLoRAGuideAdvanced"
       in base_types,
       "BASE workflow is missing "
       "LTXAddVideoICLoRAGuideAdvanced.",
   )

   require(
       "LoadImage"
       in base_types,
       "BASE workflow is missing "
       "LoadImage nodes.",
   )

    # ------------------------------------------------------------------------
    # BASE workflow
    # ------------------------------------------------------------------------

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

    # ------------------------------------------------------------------------
    # DETAILER source workflow
    # ------------------------------------------------------------------------

    detailer_required = {
        "VHS_LoadVideo",
        "LTXVConditioning",
        "STGGuiderAdvanced",
        "FloatToSigmas",
        "StringToFloatList",
        "LTXVLoopingSampler",
        "LTXVLatentUpsampler",
        LEGACY_LATENT_LOADER,
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
        "DETAILER source workflow "
        "missing nodes:\n"
        + "\n".join(
            sorted(missing)
        ),
    )

    # The source must remain legacy.
    # The adapter is responsible for conversion.

    require(
        MODERN_LATENT_LOADER
        not in detailer_types,
        "DETAILER source workflow "
        "already contains modern "
        "LatentUpscaleModelLoader.\n"
        "The source workflow must retain "
        "the legacy loader so the adapter "
        "can perform the conversion.",
    )

    print(
        "OK   BASE workflow"
    )

    print(
        "OK   DETAILER source workflow"
    )


# ============================================================================
# WORKFLOW ADAPTER
# ============================================================================

def validate_adapter() -> None:

    module = parse_python(
        ADAPTER
    )

    classes = {
        node.name
        for node in ast.walk(module)
        if isinstance(
            node,
            ast.ClassDef,
        )
    }

    require(
        "ComfyWorkflowAdapter"
        in classes,
        "ComfyWorkflowAdapter "
        "class missing.",
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

    for text in (
        required_strings
    ):

        require(
            text in source,
            "ComfyWorkflowAdapter "
            "missing contract:\n"
            f"{text}",
        )

    print(
        "OK   ComfyWorkflowAdapter contract"
    )


# ============================================================================
# REAL DETAILER CONVERSION
# ============================================================================

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
            "ComfyWorkflowAdapter "
            "import failed.",
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
            node.get(
                "class_type"
            )
            for node in api_workflow.values()
            if isinstance(
                node,
                dict,
            )
        }

        require(
            MODERN_LATENT_LOADER
            in api_types,
            "DETAILER conversion did not "
            "produce LatentUpscaleModelLoader.",
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
            f"{type(error).__name__}: "
            f"{error}"
        )

    finally:

        sys.path[:] = previous_path

    print(
        "OK   DETAILER graph -> API conversion"
    )

    print(
        "OK   legacy latent loader removed"
    )

    print(
        "OK   modern latent loader produced"
    )



# ============================================================================
# 10. PLANNING CONTRACTS
# ============================================================================

def validate_planning_contracts() -> None:

    config_source = read_text(
        PROJECT_ROOT
        / "planner"
        / "config.py"
    )

    character_schema = read_text(
        PROJECT_ROOT
        / "schemas"
        / "character.py"
    )

    scene_schema = read_text(
        PROJECT_ROOT
        / "schemas"
        / "scene.py"
    )

    character_planner = read_text(
        PROJECT_ROOT
        / "planner"
        / "character_planner.py"
    )

    scene_planner = read_text(
        PROJECT_ROOT
        / "planner"
        / "scene_planner.py"
    )

    shot_planner = read_text(
        PROJECT_ROOT
        / "planner"
        / "shot_planner.py"
    )

    character_prompt = read_text(
        PROJECT_ROOT
        / "prompts"
        / "qwen"
        / "character_plan.txt"
    )

    scene_prompt = read_text(
        PROJECT_ROOT
        / "prompts"
        / "qwen"
        / "scene_plan.txt"
    )

    required_config = {
        "QWEN_STORY_TEMPERATURE",
        "QWEN_CHARACTER_DETECTION_TEMPERATURE",
        "QWEN_CHARACTER_PLAN_TEMPERATURE",
        "QWEN_SCENE_PLAN_TEMPERATURE",
        "QWEN_SHOT_PLAN_TEMPERATURE",
    }

    for text in required_config:

        require(
            text in config_source,
            "planner/config.py missing "
            "Qwen planning contract:\n"
            f"{text}",
        )

    required_character_schema = {
        "character_state",
        "reference_mode",
        "reference_path",
        "continuity_rules",
    }

    for text in required_character_schema:

        require(
            text in character_schema,
            "schemas/character.py missing "
            "character contract:\n"
            f"{text}",
        )

    required_scene_schema = {
        "weather",
        "atmosphere",
        "environment_details",
        "key_props",
        "scene_objective",
        "mood",
        "lighting",
    }

    for text in required_scene_schema:

        require(
            text in scene_schema,
            "schemas/scene.py missing "
            "scene contract:\n"
            f"{text}",
        )

    require(
        "character_state" in character_planner,
        "character_planner.py does not "
        "propagate character_state.",
    )

    required_scene_planner = {
        "weather",
        "atmosphere",
        "environment_details",
        "key_props",
        "scene_objective",
    }

    for text in required_scene_planner:

        require(
            text in scene_planner,
            "scene_planner.py missing "
            "scene field:\n"
            f"{text}",
        )

    require(
        "QWEN_SHOT_PLAN_TEMPERATURE"
        in shot_planner,
        "shot_planner.py does not use "
        "QWEN_SHOT_PLAN_TEMPERATURE.",
    )

    required_character_prompt = {
        "character_state",
        "emotional_state",
        "physical_state",
        "clothing_state",
        "carried_objects",
        "injuries",
    }

    for text in required_character_prompt:

        require(
            text in character_prompt,
            "character_plan.txt missing "
            "character-state contract:\n"
            f"{text}",
        )

    required_scene_prompt = {
        "weather",
        "atmosphere",
        "environment_details",
        "key_props",
        "scene_objective",
        "mood",
        "lighting",
    }

    for text in required_scene_prompt:

        require(
            text in scene_prompt,
            "scene_plan.txt missing "
            "scene contract:\n"
            f"{text}",
        )

    print(
        "OK   planning/schema contract"
    )


# ============================================================================
# COMPATIBILITY BUILDER
# ============================================================================

def validate_compatibility_builder() -> None:

    source = read_text(
        COMPATIBILITY_BUILDER
    )

    required = {
        "compatibility_lock.yaml",
        "legacy_ltx_098_compat",
        "runtime_package",
        "blur_internal",
        "native torch",
    }

    for text in (
        required
    ):

        require(
            text in source,
            "Compatibility builder "
            "missing contract:\n"
            f"{text}",
        )

    print(
        "OK   compatibility builder contract"
    )


# ============================================================================
# PRODUCTION APPLICATION
# ============================================================================

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

    for text in (
        required
    ):

        require(
            text in source,
            "generate_video.py "
            "missing contract:\n"
            f"{text}",
        )

    require(
        str(
            CANONICAL_COMFYUI_PORT
        )
        in source,
        "generate_video.py does not "
        "declare canonical ComfyUI "
        "port 8188.",
    )

    # 8219 is intentionally allowed ONLY
    # as a stale endpoint rejection guard.

    require(
        str(
            STALE_COMFYUI_PORT
        )
        in source,
        "generate_video.py no longer "
        "contains the stale-port "
        "rejection guard.",
    )

    tree = parse_python(
        GENERATE_VIDEO
    )

    stale_literals = [
        node
        for node in ast.walk(tree)
        if isinstance(
            node,
            ast.Constant,
        )
        and isinstance(
            node.value,
            str,
        )
        and str(
            STALE_COMFYUI_PORT
        )
        in node.value
    ]

    require(
        stale_literals,
        "generate_video.py stale-port "
        "guard is not represented "
        "as a string literal.",
    )

    for node in stale_literals:

        value = node.value

        require(
            (
                "127.0.0.1:8219"
                in value
                or
                "localhost:8219"
                in value
            ),
            "Unexpected 8219 reference "
            "in generate_video.py.",
        )

    print(
        "OK   production application wiring"
    )

    print(
        "OK   stale ComfyUI endpoint "
        "rejection guard"
    )


# ============================================================================
# KAGGLE STARTUP WIRING
# ============================================================================

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

    start_comfyui = read_text(
        START_COMFYUI
    )

    # ------------------------------------------------------------------------
    # launch.py
    #
    # launch.py is the startup orchestrator:
    #
    #     preflight_modern.py
    #             ↓
    #         bootstrap.py
    #             ↓
    #     start_comfyui.py
    #
    # ComfyUI runs locally on the canonical backend port 8188.
    # No Cloudflare tunnel is required.
    # ------------------------------------------------------------------------

    for name in (
        "preflight_modern.py",
        "bootstrap.py",
        "start_comfyui.py",
    ):

        require(
            name in launch,
            "launch.py is not wired to:\n"
            f"{name}",
        )

    # ------------------------------------------------------------------------
    # start_comfyui.py
    # ------------------------------------------------------------------------

    require(
        str(
            CANONICAL_COMFYUI_PORT
        )
        in start_comfyui,
        "start_comfyui.py does not "
        "use canonical ComfyUI port 8188.",
    )

    require(
        str(
            STALE_COMFYUI_PORT
        )
        not in start_comfyui,
        "start_comfyui.py contains "
        "stale port 8219.",
    )

    # ------------------------------------------------------------------------
    # bootstrap.py
    # ------------------------------------------------------------------------

    require(
        "compatibility_lock.yaml"
        in bootstrap,
        "bootstrap.py is not "
        "lock-driven.",
    )

    # ------------------------------------------------------------------------
    # modern preflight
    # ------------------------------------------------------------------------

    require(
        "compatibility_lock.yaml"
        in preflight,
        "preflight_modern.py is not "
        "lock-driven.",
    )

    print(
        "OK   Kaggle launcher/bootstrap wiring"
    )

    print(
        "OK   start_comfyui.py canonical port 8188"
    )

    print(
        "OK   launch.py -> preflight/bootstrap/start_comfyui"
    )


# ============================================================================
# LIVE RUNTIME VERIFIER CONTRACT
# ============================================================================

def validate_live_runtime_verifier() -> None:

    source = read_text(
        LIVE_RUNTIME
    )

    required = {
        "127.0.0.1:8188",
        "/system_stats",
        "/object_info",

        "LTXVBaseSampler",
        "LTXVConditioning",
        "STGGuiderAdvanced",
        "FloatToSigmas",
        "StringToFloatList",

        "UnetLoaderGGUF",
        "CLIPLoaderGGUF",
        "VAELoader",

        "VHS_VideoCombine",
        "VHS_LoadVideo",

        "LTXVLoopingSampler",
        "LTXVLatentUpsampler",
        LEGACY_LATENT_LOADER,
        "LTXVTiledVAEDecode",
        "LTXVFilmGrain",
        "LoraLoaderModelOnly",

        MODERN_LATENT_LOADER,
    }

    for text in (
        required
    ):

        require(
            text in source,
            "verify_live_runtime.py "
            "missing contract:\n"
            f"{text}",
        )

    require(
        "8219"
        not in source,
        "verify_live_runtime.py still "
        "contains stale port 8219.",
    )

    # ------------------------------------------------------------------------
    # Strong model verification
    # ------------------------------------------------------------------------

    model_checks = {
        'models["ltx_q4"]["filename"]',
        'models["t5_q4"]["filename"]',
        'models["vae"]["filename"]',
        'models["ic_lora"]["filename"]',
        'models["spatial_upscaler"]["filename"]',
    }

    for text in (
        model_checks
    ):

        require(
            text in source,
            "verify_live_runtime.py does "
            "not verify locked model:\n"
            f"{text}",
        )

    require(
        "LatentUpscaleModelLoader"
        in source,
        "Live verifier does not "
        "verify modern latent upscaler.",
    )

    print(
        "OK   live runtime verifier contract"
    )

    print(
        "OK   live locked-model verification contract"
    )


# ============================================================================
# CPU PREFLIGHT CONTRACT
# ============================================================================

def validate_cpu_preflight() -> None:

    source = read_text(
        CPU_PREFLIGHT
    )

    # IMPORTANT:
    #
    # cpu_preflight.py is responsible for:
    #
    # 1. Using compatibility_lock.yaml
    # 2. Understanding the legacy DETAILER loader
    # 3. Understanding the modern DETAILER loader
    # 4. Understanding the canonical/stale ComfyUI port policy
    #
    # The actual adapter implementation contracts
    # "apply_modern_compatibility" and
    # "validate_modern_detailer" belong to
    # execution/comfy_workflow_adapter.py.
    #
    # They are therefore validated by:
    #
    #     validate_adapter()
    #     validate_real_detailer_conversion()
    #
    # They must NOT be required literally inside
    # cpu_preflight.py.

    required = {
        "compatibility_lock.yaml",
        "LTXVLatentUpsamplerModelLoader",
        "LatentUpscaleModelLoader",
    }

    for text in (
        required
    ):

        require(
            text in source,
            "cpu_preflight.py missing "
            "contract:\n"
            f"{text}",
        )

    # ------------------------------------------------------------------------
    # Port policy
    #
    # 8188 = canonical active local ComfyUI port.
    #
    # 8219 = known stale port that generate_video.py must reject.
    # ------------------------------------------------------------------------

    require(
        str(
            CANONICAL_COMFYUI_PORT
        )
        in source,
        "cpu_preflight.py does not "
        "know canonical ComfyUI port 8188.",
    )

    require(
        str(
            STALE_COMFYUI_PORT
        )
        in source,
        "cpu_preflight.py does not "
        "know stale ComfyUI port 8219.",
    )

    print(
        "OK   CPU preflight contract"
    )

    print(
        "OK   CPU preflight port-policy contract"
    )


# ============================================================================
# GLOBAL COMFYUI PORT CONTRACT
# ============================================================================

def validate_port_contract() -> None:

    # ------------------------------------------------------------------------
    # ComfyUI launcher
    # ------------------------------------------------------------------------

    start_comfyui = read_text(
        START_COMFYUI
    )

    require(
        str(
            CANONICAL_COMFYUI_PORT
        )
        in start_comfyui,
        "Canonical ComfyUI port 8188 "
        "missing from start_comfyui.py.",
    )

    require(
        str(
            STALE_COMFYUI_PORT
        )
        not in start_comfyui,
        "Stale ComfyUI port detected "
        "in start_comfyui.py.",
    )

    # ------------------------------------------------------------------------
    # Live verifier
    # ------------------------------------------------------------------------

    live = read_text(
        LIVE_RUNTIME
    )

    require(
        str(
            STALE_COMFYUI_PORT
        )
        not in live,
        "Stale ComfyUI port detected "
        "in verify_live_runtime.py.",
    )

    require(
        str(
            CANONICAL_COMFYUI_PORT
        )
        in live,
        "Canonical ComfyUI port 8188 "
        "missing from live verifier.",
    )

    # ------------------------------------------------------------------------
    # Generator
    # ------------------------------------------------------------------------

    generator = read_text(
        GENERATE_VIDEO
    )

    require(
        str(
            CANONICAL_COMFYUI_PORT
        )
        in generator,
        "Canonical ComfyUI port 8188 "
        "missing from generate_video.py.",
    )

    print(
        "OK   global ComfyUI port contract: 8188"
    )


# ============================================================================
# HTTP JSON
# ============================================================================

def http_json(
    url: str,
    timeout: float,
) -> dict[str, Any]:

    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
        },
    )

    try:

        with urllib.request.urlopen(
            request,
            timeout=timeout,
        ) as response:

            payload = response.read()

    except (
        urllib.error.URLError,
        urllib.error.HTTPError,
    ) as error:

        fail(
            "Live ComfyUI request failed:\n"
            f"{url}\n"
            f"{error}"
        )

    try:

        data = json.loads(
            payload.decode(
                "utf-8"
            )
        )

    except json.JSONDecodeError as error:

        fail(
            "Live ComfyUI returned "
            "invalid JSON:\n"
            f"{url}\n"
            f"{error}"
        )

    require(
        isinstance(
            data,
            dict,
        ),
        "Live response is not an object:\n"
        f"{url}",
    )

    return data


# ============================================================================
# LIVE COMFYUI HELPERS
# ============================================================================

def get_live_choices(
    object_info: dict[str, Any],
    node_name: str,
    input_name: str,
) -> list[str]:

    node = object_info.get(
        node_name
    )

    require(
        isinstance(
            node,
            dict,
        ),
        "Live ComfyUI node is missing:\n"
        f"{node_name}",
    )

    input_data = node.get(
        "input",
        {},
    )

    require(
        isinstance(
            input_data,
            dict,
        ),
        "Invalid input schema for:\n"
        f"{node_name}",
    )

    required = input_data.get(
        "required",
        {},
    )

    require(
        isinstance(
            required,
            dict,
        ),
        "Invalid required-input schema "
        f"for {node_name}.",
    )

    spec = required.get(
        input_name
    )

    require(
        isinstance(
            spec,
            list,
        )
        and bool(spec),
        "Live schema missing:\n"
        f"{node_name}.{input_name}",
    )

    choices = spec[0]

    require(
        isinstance(
            choices,
            list,
        ),
        "Live schema does not expose "
        "choices:\n"
        f"{node_name}.{input_name}",
    )

    return [
        str(value)
        for value in choices
    ]


def require_live_choice(
    object_info: dict[str, Any],
    node_name: str,
    input_name: str,
    expected: str,
) -> None:

    available = get_live_choices(
        object_info,
        node_name,
        input_name,
    )

    require(
        expected in available,
        "Locked model is not "
        "discoverable in live ComfyUI:\n"
        f"Node:     {node_name}\n"
        f"Input:    {input_name}\n"
        f"Expected: {expected}\n"
        f"Available: {available}",
    )

    print(
        f"OK   {node_name}."
        f"{input_name}: {expected}"
    )


# ============================================================================
# LIVE COMFYUI VALIDATION
# ============================================================================

def validate_live_runtime(
    url: str,
) -> None:

    print()
    print(
        "=" * 80
    )

    print(
        "LIVE COMFYUI VALIDATION"
    )

    print(
        "=" * 80
    )

    print(
        f"URL: {url}"
    )

    # ------------------------------------------------------------------------
    # HTTP readiness
    # ------------------------------------------------------------------------

    http_json(
        f"{url}/system_stats",
        15,
    )

    print(
        "OK   /system_stats"
    )

    # ------------------------------------------------------------------------
    # Object info
    # ------------------------------------------------------------------------

    object_info = http_json(
        f"{url}/object_info",
        30,
    )

    print(
        "OK   /object_info"
    )

    # ------------------------------------------------------------------------
    # Required node registration
    # ------------------------------------------------------------------------

    required_nodes = {
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
        "VHS_LoadVideo",
        "LTXVLoopingSampler",
        "LTXVLatentUpsampler",
        LEGACY_LATENT_LOADER,
        "LTXVTiledVAEDecode",
        "LTXVFilmGrain",
        "LoraLoaderModelOnly",
        MODERN_LATENT_LOADER,
    }

    available_nodes = set(
        object_info
    )

    missing = (
        required_nodes
        - available_nodes
    )

    require(
        not missing,
        "Live ComfyUI is missing "
        "required nodes:\n"
        + "\n".join(
            sorted(missing)
        ),
    )

    print(
        "OK   live required nodes: "
        f"{len(required_nodes)}"
    )

    # ------------------------------------------------------------------------
    # Lock
    # ------------------------------------------------------------------------

    lock = load_lock()

    models = lock[
        "models"
    ]

    # ------------------------------------------------------------------------
    # Exact model discovery
    # ------------------------------------------------------------------------

    require_live_choice(
        object_info,
        "UnetLoaderGGUF",
        "unet_name",
        models["ltx_q4"]["filename"],
    )

    require_live_choice(
        object_info,
        "CLIPLoaderGGUF",
        "clip_name",
        models["t5_q4"]["filename"],
    )

    require_live_choice(
        object_info,
        "VAELoader",
        "vae_name",
        models["vae"]["filename"],
    )

    require_live_choice(
        object_info,
        "LoraLoaderModelOnly",
        "lora_name",
        models["ic_lora"]["filename"],
    )

    require_live_choice(
        object_info,
        MODERN_LATENT_LOADER,
        "model_name",
        models["spatial_upscaler"]["filename"],
    )

    print(
        "OK   locked models are "
        "discoverable"
    )

    print(
        "OK   modern latent upscaler "
        "is discoverable"
    )

    print(
        "OK   legacy compatibility node "
        "is registered"
    )


# ============================================================================
# CUDA
# ============================================================================

def validate_cuda() -> None:

    try:

        import torch

    except ImportError as error:

        fail(
            "PyTorch is not importable:\n"
            f"{error}"
        )

    require(
        torch.cuda.is_available(),
        "CUDA is not available.",
    )

    count = (
        torch.cuda.device_count()
    )

    require(
        count > 0,
        "CUDA reports zero devices.",
    )

    print(
        f"OK   CUDA available: "
        f"{count} GPU(s)"
    )

    for index in range(
        count
    ):

        print(
            "     GPU "
            f"{index}: "
            f"{torch.cuda.get_device_name(index)}"
        )


# ============================================================================
# OPTIONAL CROSS-CHECK
# ============================================================================

def validate_live_verifier_matches_runtime_contract() -> None:

    source = read_text(
        LIVE_RUNTIME
    )

    # Make sure the live verifier and this validator
    # agree on the canonical endpoint.

    require(
        'DEFAULT_URL = "http://127.0.0.1:8188"'
        in source,
        "verify_live_runtime.py does not "
        "declare the canonical default "
        "endpoint http://127.0.0.1:8188.",
    )

    # Make sure the live verifier checks the
    # modern loader rather than only the legacy one.

    require(
        MODERN_LATENT_LOADER
        in source,
        "Live verifier does not contain "
        "modern latent loader contract.",
    )

    # Make sure the live verifier performs
    # exact locked model checks.

    require(
        "require_choice("
        in source,
        "Live verifier does not perform "
        "exact locked model discovery checks.",
    )

    print(
        "OK   validator/live-verifier "
        "contract alignment"
    )


def validate_reference_image_generator() -> None:

    source = read_text(
        REFERENCE_IMAGE_GENERATOR
    )

    required = {
        "ReferenceImageGenerator",
        "CheckpointLoaderSimple",
        "SaveImage",
        "data/characters/generated",
        "get_object_info",
        "find_image_outputs",
        "character_state",
    }

    for text in required:

        require(
            text in source,
            "reference_image_generator.py "
            "missing contract:\n"
            f"{text}",
        )

    print(
        "OK   character reference generator contract"
    )

def validate_multi_reference_execution() -> None:

    source = read_text(
        SHOT_EXECUTOR
    )

    required = {
        "shot.reference_images",
        "_prepare_shot_reference",
        "_compose_reference_images",
        "_grid_dimensions",
        "dynamic composite",
        "Image.open",
        "Image.new",
    }

    for text in required:

        require(
            text in source,
            "shot_executor.py is missing "
            "multi-reference execution contract:\n"
            f"{text}",
        )

    tree = parse_python(
        SHOT_EXECUTOR
    )

    for node in ast.walk(
        tree
    ):

        if not isinstance(
            node,
            ast.Subscript,
        ):
            continue

        value = node.value

        is_reference_sequence = (
            (
                isinstance(
                    value,
                    ast.Name,
                )
                and value.id
                in {
                    "references",
                    "reference_images",
                }
            )
            or
            (
                isinstance(
                    value,
                    ast.Attribute,
                )
                and value.attr
                == "reference_images"
            )
        )

        if not is_reference_sequence:
            continue

        slice_node = node.slice

        if isinstance(
            slice_node,
            ast.Slice,
        ):

            upper = slice_node.upper

            if (
                isinstance(
                    upper,
                    ast.Constant,
                )
                and isinstance(
                    upper.value,
                    int,
                )
                and upper.value
                in {
                    5,
                    7,
                }
            ):

                fail(
                    "shot_executor.py contains "
                    "an unintended hardcoded "
                    f"{upper.value}-reference limit."
                )

    require(
        "len(" in source,
        "shot_executor.py does not "
        "derive reference count dynamically.",
    )

    print(
        "OK   multi-reference execution contract"
    )


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Validate the complete "
            "LTX-13B repository and "
            "optional live runtime."
        )
    )

    parser.add_argument(
        "--require-runtime",
        action="store_true",
        help=(
            "Also validate the currently "
            "running ComfyUI instance."
        ),
    )

    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help=(
            "Require CUDA/GPU availability. "
            "Automatically enables runtime "
            "validation."
        ),
    )

    parser.add_argument(
        "--runtime-url",
        default=(
            "http://127.0.0.1:8188"
        ),
        help=(
            "ComfyUI runtime URL. "
            "Default: "
            "http://127.0.0.1:8188"
        ),
    )

    args = parser.parse_args()

    if args.require_cuda:

        args.require_runtime = True

    # =========================================================================
    # HEADER
    # =========================================================================

    print(
        "=" * 80
    )

    print(
        "LTX-13B COMPLETE PROJECT VALIDATION"
    )

    print(
        "=" * 80
    )

    print(
        f"Project root: {PROJECT_ROOT}"
    )

    print()

    # =========================================================================
    # STATIC REPOSITORY VALIDATION
    # =========================================================================

    print(
        "=" * 80
    )

    print(
        "STATIC REPOSITORY VALIDATION"
    )

    print(
        "=" * 80
    )

    validate_files()

    validate_python()

    lock = load_lock()

    validate_lock(
        lock
    )

    validate_model_architecture()

    validate_workflows()

    validate_planning_contracts()

    validate_adapter()

    validate_real_detailer_conversion()

    validate_compatibility_builder()

    validate_production_application()

    validate_kaggle_wiring()

    validate_cpu_preflight()

    validate_live_runtime_verifier()

    validate_port_contract()

    validate_live_verifier_matches_runtime_contract()

    validate_reference_image_generator()

    validate_multi_reference_execution()

    validate_identity_reference_contract()

    # =========================================================================
    # LIVE RUNTIME
    # =========================================================================

    if args.require_runtime:

        validate_live_runtime(
            args.runtime_url.rstrip("/")
        )

    # =========================================================================
    # CUDA
    # =========================================================================

    if args.require_cuda:

        print()

        print(
            "=" * 80
        )

        print(
            "CUDA VALIDATION"
        )

        print(
            "=" * 80
        )

        validate_cuda()

    # =========================================================================
    # FINAL
    # =========================================================================

    print()

    print(
        "=" * 80
    )

    print(
        "🎉 LTX-13B VALIDATION PASSED"
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

    print()

    print(
        "Canonical ComfyUI port:"
    )

    print(
        "  8188"
    )

    print()

    print(
        "Multi-GPU port strategy:"
    )

    print(
        "  base_port + GPU ID"
    )

    print()

    if args.require_runtime:

        print(
            "Live ComfyUI:"
        )

        print(
            "  VERIFIED"
        )

    else:

        print(
            "Live ComfyUI:"
        )

        print(
            "  NOT CHECKED "
            "(use --require-runtime)"
        )

    print()

    if args.require_cuda:

        print(
            "CUDA:"
        )

        print(
            "  VERIFIED"
        )

    else:

        print(
            "CUDA:"
        )

        print(
            "  NOT CHECKED "
            "(use --require-cuda)"
        )

    print()

    print(
        "NO VIDEO WAS GENERATED "
        "BY THIS VALIDATOR."
    )


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":

    main()
