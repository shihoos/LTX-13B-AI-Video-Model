#!/usr/bin/env python3

"""
LTX-13B CPU / REPOSITORY PREFLIGHT

Purpose
-------
Validate the complete repository architecture before GPU/runtime startup.

This preflight is CPU-only.

It does NOT:

- install packages
- start ComfyUI
- load models
- require CUDA
- require GPU
- require Kaggle dataset mounts
- generate video

Single source of truth:

    kaggle/compatibility_lock.yaml

Canonical local ComfyUI base port:

    8188

Multi-GPU local port strategy:

    GPU N -> 8188 + N

Known stale local ComfyUI port:

    8219

This preflight validates the current repository contracts only.
It does not perform obsolete compatibility-file checks.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path


# ============================================================================
# PROJECT PATHS
# ============================================================================

ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

KAGGLE_DIR = ROOT / "kaggle"
SCRIPTS_DIR = ROOT / "scripts"
WORKFLOWS_DIR = ROOT / "workflows"
EXECUTION_DIR = ROOT / "execution"
COMPATIBILITY_DIR = ROOT / "compatibility"
PLANNER_DIR = ROOT / "planner"
PIPELINE_DIR = ROOT / "pipeline"
SCHEMAS_DIR = ROOT / "schemas"


# ============================================================================
# IMPORTANT FILES
# ============================================================================

LOCK_FILE = KAGGLE_DIR / "compatibility_lock.yaml"

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

ADAPTER = (
    EXECUTION_DIR / "comfy_workflow_adapter.py"
)

REFERENCE_IMAGE_GENERATOR = (
    EXECUTION_DIR / "reference_image_generator.py"
)

COMPATIBILITY_BUILDER = (
    COMPATIBILITY_DIR / "prepare_modern_ltx.py"
)

GENERATE_VIDEO = (
    SCRIPTS_DIR / "generate_video.py"
)

VALIDATOR = (
    SCRIPTS_DIR / "validate_project.py"
)

LIVE_RUNTIME = (
    KAGGLE_DIR / "verify_live_runtime.py"
)

LAUNCH = (
    KAGGLE_DIR / "launch.py"
)

BOOTSTRAP = (
    KAGGLE_DIR / "bootstrap.py"
)

PREFLIGHT_MODERN = (
    KAGGLE_DIR / "preflight_modern.py"
)

START_COMFYUI = (
    KAGGLE_DIR / "start_comfyui.py"
)

CHARACTER_REFERENCE_PROCESSOR = (
    EXECUTION_DIR
    / "character_reference_processor.py"
)

CHARACTER_SCHEMA = (
    SCHEMAS_DIR
    / "character.py"
)

SHOT_SCHEMA = (
    SCHEMAS_DIR
    / "shot.py"
)


# ============================================================================
# CONSTANTS
# ============================================================================

CANONICAL_COMFYUI_PORT = 8188
STALE_COMFYUI_PORT = 8219

COMFYUI_HOST = "127.0.0.1"
COMFYUI_PROTOCOL = "http"
COMFYUI_PORT_STRATEGY = "base_port_plus_gpu_id"

LEGACY_LATENT_LOADER = (
    "LTXVLatentUpsamplerModelLoader"
)

MODERN_LATENT_LOADER = (
    "LatentUpscaleModelLoader"
)


# ============================================================================
# ERROR HELPERS
# ============================================================================

def fail(message: str) -> None:
    raise RuntimeError(message)


def require(
    condition: bool,
    message: str,
) -> None:
    if not condition:
        fail(message)


# ============================================================================
# FILE HELPERS
# ============================================================================

def read_text(path: Path) -> str:
    require(
        path.is_file(),
        "Required file missing:\n"
        f"{path}",
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


def parse_python(path: Path) -> ast.Module:
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


def parse_json(path: Path) -> dict:
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


# ============================================================================
# PYTHON FILE DISCOVERY
# ============================================================================

def python_files() -> list[Path]:
    result: list[Path] = []

    excluded_parts = {
        ".git",
        "ComfyUI",
        ".runtime_ltx098",
        "__pycache__",
    }

    for path in ROOT.rglob("*.py"):
        relative = path.relative_to(ROOT)

        if any(
            part in excluded_parts
            for part in relative.parts
        ):
            continue

        result.append(path)

    return sorted(result)


# ============================================================================
# WORKFLOW HELPERS
# ============================================================================

def node_types(
    workflow: dict,
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


def find_nodes(
    workflow: dict,
    node_type: str,
) -> list[dict]:

    nodes = workflow.get(
        "nodes",
        [],
    )

    return [
        node
        for node in nodes
        if (
            isinstance(node, dict)
            and node.get("type")
            == node_type
        )
    ]


# ============================================================================
# 1. FILE STRUCTURE
# ============================================================================

def check_files() -> None:

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
    ]

    for relative in required:
        require(
            (ROOT / relative).is_file(),
            "Required file missing:\n"
            f"{ROOT / relative}",
        )

    print(
        f"OK   repository files: "
        f"{len(required)}"
    )

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
        EXECUTION_DIR
        / "shot_executor.py"
    )

    manager = read_text(
        ROOT
        / "pipeline"
        / "reference_manager.py"
    )

    planner = read_text(
        ROOT
        / "planner"
        / "character_planner.py"
    )

    shot_planner = read_text(
        ROOT
        / "planner"
        / "shot_planner.py"
    )

    # ------------------------------------------------------------------------
    # Processor
    # ------------------------------------------------------------------------

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

    # ------------------------------------------------------------------------
    # Character schema
    # ------------------------------------------------------------------------

    require(
        "reference_mask_path"
        in character_schema,
        "schemas/character.py must contain "
        "reference_mask_path.",
    )

    # ------------------------------------------------------------------------
    # Shot schema
    # ------------------------------------------------------------------------

    require(
        "reference_masks"
        in shot_schema,
        "schemas/shot.py must contain "
        "reference_masks.",
    )

    # ------------------------------------------------------------------------
    # Reference manager
    # ------------------------------------------------------------------------

    require(
        "mask_path"
        in manager,
        "ReferenceManager must propagate "
        "identity mask paths.",
    )

    # ------------------------------------------------------------------------
    # Character planner
    # ------------------------------------------------------------------------

    require(
        "reference_mask_path"
        in planner,
        "CharacterPlanner must propagate "
        "reference_mask_path.",
    )

    # ------------------------------------------------------------------------
    # Shot planner
    # ------------------------------------------------------------------------

    require(
        "reference_masks"
        in shot_planner,
        "ShotPlanner must propagate "
        "reference_masks.",
    )

    # ------------------------------------------------------------------------
    # Adapter
    # ------------------------------------------------------------------------

    for text in (
        "set_identity_reference_image",
        "set_identity_mask_image",
    ):

        require(
            text in adapter,
            "Workflow adapter missing "
            f"{text}.",
        )

    # ------------------------------------------------------------------------
    # Executor
    # ------------------------------------------------------------------------

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
# 2. PYTHON SYNTAX
# ============================================================================

def check_python_syntax() -> None:

    files = python_files()

    require(
        bool(files),
        "No Python files found.",
    )

    for path in files:
        parse_python(path)

    print(
        "OK   Python AST syntax: "
        f"{len(files)} files"
    )


# ============================================================================
# 3. COMPATIBILITY LOCK
# ============================================================================

def load_lock() -> dict:

    try:
        import yaml
    except ImportError as error:
        fail(
            "PyYAML is required for CPU preflight:\n"
            f"{error}"
        )

    try:
        data = yaml.safe_load(
            read_text(LOCK_FILE)
        )
    except Exception as error:
        fail(
            "Could not parse "
            "compatibility_lock.yaml:\n"
            f"{error}"
        )

    require(
        isinstance(data, dict),
        "compatibility_lock.yaml "
        "must contain a mapping.",
    )

    return data


def check_lock(
    lock: dict,
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

    comfy = lock["comfyui"]
    models = lock["models"]
    custom_nodes = lock["custom_nodes"]

    require(
        isinstance(comfy, dict),
        "lock.comfyui must be a mapping.",
    )

    require(
        isinstance(models, dict),
        "lock.models must be a mapping.",
    )

    require(
        isinstance(custom_nodes, dict),
        "lock.custom_nodes "
        "must be a mapping.",
    )

    commit = str(
        comfy.get("commit", "")
    )

    require(
        len(commit) == 40
        and all(
            character
            in "0123456789abcdef"
            for character
            in commit.lower()
        ),
        "ComfyUI lock commit must be "
        "a 40-character SHA.",
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
            isinstance(spec, dict),
            f"models.{name} "
            "must be a mapping.",
        )

        for field in (
            "dataset",
            "filename",
            "target",
        ):
            value = spec.get(field)

            require(
                isinstance(value, str)
                and value.strip(),
                f"models.{name}.{field} "
                "is missing.",
            )

    runtime = lock["python_runtime"]

    require(
        isinstance(runtime, dict),
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
        value = runtime.get(field)

        require(
            isinstance(value, str)
            and value.strip(),
            "lock.python_runtime."
            f"{field} is missing.",
        )

    print(
        "OK   compatibility lock"
    )


# ============================================================================
# 4. RUNTIME / NETWORK CONTRACT
# ============================================================================

def check_runtime_network_contract(
    lock: dict,
) -> None:

    runtime = lock.get("runtime")

    require(
        isinstance(runtime, dict),
        "compatibility_lock.yaml "
        "missing runtime section.",
    )

    network = runtime.get("network")

    require(
        isinstance(network, dict),
        "compatibility_lock.yaml "
        "missing runtime.network section.",
    )

    require(
        network.get("host")
        == COMFYUI_HOST,
        "runtime.network.host must be "
        "127.0.0.1.",
    )

    require(
        network.get("base_port")
        == CANONICAL_COMFYUI_PORT,
        "runtime.network.base_port "
        "must be 8188.",
    )

    require(
        network.get("port_strategy")
        == COMFYUI_PORT_STRATEGY,
        "runtime.network.port_strategy "
        "must be base_port_plus_gpu_id.",
    )

    require(
        network.get("protocol")
        == COMFYUI_PROTOCOL,
        "runtime.network.protocol "
        "must be http.",
    )

    stale_ports = network.get(
        "stale_ports"
    )

    require(
        isinstance(stale_ports, list),
        "runtime.network.stale_ports "
        "must be a list.",
    )

    require(
        STALE_COMFYUI_PORT
        in stale_ports,
        "runtime.network.stale_ports "
        "must contain 8219.",
    )

    print(
        "OK   runtime/network contract"
    )


# ============================================================================
# 5. QWEN PLANNING CONTRACT
# ============================================================================

def check_qwen_planning_contract() -> None:

    config = read_text(
        PLANNER_DIR / "config.py"
    )

    expected_constants = {
        "QWEN_STORY_TEMPERATURE",
        "QWEN_CHARACTER_DETECTION_TEMPERATURE",
        "QWEN_CHARACTER_PLAN_TEMPERATURE",
        "QWEN_SCENE_PLAN_TEMPERATURE",
        "QWEN_SHOT_PLAN_TEMPERATURE",
        "QWEN_TOP_P",
    }

    for text in expected_constants:
        require(
            text in config,
            "planner/config.py missing "
            "Qwen planning constant:\n"
            f"{text}",
        )

    planner_contracts = {
        "story_planner.py":
            "QWEN_STORY_TEMPERATURE",

        "character_detector.py":
            "QWEN_CHARACTER_DETECTION_TEMPERATURE",

        "character_planner.py":
            "QWEN_CHARACTER_PLAN_TEMPERATURE",

        "scene_planner.py":
            "QWEN_SCENE_PLAN_TEMPERATURE",

        "shot_planner.py":
            "QWEN_SHOT_PLAN_TEMPERATURE",
    }

    for filename, contract in (
        planner_contracts.items()
    ):
        source = read_text(
            PLANNER_DIR / filename
        )

        require(
            contract in source,
            f"planner/{filename} is not "
            f"using {contract}.",
        )

    print(
        "OK   task-specific Qwen temperature contract"
    )


# ============================================================================
# 6. CHARACTER / SCENE PLANNING CONTRACT
# ============================================================================

def check_planning_schema_contract() -> None:

    character_schema = read_text(
        SCHEMAS_DIR / "character.py"
    )

    scene_schema = read_text(
        SCHEMAS_DIR / "scene.py"
    )

    character_prompt = read_text(
        PLANNER_DIR.parent
        / "prompts"
        / "qwen"
        / "character_plan.txt"
    )

    scene_prompt = read_text(
        PLANNER_DIR.parent
        / "prompts"
        / "qwen"
        / "scene_plan.txt"
    )

    character_planner = read_text(
        PLANNER_DIR / "character_planner.py"
    )

    scene_planner = read_text(
        PLANNER_DIR / "scene_planner.py"
    )

    # ------------------------------------------------------------------------
    # Character state
    #
    # The Character schema stores the complete state in one dictionary:
    #
    #     character_state
    #
    # The individual state keys are defined by the Qwen prompt and passed
    # through CharacterPlanner as part of that dictionary.
    # ------------------------------------------------------------------------

    require(
        "character_state" in character_schema,
        "Character schema missing:\n"
        "character_state",
    )

    for text in (
        "emotional_state",
        "physical_state",
        "clothing_state",
        "carried_objects",
        "injuries",
    ):

        require(
            text in character_prompt,
            "Character prompt missing:\n"
            f"{text}",
        )

    require(
        "character_state" in character_planner,
        "Character planner missing:\n"
        "character_state",
    )

    # ------------------------------------------------------------------------
    # Scene planning fields
    #
    # These are actual Scene schema fields, so each must exist in:
    #
    #     schemas/scene.py
    #     prompts/qwen/scene_plan.txt
    #     planner/scene_planner.py
    # ------------------------------------------------------------------------

    scene_fields = {
        "weather",
        "atmosphere",
        "environment_details",
        "key_props",
        "scene_objective",
        "mood",
        "lighting",
    }

    for text in scene_fields:

        require(
            text in scene_schema,
            "Scene schema missing:\n"
            f"{text}",
        )

        require(
            text in scene_prompt,
            "Scene prompt missing:\n"
            f"{text}",
        )

        require(
            text in scene_planner,
            "Scene planner missing:\n"
            f"{text}",
        )

    print(
        "OK   character/scene planning contract"
    )


# ============================================================================
# 7. WORKFLOWS
# ============================================================================

def check_workflows() -> None:

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
        "LoadImage",
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

    base_sampler_nodes = find_nodes(
        base,
        "LTXVBaseSampler",
    )

    require(
        len(base_sampler_nodes) == 1,
        "BASE workflow must contain "
        "exactly one LTXVBaseSampler.",
    )

    load_image_nodes = find_nodes(
        base,
        "LoadImage",
    )

    require(
        len(load_image_nodes) == 1,
        "BASE workflow must contain "
        "exactly one LoadImage node.",
    )

    sampler_inputs = (
        base_sampler_nodes[0]
        .get("inputs", [])
    )

    require(
        any(
            isinstance(item, dict)
            and item.get("name")
            == "optional_cond_images"
            for item in sampler_inputs
        ),
        "LTXVBaseSampler must expose "
        "optional_cond_images.",
    )

    require(
        any(
            isinstance(item, dict)
            and item.get("name")
            == "optional_cond_indices"
            for item in sampler_inputs
        )
        or
        "optional_cond_indices"
        in base_sampler_nodes[0].get(
            "widgets_values_named",
            {},
        ),
        "LTXVBaseSampler must expose "
        "optional_cond_indices.",
    )

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
        "DETAILER source workflow missing nodes:\n"
        + "\n".join(
            sorted(missing)
        ),
    )

    require(
        MODERN_LATENT_LOADER
        not in detailer_types,
        "DETAILER source workflow "
        "must retain the legacy loader.",
    )

    print(
        "OK   BASE workflow"
    )

    print(
        "OK   BASE conditioning-image contract"
    )

    print(
        "OK   DETAILER source workflow"
    )


# ============================================================================
# 8. WORKFLOW ADAPTER
# ============================================================================

def check_adapter() -> None:

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
        "set_input_image",
    }

    for text in required_strings:
        require(
            text in source,
            "ComfyWorkflowAdapter "
            "missing contract:\n"
            f"{text}",
        )

    print(
        "OK   ComfyWorkflowAdapter contract"
    )


def check_real_detailer_conversion() -> None:

    previous_path = list(sys.path)

    try:
        root_string = str(ROOT)

        if root_string not in sys.path:
            sys.path.insert(
                0,
                root_string,
            )

        from execution.comfy_workflow_adapter import (
            ComfyWorkflowAdapter,
        )

        adapter = ComfyWorkflowAdapter(
            DETAILER_WORKFLOW
        )

        api = (
            adapter.to_api_workflow()
        )

        require(
            isinstance(api, dict)
            and bool(api),
            "DETAILER conversion "
            "produced empty API workflow.",
        )

        adapter.validate_modern_detailer(
            api
        )

        node_types_api = {
            node.get("class_type")
            for node in api.values()
            if isinstance(node, dict)
        }

        require(
            MODERN_LATENT_LOADER
            in node_types_api,
            "DETAILER conversion did not "
            "produce LatentUpscaleModelLoader.",
        )

        require(
            LEGACY_LATENT_LOADER
            not in node_types_api,
            "Legacy latent loader survived "
            "DETAILER conversion.",
        )

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


# ============================================================================
# 9. REFERENCE IMAGE GENERATOR
# ============================================================================

def check_reference_image_generator() -> None:

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


# ============================================================================
# 10. COMPATIBILITY BUILDER
# ============================================================================

def check_compatibility_builder() -> None:

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

    for text in required:
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
# 11. PRODUCTION APPLICATION
# ============================================================================

def check_generate_video() -> None:

    source = read_text(
        GENERATE_VIDEO
    )

    required = {
        "ProductionOrchestrator",
        "ProductionManager",
        "ProductionRunner",
        "BASE_WORKFLOW",
        "DETAILER_WORKFLOW",
        "gpu-url",
        str(CANONICAL_COMFYUI_PORT),
        str(STALE_COMFYUI_PORT),
    }

    for text in required:
        require(
            text in source,
            "generate_video.py "
            "missing contract:\n"
            f"{text}",
        )

    tree = parse_python(
        GENERATE_VIDEO
    )

    stale_guard_found = False

    for node in ast.walk(tree):

        if not isinstance(
            node,
            ast.FunctionDef,
        ):
            continue

        if node.name != "validate_worker_url":
            continue

        function_source = (
            ast.get_source_segment(
                source,
                node,
            )
        )

        if not function_source:
            continue

        if (
            str(STALE_COMFYUI_PORT)
            not in function_source
        ):
            continue

        if (
            "127.0.0.1:8219"
            not in function_source
            and
            "localhost:8219"
            not in function_source
        ):
            continue

        if (
            "Stale ComfyUI port detected"
            not in function_source
        ):
            continue

        if (
            "CANONICAL_COMFYUI_PORT"
            not in function_source
        ):
            continue

        stale_guard_found = True

    require(
        stale_guard_found,
        "generate_video.py does not contain "
        "the expected stale-port rejection guard.",
    )

    print(
        "OK   production application wiring"
    )

    print(
        "OK   stale-port rejection guard"
    )


# ============================================================================
# 12. KAGGLE STARTUP
# ============================================================================

def check_kaggle_wiring() -> None:

    launch = read_text(LAUNCH)
    bootstrap = read_text(BOOTSTRAP)
    preflight = read_text(PREFLIGHT_MODERN)
    start_comfyui = read_text(
        START_COMFYUI
    )

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

    require(
        "compatibility_lock.yaml"
        in bootstrap,
        "bootstrap.py is not lock-driven.",
    )

    require(
        "compatibility_lock.yaml"
        in preflight,
        "preflight_modern.py is not lock-driven.",
    )

    require(
        str(CANONICAL_COMFYUI_PORT)
        in start_comfyui,
        "start_comfyui.py is not using "
        "canonical port 8188.",
    )

    require(
        str(STALE_COMFYUI_PORT)
        not in start_comfyui,
        "start_comfyui.py contains "
        "stale port 8219.",
    )

    print(
        "OK   Kaggle launcher/bootstrap wiring"
    )


# ============================================================================
# 13. LIVE RUNTIME VERIFIER
# ============================================================================

def check_live_runtime_contract() -> None:

    source = read_text(
        LIVE_RUNTIME
    )

    required = {
        "127.0.0.1:8188",
        "/system_stats",
        "/object_info",
        "require_choice(",
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
        LEGACY_LATENT_LOADER,
        "LTXVTiledVAEDecode",
        "LTXVFilmGrain",
        "LoraLoaderModelOnly",
        MODERN_LATENT_LOADER,
    }

    for text in required:
        require(
            text in source,
            "verify_live_runtime.py "
            "missing contract:\n"
            f"{text}",
        )

    require(
        str(STALE_COMFYUI_PORT)
        not in source,
        "verify_live_runtime.py contains "
        "stale port 8219.",
    )

    model_checks = {
        'models["ltx_q4"]["filename"]',
        'models["t5_q4"]["filename"]',
        'models["vae"]["filename"]',
        'models["ic_lora"]["filename"]',
        'models["spatial_upscaler"]["filename"]',
    }

    for text in model_checks:
        require(
            text in source,
            "Live verifier does not verify "
            "locked model:\n"
            f"{text}",
        )

    print(
        "OK   live runtime verifier contract"
    )


# ============================================================================
# 14. CPU PREFLIGHT SELF CONTRACT
# ============================================================================

def check_self_contract() -> None:

    source = read_text(
        Path(__file__)
    )

    required = {
        "compatibility_lock.yaml",
        LEGACY_LATENT_LOADER,
        MODERN_LATENT_LOADER,
        "execution/reference_image_generator.py",
        "ReferenceImageGenerator",
        "CheckpointLoaderSimple",
        "SaveImage",
        "data/characters/generated",
        "QWEN_SHOT_PLAN_TEMPERATURE",
        "character_state",
        "weather",
        "scene_objective",
    }

    for text in required:
        require(
            text in source,
            "CPU preflight missing "
            "self-contract:\n"
            f"{text}",
        )

    print(
        "OK   CPU preflight self-contract"
    )


# ============================================================================
# 15. PORT CONTRACT
# ============================================================================

def check_port_contract() -> None:

    generator = read_text(
        GENERATE_VIDEO
    )

    start_comfyui = read_text(
        START_COMFYUI
    )

    live = read_text(
        LIVE_RUNTIME
    )

    require(
        str(CANONICAL_COMFYUI_PORT)
        in generator,
        "generate_video.py does not "
        "declare canonical port 8188.",
    )

    tree = parse_python(
        GENERATE_VIDEO
    )

    stale_guard_count = 0

    for node in ast.walk(tree):

        if not isinstance(
            node,
            ast.FunctionDef,
        ):
            continue

        if node.name != "validate_worker_url":
            continue

        function_source = (
            ast.get_source_segment(
                generator,
                node,
            )
        )

        if (
            function_source
            and str(STALE_COMFYUI_PORT)
            in function_source
        ):
            stale_guard_count += 1

    require(
        stale_guard_count == 1,
        "generate_video.py must contain "
        "exactly one stale-port guard.",
    )

    require(
        str(CANONICAL_COMFYUI_PORT)
        in start_comfyui,
        "start_comfyui.py does not use "
        "canonical port 8188.",
    )

    require(
        str(STALE_COMFYUI_PORT)
        not in start_comfyui,
        "start_comfyui.py contains "
        "stale port 8219.",
    )

    require(
        str(CANONICAL_COMFYUI_PORT)
        in live,
        "verify_live_runtime.py does not "
        "contain canonical port 8188.",
    )

    require(
        str(STALE_COMFYUI_PORT)
        not in live,
        "verify_live_runtime.py contains "
        "stale port 8219.",
    )

    print(
        "OK   global ComfyUI port contract"
    )


# ============================================================================
# 16. VALIDATOR ALIGNMENT
# ============================================================================

def check_validator_alignment() -> None:

    source = read_text(
        VALIDATOR
    )

    required = {
        "scripts/cpu_preflight.py",
        "compatibility_lock.yaml",
        "reference_image_generator.py",
        "validate_reference_image_generator",
        "LatentUpscaleModelLoader",
        "LTXVLatentUpsamplerModelLoader",
        "runtime",
        "runtime.network",
        "base_port_plus_gpu_id",
        "8188",
        "8219",
    }

    for text in required:
        require(
            text in source,
            "validate_project.py is missing "
            "expected synchronization contract:\n"
            f"{text}",
        )

    print(
        "OK   validator alignment contract"
    )


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:

    print("=" * 80)
    print(
        "LTX-13B CPU / REPOSITORY PREFLIGHT"
    )
    print("=" * 80)
    print(
        f"Project root: {ROOT}"
    )
    print()

    check_files()
    check_python_syntax()

    lock = load_lock()

    check_lock(lock)
    check_runtime_network_contract(lock)
    check_qwen_planning_contract()
    check_planning_schema_contract()
    check_workflows()
    check_adapter()
    check_real_detailer_conversion()
    check_reference_image_generator()
    check_compatibility_builder()
    check_generate_video()
    check_kaggle_wiring()
    check_live_runtime_contract()
    check_self_contract()
    check_port_contract()
    check_validator_alignment()
    
    print()
    print("=" * 80)
    print(
        "✅ CPU / REPOSITORY PREFLIGHT PASSED"
    )
    print("=" * 80)
    print()
    print(
        "Single source of truth:"
    )
    print(
        "  kaggle/compatibility_lock.yaml"
    )
    print()
    print(
        "Canonical ComfyUI base port:"
    )
    print(
        "  8188"
    )
    print()
    print(
        "Multi-GPU port strategy:"
    )
    print(
        "  GPU N -> 8188 + N"
    )
    print()
    print(
        "Stale local port 8219:"
    )
    print(
        "  REJECTED BY PRODUCTION WORKER VALIDATION"
    )
    print()
    print(
        "No GPU was required."
    )
    print(
        "No ComfyUI instance was started."
    )
    print(
        "No video was generated."
    )


if __name__ == "__main__":
    main()
