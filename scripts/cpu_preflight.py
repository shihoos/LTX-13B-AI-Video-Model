#!/usr/bin/env python3

"""
LTX-13B CPU / REPOSITORY PREFLIGHT

Purpose
-------
Validate the complete repository architecture before GPU/runtime startup.

This preflight is intentionally CPU-only.

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


Canonical local ComfyUI port:

    8188

Known stale local ComfyUI port:

    8219

IMPORTANT
---------
generate_video.py intentionally contains the stale port 8219 as a
rejection guard.

Therefore this preflight does NOT reject the string "8219" merely
because it exists.

Instead it verifies that:

    8188 = canonical local endpoint

and that 8219 is used only by the explicit stale-endpoint guard.

This keeps the CPU preflight synchronized with the production
application contract.
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

KAGGLE_DIR = (
    ROOT
    / "kaggle"
)

SCRIPTS_DIR = (
    ROOT
    / "scripts"
)

WORKFLOWS_DIR = (
    ROOT
    / "workflows"
)

EXECUTION_DIR = (
    ROOT
    / "execution"
)

COMPATIBILITY_DIR = (
    ROOT
    / "compatibility"
)


# ============================================================================
# IMPORTANT FILES
# ============================================================================

LOCK_FILE = (
    KAGGLE_DIR
    / "compatibility_lock.yaml"
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

ADAPTER = (
    EXECUTION_DIR
    / "comfy_workflow_adapter.py"
)

COMPATIBILITY_BUILDER = (
    COMPATIBILITY_DIR
    / "prepare_modern_ltx.py"
)

GENERATE_VIDEO = (
    SCRIPTS_DIR
    / "generate_video.py"
)

VALIDATOR = (
    SCRIPTS_DIR
    / "validate_project.py"
)

LIVE_RUNTIME = (
    KAGGLE_DIR
    / "verify_live_runtime.py"
)

LAUNCH = (
    KAGGLE_DIR
    / "launch.py"
)

BOOTSTRAP = (
    KAGGLE_DIR
    / "bootstrap.py"
)

PREFLIGHT_MODERN = (
    KAGGLE_DIR
    / "preflight_modern.py"
)

START_COMFYUI = (
    KAGGLE_DIR
    / "start_comfyui.py"
)


# ============================================================================
# CONSTANTS
# ============================================================================

CANONICAL_COMFYUI_PORT = 8188

STALE_COMFYUI_PORT = 8219

LEGACY_LATENT_LOADER = (
    "LTXVLatentUpsamplerModelLoader"
)

MODERN_LATENT_LOADER = (
    "LatentUpscaleModelLoader"
)


# ============================================================================
# ERROR HELPERS
# ============================================================================

def fail(
    message: str,
) -> None:

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
) -> dict:

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
        "JSON root must be an object:\n"
        f"{path}",
    )

    return data


# ============================================================================
# PYTHON FILE DISCOVERY
# ============================================================================

def python_files() -> list[Path]:

    result: list[Path] = []

    for path in ROOT.rglob(
        "*.py"
    ):

        relative = (
            path.relative_to(ROOT)
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


# ============================================================================
# 1. FILE STRUCTURE
# ============================================================================

def check_files() -> None:

    required = [

        # --------------------------------------------------------------------
        # Planner
        # --------------------------------------------------------------------

        "planner/config.py",
        "planner/qwen_loader.py",
        "planner/story_planner.py",
        "planner/character_detector.py",
        "planner/character_planner.py",
        "planner/scene_planner.py",
        "planner/shot_planner.py",

        # --------------------------------------------------------------------
        # Pipeline
        # --------------------------------------------------------------------

        "pipeline/__init__.py",
        "pipeline/continuity_manager.py",
        "pipeline/modes.py",
        "pipeline/production_manager.py",
        "pipeline/production_orchestrator.py",
        "pipeline/reference_manager.py",

        # --------------------------------------------------------------------
        # Execution
        # --------------------------------------------------------------------

        "execution/__init__.py",
        "execution/checkpoint_manager.py",
        "execution/comfy_client.py",
        "execution/comfy_workflow_adapter.py",
        "execution/shot_executor.py",
        "execution/assembly_manager.py",
        "execution/production_runner.py",

        # --------------------------------------------------------------------
        # Scheduler
        # --------------------------------------------------------------------

        "scheduler/__init__.py",
        "scheduler/gpu_scheduler.py",
        "scheduler/shot_queue.py",

        # --------------------------------------------------------------------
        # Schemas
        # --------------------------------------------------------------------

        "schemas/__init__.py",
        "schemas/character.py",
        "schemas/scene.py",
        "schemas/shot.py",
        "schemas/parser.py",

        # --------------------------------------------------------------------
        # Kaggle
        # --------------------------------------------------------------------

        "kaggle/compatibility_lock.yaml",
        "kaggle/bootstrap.py",
        "kaggle/config.py",
        "kaggle/launch.py",
        "kaggle/preflight_modern.py",
        "kaggle/start_comfyui.py",
        "kaggle/verify_live_runtime.py",

        # --------------------------------------------------------------------
        # Compatibility
        # --------------------------------------------------------------------

        "compatibility/prepare_modern_ltx.py",

        # --------------------------------------------------------------------
        # Scripts
        # --------------------------------------------------------------------

        "scripts/cpu_preflight.py",
        "scripts/generate_video.py",
        "scripts/validate_project.py",

        # --------------------------------------------------------------------
        # Workflows
        # --------------------------------------------------------------------

        "workflows/baseline/ltxv-13b-dist-i2v-base.json",
        "workflows/detailer/ltxv-13b-098-ic-lora-upscale.json",
    ]

    for relative in required:

        path = (
            ROOT
            / relative
        )

        require(
            path.is_file(),
            "Required file missing:\n"
            f"{path}",
        )

    print(
        f"OK   repository files: "
        f"{len(required)}"
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

        parse_python(
            path
        )

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
            read_text(
                LOCK_FILE
            )
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

    custom_nodes = lock[
        "custom_nodes"
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

    require(
        isinstance(
            custom_nodes,
            dict,
        ),
        "lock.custom_nodes "
        "must be a mapping.",
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
        "ComfyUI lock commit must be "
        "a 40-character SHA.",
    )

    # ------------------------------------------------------------------------
    # Required models
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

        spec = models[
            name
        ]

        require(
            isinstance(
                spec,
                dict,
            ),
            f"models.{name} "
            "must be a mapping.",
        )

        for key in (
            "dataset",
            "filename",
            "target",
        ):

            value = spec.get(
                key
            )

            require(
                isinstance(
                    value,
                    str,
                )
                and value.strip(),
                f"models.{name}.{key} "
                "is missing.",
            )

    print(
        "OK   compatibility lock"
    )


# ============================================================================
# 4. WORKFLOWS
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

    # ------------------------------------------------------------------------
    # BASE
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
    # DETAILER
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

    # The source workflow MUST remain legacy.
    # The adapter converts it to the modern API graph.

    require(
        MODERN_LATENT_LOADER
        not in detailer_types,
        "DETAILER source workflow "
        "unexpectedly contains "
        "LatentUpscaleModelLoader.\n\n"
        "The source workflow must retain "
        "the legacy loader.\n"
        "The adapter performs the conversion.",
    )

    print(
        "OK   BASE workflow"
    )

    print(
        "OK   DETAILER source workflow"
    )


# ============================================================================
# 5. WORKFLOW ADAPTER
# ============================================================================

def check_adapter() -> None:

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


def check_real_conversion() -> None:

    previous_path = list(
        sys.path
    )

    try:

        root_string = str(
            ROOT
        )

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
            isinstance(
                api,
                dict,
            )
            and bool(api),
            "DETAILER conversion "
            "produced empty API workflow.",
        )

        adapter.validate_modern_detailer(
            api
        )

        types = {
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
            MODERN_LATENT_LOADER
            in types,
            "DETAILER conversion did not "
            "produce LatentUpscaleModelLoader.",
        )

        require(
            LEGACY_LATENT_LOADER
            not in types,
            "Legacy LTX latent loader "
            "survived conversion.",
        )

    except Exception as error:

        fail(
            "Real DETAILER adapter "
            "conversion failed:\n"
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
# 6. COMPATIBILITY BUILDER
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
# 7. PRODUCTION APPLICATION WIRING
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

    for text in (
        required
    ):

        require(
            text in source,
            "generate_video.py "
            "missing contract:\n"
            f"{text}",
        )

    # ------------------------------------------------------------------------
    # IMPORTANT PORT CONTRACT
    # ------------------------------------------------------------------------
    #
    # 8188 MUST be canonical.
    #
    # 8219 MUST NOT be an active endpoint.
    #
    # However, generate_video.py intentionally contains 8219 inside
    # validate_worker_url() so stale local workers are explicitly rejected.
    #
    # Therefore we validate the AST semantics rather than simply searching
    # for absence of the string "8219".
    # ------------------------------------------------------------------------

    tree = parse_python(
        GENERATE_VIDEO
    )

    stale_guard_found = False

    for node in ast.walk(
        tree
    ):

        if not isinstance(
            node,
            ast.FunctionDef,
        ):

            continue

        if node.name != (
            "validate_worker_url"
        ):

            continue

        function_source = ast.get_source_segment(
            source,
            node,
        )

        if not function_source:

            continue

        if (
            "8219"
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
        "the expected stale-port rejection "
        "guard for 8219.",
    )

    print(
        "OK   production application wiring"
    )

    print(
        "OK   canonical ComfyUI port: "
        f"{CANONICAL_COMFYUI_PORT}"
    )

    print(
        "OK   stale-port rejection guard: "
        f"{STALE_COMFYUI_PORT}"
    )


# ============================================================================
# 8. KAGGLE STARTUP WIRING
# ============================================================================

def check_kaggle_wiring() -> None:

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
    # ------------------------------------------------------------------------

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
        "start_comfyui.py"
        in launch,
        "launch.py is not wired to "
        "start_comfyui.py.",
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

    # ------------------------------------------------------------------------
    # start_comfyui.py
    # ------------------------------------------------------------------------

    require(
        str(
            CANONICAL_COMFYUI_PORT
        )
        in start_comfyui,
        "ComfyUI launcher is not "
        "using canonical port 8188.",
    )

    require(
        str(
            STALE_COMFYUI_PORT
        )
        not in start_comfyui,
        "ComfyUI launcher contains "
        "stale port 8219.",
    )

    print(
        "OK   Kaggle launcher wiring"
    )

    print(
        "OK   bootstrap wiring"
    )

    print(
        "OK   canonical ComfyUI port: "
        f"{CANONICAL_COMFYUI_PORT}"
    )


# ============================================================================
# 9. LIVE RUNTIME VERIFIER CONTRACT
# ============================================================================

def check_live_runtime_contract() -> None:

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

        "require_choice(",
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
        str(
            STALE_COMFYUI_PORT
        )
        not in source,
        "verify_live_runtime.py contains "
        "stale port 8219.",
    )

    # ------------------------------------------------------------------------
    # Locked model checks
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
            "Live verifier does not verify "
            "locked model:\n"
            f"{text}",
        )

    print(
        "OK   live runtime verifier contract"
    )

    print(
        "OK   live locked-model verification"
    )


# ============================================================================
# 10. CPU PREFLIGHT SELF-CONTRACT
# ============================================================================

def check_self_contract() -> None:

    source = read_text(
        Path(__file__)
    )

    # ------------------------------------------------------------------------
    # Lock must remain the source of truth.
    # ------------------------------------------------------------------------

    require(
        "compatibility_lock.yaml"
        in source,
        "CPU preflight must validate "
        "compatibility_lock.yaml.",
    )

    # ------------------------------------------------------------------------
    # Modern/legacy adapter contract.
    # ------------------------------------------------------------------------

    require(
        LEGACY_LATENT_LOADER
        in source,
        "CPU preflight must know the "
        "legacy latent loader.",
    )

    require(
        MODERN_LATENT_LOADER
        in source,
        "CPU preflight must know the "
        "modern latent loader.",
    )

    print(
        "OK   CPU preflight self-contract"
    )


# ============================================================================
# 11. PORT CONTRACT
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

    # ------------------------------------------------------------------------
    # Generator
    # ------------------------------------------------------------------------

    require(
        str(
            CANONICAL_COMFYUI_PORT
        )
        in generator,
        "generate_video.py does not "
        "declare canonical port 8188.",
    )

    # 8219 is permitted ONLY as the intentional stale guard.

    tree = parse_python(
        GENERATE_VIDEO
    )

    stale_guard_count = 0

    for node in ast.walk(
        tree
    ):

        if not isinstance(
            node,
            ast.FunctionDef,
        ):

            continue

        if node.name != (
            "validate_worker_url"
        ):

            continue

        function_source = (
            ast.get_source_segment(
                generator,
                node,
            )
        )

        if (
            function_source
            and
            str(
                STALE_COMFYUI_PORT
            )
            in function_source
        ):

            stale_guard_count += 1

    require(
        stale_guard_count == 1,
        "generate_video.py must contain "
        "exactly one stale-port guard "
        "for 8219.",
    )

    # ------------------------------------------------------------------------
    # ComfyUI launcher
    # ------------------------------------------------------------------------

    require(
        str(
            CANONICAL_COMFYUI_PORT
        )
        in start_comfyui,
        "start_comfyui.py does not "
        "use canonical port 8188.",
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
    # Live verifier
    # ------------------------------------------------------------------------

    require(
        str(
            CANONICAL_COMFYUI_PORT
        )
        in live,
        "verify_live_runtime.py does "
        "not contain canonical port 8188.",
    )

    require(
        str(
            STALE_COMFYUI_PORT
        )
        not in live,
        "verify_live_runtime.py contains "
        "stale port 8219.",
    )

    print(
        "OK   global ComfyUI port contract"
    )


# ============================================================================
# 12. VALIDATOR ALIGNMENT
# ============================================================================

def check_validator_alignment() -> None:

    source = read_text(
        VALIDATOR
    )

    required = {
        "scripts/cpu_preflight.py",
        "compatibility_lock.yaml",
        "LatentUpscaleModelLoader",
        "LTXVLatentUpsamplerModelLoader",
        "8188",
        "8219",
    }

    for text in (
        required
    ):

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

    print(
        "=" * 80
    )

    print(
        "LTX-13B CPU / REPOSITORY PREFLIGHT"
    )

    print(
        "=" * 80
    )

    print(
        f"Project root: {ROOT}"
    )

    print()

    # ------------------------------------------------------------------------
    # Static repository validation
    # ------------------------------------------------------------------------

    check_files()

    check_python_syntax()

    lock = load_lock()

    check_lock(
        lock
    )

    check_workflows()

    check_adapter()

    check_real_conversion()

    check_compatibility_builder()

    check_generate_video()

    check_kaggle_wiring()

    check_live_runtime_contract()

    check_self_contract()

    check_port_contract()

    check_validator_alignment()

    # ------------------------------------------------------------------------
    # Final
    # ------------------------------------------------------------------------

    print()

    print(
        "=" * 80
    )

    print(
        "✅ CPU / REPOSITORY PREFLIGHT PASSED"
    )

    print(
        "=" * 80
    )

    print()

    print(
        "Single source of truth:"
    )

    print(
        "  kaggle/compatibility_lock.yaml"
    )

    print()

    print(
        "Canonical ComfyUI runtime port:"
    )

    print(
        "  8188"
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
        "Legacy DETAILER loader:"
    )

    print(
        f"  {LEGACY_LATENT_LOADER}"
    )

    print()

    print(
        "Modern runtime loader:"
    )

    print(
        f"  {MODERN_LATENT_LOADER}"
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


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":

    main()
