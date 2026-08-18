#!/usr/bin/env python3

"""
LTX-13B CPU / REPOSITORY PREFLIGHT

This validation is intentionally CPU-only.

It does NOT:

- install packages
- start ComfyUI
- load models
- require CUDA
- require GPU
- require Kaggle dataset mounts

It validates the repository architecture before the
GPU/runtime bootstrap is allowed to modify anything.

Single source of truth:

    kaggle/compatibility_lock.yaml

The obsolete files below MUST NOT exist:

    kaggle/model_paths.yaml
    kaggle/runtime_requirements.lock
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

LOCK_FILE = ROOT / "kaggle" / "compatibility_lock.yaml"

BASE_WORKFLOW = (
    ROOT
    / "workflows"
    / "baseline"
    / "ltxv-13b-dist-i2v-base.json"
)

DETAILER_WORKFLOW = (
    ROOT
    / "workflows"
    / "detailer"
    / "ltxv-13b-098-ic-lora-upscale.json"
)

ADAPTER = (
    ROOT
    / "execution"
    / "comfy_workflow_adapter.py"
)

COMPATIBILITY_BUILDER = (
    ROOT
    / "compatibility"
    / "prepare_modern_ltx.py"
)

GENERATE_VIDEO = (
    ROOT
    / "scripts"
    / "generate_video.py"
)

VALIDATOR = (
    ROOT
    / "scripts"
    / "validate_project.py"
)

LIVE_RUNTIME = (
    ROOT
    / "kaggle"
    / "verify_live_runtime.py"
)

OBSOLETE_FILES = {
    ROOT / "kaggle" / "model_paths.yaml",
    ROOT / "kaggle" / "runtime_requirements.lock",
}


def fail(message: str) -> None:
    raise RuntimeError(message)


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def read_text(path: Path) -> str:
    require(
        path.is_file(),
        f"Required file missing:\n{path}",
    )

    return path.read_text(
        encoding="utf-8",
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
            f"Line {error.lineno}: {error.msg}"
        )


def parse_json(path: Path) -> dict:
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


def python_files() -> list[Path]:
    result = []

    for path in ROOT.rglob("*.py"):

        relative = path.relative_to(ROOT)

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


def node_types(workflow: dict) -> set[str]:
    nodes = workflow.get("nodes", [])

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


# ============================================================
# 1. FILE STRUCTURE
# ============================================================

def check_files() -> None:

    required = [
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
        "kaggle/verify_live_runtime.py",

        "compatibility/prepare_modern_ltx.py",

        "scripts/cpu_preflight.py",
        "scripts/generate_video.py",
        "scripts/validate_project.py",

        "workflows/baseline/ltxv-13b-dist-i2v-base.json",
        "workflows/detailer/ltxv-13b-098-ic-lora-upscale.json",
    ]

    for relative in required:

        path = ROOT / relative

        require(
            path.is_file(),
            f"Required file missing:\n{path}",
        )

    for obsolete in OBSOLETE_FILES:

        require(
            not obsolete.exists(),
            "Obsolete configuration still exists:\n"
            f"{obsolete}\n\n"
            "Do NOT recreate this file. "
            "compatibility_lock.yaml is the single source of truth.",
        )

    print(
        f"OK   repository files: {len(required)}"
    )


# ============================================================
# 2. PYTHON SYNTAX
# ============================================================

def check_python_syntax() -> None:

    files = python_files()

    for path in files:
        parse_python(path)

    print(
        f"OK   Python AST syntax: {len(files)} files"
    )


# ============================================================
# 3. LOCK FILE
# ============================================================

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
            "Could not parse compatibility_lock.yaml:\n"
            f"{error}"
        )

    require(
        isinstance(data, dict),
        "compatibility_lock.yaml must contain a mapping.",
    )

    return data


def check_lock(lock: dict) -> None:

    required_sections = {
        "comfyui",
        "python_runtime",
        "custom_nodes",
        "legacy_ltx_098_compat",
        "detailer_compat",
        "models",
        "validation",
    }

    missing = required_sections - set(lock)

    require(
        not missing,
        "compatibility_lock.yaml missing sections:\n"
        + "\n".join(sorted(missing)),
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
        "lock.custom_nodes must be a mapping.",
    )

    commit = str(
        comfy.get("commit", "")
    )

    require(
        len(commit) == 40
        and all(
            character in "0123456789abcdef"
            for character in commit.lower()
        ),
        "ComfyUI lock commit must be a 40-character SHA.",
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
        + "\n".join(sorted(missing_models)),
    )

    for name in required_models:

        spec = models[name]

        require(
            isinstance(spec, dict),
            f"models.{name} must be a mapping.",
        )

        for key in (
            "dataset",
            "filename",
            "target",
        ):

            require(
                isinstance(spec.get(key), str)
                and spec[key].strip(),
                f"models.{name}.{key} is missing.",
            )

    print(
        "OK   compatibility lock"
    )


# ============================================================
# 4. WORKFLOWS
# ============================================================

def check_workflows() -> None:

    base = parse_json(BASE_WORKFLOW)
    detailer = parse_json(DETAILER_WORKFLOW)

    base_types = node_types(base)
    detailer_types = node_types(detailer)

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
        + "\n".join(sorted(missing)),
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
        + "\n".join(sorted(missing)),
    )

    require(
        "LatentUpscaleModelLoader"
        not in detailer_types,
        "DETAILER source workflow unexpectedly contains "
        "LatentUpscaleModelLoader.\n"
        "The source workflow must use the legacy loader; "
        "the adapter performs the conversion.",
    )

    print(
        "OK   BASE workflow"
    )

    print(
        "OK   DETAILER source workflow"
    )


# ============================================================
# 5. WORKFLOW ADAPTER
# ============================================================

def check_adapter() -> None:

    module = parse_python(ADAPTER)

    classes = {
        node.name
        for node in ast.walk(module)
        if isinstance(node, ast.ClassDef)
    }

    require(
        "ComfyWorkflowAdapter" in classes,
        "ComfyWorkflowAdapter class missing.",
    )

    source = read_text(ADAPTER)

    required_strings = {
        "LTXVLatentUpsamplerModelLoader",
        "LatentUpscaleModelLoader",
        "apply_modern_compatibility",
        "validate_modern_detailer",
    }

    for text in required_strings:

        require(
            text in source,
            f"ComfyWorkflowAdapter missing contract:\n{text}",
        )

    print(
        "OK   ComfyWorkflowAdapter contract"
    )


def check_real_conversion() -> None:

    previous_path = list(sys.path)

    try:

        root_string = str(ROOT)

        if root_string not in sys.path:
            sys.path.insert(0, root_string)

        from execution.comfy_workflow_adapter import (
            ComfyWorkflowAdapter,
        )

        adapter = ComfyWorkflowAdapter(
            DETAILER_WORKFLOW
        )

        api = adapter.to_api_workflow()

        require(
            isinstance(api, dict) and api,
            "DETAILER conversion produced empty API workflow.",
        )

        adapter.validate_modern_detailer(api)

        types = {
            node.get("class_type")
            for node in api.values()
            if isinstance(node, dict)
        }

        require(
            "LatentUpscaleModelLoader" in types,
            "DETAILER conversion did not produce "
            "LatentUpscaleModelLoader.",
        )

        require(
            "LTXVLatentUpsamplerModelLoader" not in types,
            "Legacy LTX latent loader survived conversion.",
        )

    except Exception as error:

        fail(
            "Real DETAILER adapter conversion failed:\n"
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


# ============================================================
# 6. COMPATIBILITY BUILDER
# ============================================================

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
        "LTX098ModernCompat",
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


# ============================================================
# 7. APPLICATION WIRING
# ============================================================

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
    }

    for text in required:

        require(
            text in source,
            "generate_video.py missing:\n"
            f"{text}",
        )

    require(
        "8219" not in source,
        "Stale ComfyUI port 8219 exists in generate_video.py.\n"
        "Use 8188.",
    )

    print(
        "OK   production application wiring"
    )


# ============================================================
# 8. KAGGLE WIRING
# ============================================================

def check_kaggle_wiring() -> None:

    launch = read_text(
        ROOT / "kaggle" / "launch.py"
    )

    bootstrap = read_text(
        ROOT / "kaggle" / "bootstrap.py"
    )

    tunnel = read_text(
        ROOT / "kaggle" / "start_comfyui_tunnel.py"
    )

    preflight = read_text(
        ROOT / "kaggle" / "preflight_modern.py"
    )

    require(
        "preflight_modern.py" in launch
        or "preflight_modern.py" in launch.replace("\\", "/"),
        "launch.py is not wired to preflight_modern.py.",
    )

    require(
        "bootstrap.py" in launch,
        "launch.py is not wired to bootstrap.py.",
    )

    require(
        "start_comfyui_tunnel.py" in launch,
        "launch.py is not wired to start_comfyui_tunnel.py.",
    )

    require(
        "compatibility_lock.yaml" in bootstrap,
        "bootstrap.py is not lock-driven.",
    )

    require(
        "model_paths.yaml" not in bootstrap,
        "bootstrap.py still references obsolete model_paths.yaml.",
    )

    require(
        "8188" in tunnel,
        "ComfyUI tunnel launcher is not using canonical port 8188.",
    )

    require(
        "compatibility_lock.yaml" in preflight,
        "preflight_modern.py is not lock-driven.",
    )

    print(
        "OK   Kaggle bootstrap/launcher wiring"
    )


# ============================================================
# 9. CPU PREFLIGHT SELF-CONTRACT
# ============================================================

def check_self_contract() -> None:

    source = read_text(
        Path(__file__)
    )

    require(
        "model_paths.yaml" in source,
        "CPU preflight must explicitly protect against "
        "obsolete model_paths.yaml.",
    )

    require(
        "compatibility_lock.yaml" in source,
        "CPU preflight must validate compatibility_lock.yaml.",
    )

    print(
        "OK   CPU preflight contract"
    )


# ============================================================
# 10. FINAL
# ============================================================

def main() -> None:

    print("=" * 80)
    print("LTX-13B CPU / REPOSITORY PREFLIGHT")
    print("=" * 80)

    check_files()
    check_python_syntax()

    lock = load_lock()

    check_lock(lock)
    check_workflows()
    check_adapter()
    check_real_conversion()
    check_compatibility_builder()
    check_generate_video()
    check_kaggle_wiring()
    check_self_contract()

    print()
    print("=" * 80)
    print("✅ CPU / REPOSITORY PREFLIGHT PASSED")
    print("=" * 80)

    print(
        "Single source of truth: "
        "kaggle/compatibility_lock.yaml"
    )

    print(
        "Obsolete model_paths.yaml: NOT USED"
    )

    print(
        "Canonical ComfyUI runtime port: 8188"
    )


if __name__ == "__main__":
    main()
