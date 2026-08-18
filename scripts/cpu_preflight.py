#!/usr/bin/env python3
"""
LTX-13B CPU PREFLIGHT

Runs repository-level checks without:

- installing runtime packages
- starting ComfyUI
- loading models
- requiring CUDA
- consuming GPU

This script validates:

1. Project file structure
2. Python AST syntax
3. Python package structure
4. Internal project import graph
5. BASE workflow source structure
6. DETAILER workflow source structure
7. ComfyWorkflowAdapter structure
8. DETAILER source -> API conversion
9. Legacy -> modern latent loader conversion
10. Core pipeline class contracts
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path


# =====================================================================
# PATHS
# =====================================================================

ROOT = Path(__file__).resolve().parents[1]

PLANNER = ROOT / "planner"
PIPELINE = ROOT / "pipeline"
EXECUTION = ROOT / "execution"
SCHEDULER = ROOT / "scheduler"
SCHEMAS = ROOT / "schemas"
KAGGLE = ROOT / "kaggle"
COMPATIBILITY = ROOT / "compatibility"
SCRIPTS = ROOT / "scripts"
WORKFLOWS = ROOT / "workflows"

BASE_WORKFLOW = (
    WORKFLOWS
    / "baseline"
    / "ltxv-13b-dist-i2v-base.json"
)

DETAILER_WORKFLOW = (
    WORKFLOWS
    / "detailer"
    / "ltxv-13b-098-ic-lora-upscale.json"
)


# =====================================================================
# HELPERS
# =====================================================================

def fail(message: str) -> None:
    raise RuntimeError(message)


def require(
    condition: bool,
    message: str,
) -> None:
    if not condition:
        fail(message)


def read(path: Path) -> str:
    require(
        path.is_file(),
        f"Required file missing:\n{path}",
    )

    return path.read_text(
        encoding="utf-8",
    )


def parse(path: Path) -> ast.Module:
    source = read(path)

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


def classes(
    module: ast.Module,
) -> dict[str, ast.ClassDef]:

    result: dict[str, ast.ClassDef] = {}

    for node in module.body:
        if isinstance(
            node,
            ast.ClassDef,
        ):
            result[node.name] = node

    return result


def methods(
    class_node: ast.ClassDef,
) -> set[str]:

    result: set[str] = set()

    for node in class_node.body:
        if isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
            ),
        ):
            result.add(node.name)

    return result


def python_files() -> list[Path]:

    files: list[Path] = []

    for path in ROOT.rglob("*.py"):

        relative = path.relative_to(ROOT)

        if "__pycache__" in relative.parts:
            continue

        if ".git" in relative.parts:
            continue

        files.append(path)

    return sorted(files)


# =====================================================================
# 1. PROJECT FILE STRUCTURE
# =====================================================================

def check_project_files() -> None:

    required_files = [
        PLANNER / "config.py",
        PLANNER / "qwen_loader.py",
        PLANNER / "story_planner.py",
        PLANNER / "character_detector.py",
        PLANNER / "character_planner.py",
        PLANNER / "scene_planner.py",
        PLANNER / "shot_planner.py",

        PIPELINE / "continuity_manager.py",
        PIPELINE / "modes.py",
        PIPELINE / "production_manager.py",
        PIPELINE / "production_orchestrator.py",
        PIPELINE / "reference_manager.py",

        EXECUTION / "checkpoint_manager.py",
        EXECUTION / "comfy_client.py",
        EXECUTION / "comfy_workflow_adapter.py",
        EXECUTION / "shot_executor.py",
        EXECUTION / "assembly_manager.py",
        EXECUTION / "production_runner.py",

        SCHEDULER / "gpu_scheduler.py",
        SCHEDULER / "shot_queue.py",

        SCHEMAS / "character.py",
        SCHEMAS / "scene.py",
        SCHEMAS / "shot.py",
        SCHEMAS / "parser.py",

        KAGGLE / "bootstrap.py",
        KAGGLE / "config.py",
        KAGGLE / "launch.py",
        KAGGLE / "model_paths.yaml",
        KAGGLE / "preflight_modern.py",
        KAGGLE / "start_comfyui.py",
        KAGGLE / "start_comfyui_tunnel.py",

        COMPATIBILITY / "prepare_modern_ltx.py",

        SCRIPTS / "generate_video.py",

        BASE_WORKFLOW,
        DETAILER_WORKFLOW,
    ]

    for path in required_files:
        require(
            path.is_file(),
            f"Required project file missing:\n{path}",
        )

    print(
        f"OK   project files: "
        f"{len(required_files)}"
    )


# =====================================================================
# 2. PYTHON AST COMPILATION
# =====================================================================

def check_python_ast() -> None:

    files = python_files()

    for path in files:
        parse(path)

    print(
        f"OK   Python AST compilation: "
        f"{len(files)} files"
    )


# =====================================================================
# 3. PACKAGE STRUCTURE
# =====================================================================

def check_packages() -> None:

    packages = [
        EXECUTION,
        PIPELINE,
        PLANNER,
        SCHEMAS,
        SCHEDULER,
    ]

    for package in packages:

        init_file = (
            package
            / "__init__.py"
        )

        require(
            init_file.is_file(),
            "Package __init__.py missing:\n"
            f"{init_file}",
        )

    print(
        "OK   package __init__.py files"
    )


# =====================================================================
# 4. PROJECT IMPORT GRAPH
# =====================================================================

def check_import_graph() -> None:

    modules = [
        "planner.config",
        "planner.qwen_loader",
        "planner.story_planner",
        "planner.character_detector",
        "planner.character_planner",
        "planner.scene_planner",
        "planner.shot_planner",

        "pipeline.continuity_manager",
        "pipeline.modes",
        "pipeline.production_manager",
        "pipeline.production_orchestrator",
        "pipeline.reference_manager",

        "execution.checkpoint_manager",
        "execution.comfy_client",
        "execution.comfy_workflow_adapter",
        "execution.shot_executor",
        "execution.assembly_manager",
        "execution.production_runner",

        "scheduler.gpu_scheduler",
        "scheduler.shot_queue",

        "schemas.character",
        "schemas.scene",
        "schemas.shot",
        "schemas.parser",
    ]

    previous_path = list(sys.path)

    try:

        project_path = str(ROOT)

        if project_path not in sys.path:
            sys.path.insert(
                0,
                project_path,
            )

        for module_name in modules:

            parts = module_name.split(".")

            relative = (
                Path(*parts)
                .with_suffix(".py")
            )

            module_path = (
                ROOT
                / relative
            )

            require(
                module_path.is_file(),
                "Import graph module missing:\n"
                f"{module_name}\n"
                f"{module_path}",
            )

            parse(module_path)

    finally:
        sys.path[:] = previous_path

    print(
        f"OK   project import graph: "
        f"{len(modules)} modules"
    )


# =====================================================================
# 5. WORKFLOW SOURCE VALIDATION
# =====================================================================

def check_workflows() -> None:

    base = json.loads(
        read(BASE_WORKFLOW)
    )

    detailer = json.loads(
        read(DETAILER_WORKFLOW)
    )

    require(
        isinstance(base, dict),
        "BASE workflow root "
        "is not an object.",
    )

    require(
        isinstance(detailer, dict),
        "DETAILER workflow root "
        "is not an object.",
    )

    base_nodes = base.get(
        "nodes",
        [],
    )

    detailer_nodes = detailer.get(
        "nodes",
        [],
    )

    require(
        isinstance(base_nodes, list),
        "BASE workflow nodes "
        "is not a list.",
    )

    require(
        isinstance(detailer_nodes, list),
        "DETAILER workflow nodes "
        "is not a list.",
    )

    base_types = {
        node.get("type")
        for node in base_nodes
        if isinstance(node, dict)
    }

    detailer_types = {
        node.get("type")
        for node in detailer_nodes
        if isinstance(node, dict)
    }

    # -------------------------------------------------------------
    # BASE SOURCE WORKFLOW
    # -------------------------------------------------------------

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

    missing_base = (
        base_required
        - base_types
    )

    require(
        not missing_base,
        "BASE workflow missing:\n"
        + "\n".join(
            sorted(missing_base)
        ),
    )

    print(
        "OK   BASE source workflow"
    )

    # -------------------------------------------------------------
    # DETAILER SOURCE WORKFLOW
    #
    # IMPORTANT:
    #
    # The source workflow intentionally contains:
    #
    # LTXVLatentUpsamplerModelLoader
    #
    # This is converted by ComfyWorkflowAdapter into:
    #
    # LatentUpscaleModelLoader
    #
    # Therefore the SOURCE workflow must NOT be required to already
    # contain LatentUpscaleModelLoader.
    # -------------------------------------------------------------

    detailer_required_source = {
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

    missing_detailer = (
        detailer_required_source
        - detailer_types
    )

    require(
        not missing_detailer,
        "DETAILER source workflow missing:\n"
        + "\n".join(
            sorted(missing_detailer)
        ),
    )

    print(
        "OK   DETAILER source workflow"
    )


# =====================================================================
# 6. WORKFLOW ADAPTER STRUCTURE
# =====================================================================

def check_workflow_adapter_structure() -> None:

    adapter_path = (
        EXECUTION
        / "comfy_workflow_adapter.py"
    )

    module = parse(
        adapter_path
    )

    available_classes = classes(
        module
    )

    adapter_class = (
        available_classes.get(
            "ComfyWorkflowAdapter"
        )
    )

    require(
        adapter_class is not None,
        "ComfyWorkflowAdapter "
        "class is missing.",
    )

    required_methods = {
        "__init__",
        "to_api_workflow",
        "apply_modern_compatibility",
        "validate_modern_detailer",
    }

    available_methods = methods(
        adapter_class
    )

    missing = (
        required_methods
        - available_methods
    )

    require(
        not missing,
        "ComfyWorkflowAdapter missing methods:\n"
        + "\n".join(
            sorted(missing)
        ),
    )

    print(
        "OK   ComfyWorkflowAdapter contract"
    )


# =====================================================================
# 7. REAL DETAILER ADAPTER CONVERSION
# =====================================================================

def check_detailer_adapter_conversion() -> None:

    project_path = str(ROOT)

    previous_path = list(sys.path)

    try:

        if project_path not in sys.path:

            sys.path.insert(
                0,
                project_path,
            )

        from execution.comfy_workflow_adapter import (
            ComfyWorkflowAdapter,
        )

        adapter = (
            ComfyWorkflowAdapter(
                DETAILER_WORKFLOW
            )
        )

        detailer_api = (
            adapter.to_api_workflow()
        )

        require(
            isinstance(
                detailer_api,
                dict,
            ),
            "DETAILER adapter did not "
            "produce a dictionary.",
        )

        require(
            detailer_api,
            "DETAILER adapter produced "
            "an empty API workflow.",
        )

        adapter.validate_modern_detailer(
            detailer_api
        )

        api_types = {
            node.get("class_type")
            for node in detailer_api.values()
            if isinstance(
                node,
                dict,
            )
        }

        require(
            "LatentUpscaleModelLoader"
            in api_types,
            "DETAILER adapter conversion "
            "did not produce:\n"
            "LatentUpscaleModelLoader",
        )

        require(
            "LTXVLatentUpsamplerModelLoader"
            not in api_types,
            "Legacy loader survived "
            "DETAILER adapter conversion:\n"
            "LTXVLatentUpsamplerModelLoader",
        )

    except RuntimeError:
        raise

    except Exception as error:

        fail(
            "DETAILER adapter conversion failed:\n"
            f"{type(error).__name__}: "
            f"{error}"
        )

    finally:

        sys.path[:] = previous_path

    print(
        "OK   DETAILER source → API conversion"
    )

    print(
        "OK   legacy latent loader removed"
    )

    print(
        "OK   modern latent loader produced"
    )


# =====================================================================
# 8. CORE CLASS CONTRACTS
# =====================================================================

def check_core_contracts() -> None:

    contracts = {
        EXECUTION
        / "comfy_client.py": {
            "ComfyClient": {
                "__init__",
            },
        },

        EXECUTION
        / "shot_executor.py": {
            "ShotExecutor": {
                "__init__",
            },
        },

        EXECUTION
        / "production_runner.py": {
            "ProductionRunner": {
                "__init__",
            },
        },

        PIPELINE
        / "production_orchestrator.py": {
            "ProductionOrchestrator": {
                "__init__",
            },
        },

        PLANNER
        / "story_planner.py": {
            "StoryPlanner": {
                "__init__",
            },
        },
    }

    for path, class_contracts in contracts.items():

        module = parse(path)

        available_classes = classes(
            module
        )

        for (
            class_name,
            required_methods,
        ) in class_contracts.items():

            class_node = (
                available_classes.get(
                    class_name
                )
            )

            require(
                class_node is not None,
                "Required class missing:\n"
                f"{path}\n"
                f"{class_name}",
            )

            available_methods = methods(
                class_node
            )

            missing = (
                required_methods
                - available_methods
            )

            require(
                not missing,
                "Required methods missing:\n"
                f"{path}\n"
                f"{class_name}\n"
                + "\n".join(
                    sorted(missing)
                ),
            )

    print(
        "OK   core class contracts"
    )


# =====================================================================
# MAIN
# =====================================================================

def main() -> None:

    print(
        "=" * 80
    )

    print(
        "LTX-13B CPU PREFLIGHT"
    )

    print(
        "=" * 80
    )

    print()

    print(
        "This test does NOT:"
    )

    print(
        "  - install packages"
    )

    print(
        "  - start ComfyUI"
    )

    print(
        "  - load models"
    )

    print(
        "  - require CUDA"
    )

    print(
        "  - consume GPU"
    )

    print()

    # -------------------------------------------------------------
    # CHECK 1
    # -------------------------------------------------------------

    check_project_files()

    # -------------------------------------------------------------
    # CHECK 2
    # -------------------------------------------------------------

    check_python_ast()

    # -------------------------------------------------------------
    # CHECK 3
    # -------------------------------------------------------------

    check_packages()

    # -------------------------------------------------------------
    # CHECK 4
    # -------------------------------------------------------------

    check_import_graph()

    # -------------------------------------------------------------
    # CHECK 5
    # -------------------------------------------------------------

    check_workflows()

    # -------------------------------------------------------------
    # CHECK 6
    # -------------------------------------------------------------

    check_workflow_adapter_structure()

    # -------------------------------------------------------------
    # CHECK 7
    # -------------------------------------------------------------

    check_detailer_adapter_conversion()

    # -------------------------------------------------------------
    # CHECK 8
    # -------------------------------------------------------------

    check_core_contracts()

    print()

    print(
        "=" * 80
    )

    print(
        "CPU PREFLIGHT PASSED"
    )

    print(
        "=" * 80
    )

    print(
        "Repository structure: PASS"
    )

    print(
        "Python syntax: PASS"
    )

    print(
        "Package structure: PASS"
    )

    print(
        "Project import graph: PASS"
    )

    print(
        "BASE workflow: PASS"
    )

    print(
        "DETAILER source workflow: PASS"
    )

    print(
        "Workflow adapter contract: PASS"
    )

    print(
        "Legacy → modern loader conversion: PASS"
    )

    print(
        "Core pipeline contracts: PASS"
    )

    print()

    print(
        "No GPU was required."
    )


if __name__ == "__main__":
    main()
