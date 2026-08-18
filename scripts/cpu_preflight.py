from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def fail(message: str) -> None:
    raise RuntimeError(message)


def read(path: Path) -> str:
    if not path.is_file():
        fail(f"Missing required file:\n{path}")
    return path.read_text(encoding="utf-8")


def parse(path: Path) -> ast.Module:
    try:
        return ast.parse(read(path), filename=str(path))
    except SyntaxError as error:
        fail(f"Syntax error in:\n{path}\n{error}")


def classes(module: ast.AST) -> dict[str, ast.ClassDef]:
    return {
        node.name: node
        for node in ast.walk(module)
        if isinstance(node, ast.ClassDef)
    }


def functions(module: ast.AST) -> set[str]:
    return {
        node.name
        for node in ast.walk(module)
        if isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef),
        )
    }


def methods(class_node: ast.ClassDef) -> set[str]:
    return {
        node.name
        for node in class_node.body
        if isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef),
        )
    }


def strings(module: ast.AST) -> set[str]:
    return {
        node.value
        for node in ast.walk(module)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
    }


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def check_files() -> None:
    required = [
        "kaggle/compatibility_lock.yaml",
        "kaggle/bootstrap.py",
        "kaggle/config.py",
        "kaggle/launch.py",
        "kaggle/model_paths.yaml",
        "kaggle/preflight_modern.py",
        "kaggle/start_comfyui.py",
        "kaggle/start_comfyui_tunnel.py",
        "compatibility/prepare_modern_ltx.py",

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

        "planner/__init__.py",

        "scripts/validate_project.py",
        "scripts/generate_video.py",

        "workflows/baseline/ltxv-13b-dist-i2v-base.json",
        "workflows/detailer/ltxv-13b-098-ic-lora-upscale.json",
    ]

    for relative in required:
        path = ROOT / relative
        read(path)

    obsolete = ROOT / "kaggle" / "runtime_requirements.lock"

    require(
        not obsolete.exists(),
        "Obsolete kaggle/runtime_requirements.lock still exists.",
    )

    print(f"OK   project files: {len(required)}")


def check_python() -> None:
    python_files = sorted(ROOT.rglob("*.py"))

    excluded = {
        ROOT / ".git",
        ROOT / "ComfyUI",
        ROOT / ".runtime_ltx098",
    }

    checked = 0

    for path in python_files:
        if any(
            excluded_path in path.parents
            for excluded_path in excluded
        ):
            continue

        parse(path)
        checked += 1

    print(f"OK   Python AST compilation: {checked} files")


def check_package_initializers() -> None:
    expected = {
        "execution": '"""LTX-13B AI Video Model execution package."""',
        "pipeline": '"""LTX-13B AI Video Model package."""',
        "planner": '"""LTX-13B AI Video Model package."""',
        "scheduler": None,
        "schemas": None,
    }

    for package, expected_text in expected.items():
        path = ROOT / package / "__init__.py"
        text = read(path)

        require(
            text.strip(),
            f"{package}/__init__.py is empty.",
        )

        if expected_text is not None:
            require(
                expected_text in text,
                f"{package}/__init__.py has unexpected content.",
            )

    print("OK   package __init__.py files")


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

    env = {
        **dict(__import__("os").environ),
        "PYTHONPATH": str(ROOT),
    }

    for module in modules:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                f"import {module}",
            ],
            cwd=ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        if result.returncode != 0:
            fail(
                f"CPU import failed:\n"
                f"{module}\n\n"
                f"{result.stdout}"
            )

    print(f"OK   project import graph: {len(modules)} modules")


def check_workflows() -> None:
    base_path = (
        ROOT
        / "workflows"
        / "baseline"
        / "ltxv-13b-dist-i2v-base.json"
    )

    detailer_path = (
        ROOT
        / "workflows"
        / "detailer"
        / "ltxv-13b-098-ic-lora-upscale.json"
    )

    base = json.loads(read(base_path))
    detailer = json.loads(read(detailer_path))

    require(
        isinstance(base, dict),
        "BASE workflow root is not an object.",
    )

    require(
        isinstance(detailer, dict),
        "DETAILER workflow root is not an object.",
    )

    base_types = {
        node.get("type")
        for node in base.get("nodes", [])
        if isinstance(node, dict)
    }

    detailer_types = {
        node.get("type")
        for node in detailer.get("nodes", [])
        if isinstance(node, dict)
    }

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

    detailer_required = {
        "VHS_LoadVideo",
        "LTXVLoopingSampler",
        "LTXVLatentUpsampler",
        "LTXVLatentUpsamplerModelLoader",
        "LTXVTiledVAEDecode",
        "LTXVFilmGrain",
        "LoraLoaderModelOnly",
        "LatentUpscaleModelLoader",
        "VHS_VideoCombine",
    }

    missing = base_required - base_types

    require(
        not missing,
        "BASE workflow missing:\n"
        + "\n".join(sorted(missing)),
    )

    missing = detailer_required - detailer_types

    require(
        not missing,
        "DETAILER workflow missing:\n"
        + "\n".join(sorted(missing)),
    )

    print("OK   BASE workflow structure")
    print("OK   DETAILER workflow structure")


def check_adapter() -> None:
    path = (
        ROOT
        / "execution"
        / "comfy_workflow_adapter.py"
    )

    module = parse(path)

    adapter = classes(module).get(
        "ComfyWorkflowAdapter"
    )

    require(
        adapter is not None,
        "ComfyWorkflowAdapter class missing.",
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

    missing = required_methods - methods(adapter)

    require(
        not missing,
        "ComfyWorkflowAdapter missing methods:\n"
        + "\n".join(sorted(missing)),
    )

    text = read(path)

    require(
        "LTXVLatentUpsamplerModelLoader" in text,
        "Legacy latent-loader compatibility constant missing.",
    )

    require(
        "LatentUpscaleModelLoader" in text,
        "Modern latent-loader compatibility missing.",
    )

    print("OK   workflow adapter contract")


def check_compatibility() -> None:
    path = (
        ROOT
        / "compatibility"
        / "prepare_modern_ltx.py"
    )

    module = parse(path)

    required_functions = {
        "load_lock",
        "get_legacy_commit",
        "ensure_repo",
        "patch_blur",
        "write_curated_init",
        "build_compat_package",
    }

    missing = (
        required_functions
        - functions(module)
    )

    require(
        not missing,
        "Compatibility builder missing:\n"
        + "\n".join(sorted(missing)),
    )

    text = read(path)

    require(
        "def blur_internal(image, blur_radius):"
        in text,
        "Native blur implementation missing.",
    )

    require(
        "torch.nn.functional"
        in text,
        "Native Torch blur is missing.",
    )

    require(
        "F.conv2d"
        in text,
        "Native Gaussian blur does not use conv2d.",
    )

    require(
        "Blur().blur("
        not in text,
        "Legacy Blur().blur() call remains.",
    )

    print("OK   modern compatibility + native blur")


def check_runner_chain() -> None:
    runner_path = (
        ROOT
        / "execution"
        / "production_runner.py"
    )

    runner_module = parse(runner_path)

    runner = classes(
        runner_module
    ).get("ProductionRunner")

    require(
        runner is not None,
        "ProductionRunner missing.",
    )

    required_runner_methods = {
        "__init__",
        "prepare",
        "run",
        "_run_one_shot",
        "_dict_to_shot",
    }

    missing = (
        required_runner_methods
        - methods(runner)
    )

    require(
        not missing,
        "ProductionRunner missing:\n"
        + "\n".join(sorted(missing)),
    )

    runner_text = read(runner_path)

    require(
        "self.scheduler.run("
        in runner_text,
        "ProductionRunner is not wired to GPUScheduler.",
    )

    require(
        "ShotExecutor("
        in runner_text,
        "ProductionRunner is not wired to ShotExecutor.",
    )

    orchestrator_path = (
        ROOT
        / "pipeline"
        / "production_orchestrator.py"
    )

    orchestrator = parse(
        orchestrator_path
    )

    orchestrator_class = classes(
        orchestrator
    ).get("ProductionOrchestrator")

    require(
        orchestrator_class is not None,
        "ProductionOrchestrator missing.",
    )

    require(
        {
            "create_production_plan",
            "unload_models",
        }.issubset(
            methods(orchestrator_class)
        ),
        "ProductionOrchestrator contract is incomplete.",
    )

    print("OK   planner → orchestrator → runner → scheduler")


def check_shot_executor() -> None:
    path = (
        ROOT
        / "execution"
        / "shot_executor.py"
    )

    module = parse(path)

    executor = classes(
        module
    ).get("ShotExecutor")

    require(
        executor is not None,
        "ShotExecutor missing.",
    )

    text = read(path)

    required_strings = {
        "LTXVBaseSampler",
        "LTXVLoopingSampler",
        "LTXVLatentUpsampler",
        "LoraLoaderModelOnly",
        "VHS_LoadVideo",
        "VHS_VideoCombine",
    }

    missing = {
        value
        for value in required_strings
        if value not in text
    }

    require(
        not missing,
        "ShotExecutor missing integration:\n"
        + "\n".join(sorted(missing)),
    )

    require(
        "physical"
        in text.lower(),
        "BASE → DETAILER physical handoff is not represented.",
    )

    print("OK   BASE → physical video → IC-LoRA → detailer")


def check_resolution_policy() -> None:
    files = [
        ROOT / "execution" / "shot_executor.py",
        ROOT / "execution" / "comfy_workflow_adapter.py",
        ROOT / "kaggle" / "config.py",
        ROOT / "workflows" / "baseline"
        / "ltxv-13b-dist-i2v-base.json",
    ]

    combined = "\n".join(
        read(path)
        for path in files
    )

    require(
        "1536" in combined,
        "1536 master width is missing.",
    )

    require(
        "864" in combined,
        "864 master height is missing.",
    )

    print("OK   1536x864 16:9 master policy")


def check_frontend_policy() -> None:
    forbidden = [
        "--front",
        "--frontend-version",
        "frontend_version",
        "frontend_package",
    ]

    paths = [
        ROOT / "kaggle" / "launch.py",
        ROOT / "kaggle" / "start_comfyui.py",
        ROOT / "kaggle" / "start_comfyui_tunnel.py",
    ]

    for path in paths:
        text = read(path)

        for token in forbidden:
            require(
                token not in text,
                f"{path} contains frontend override: {token}",
            )

    print("OK   frontend policy")


def check_generate_entrypoint() -> None:
    path = ROOT / "scripts" / "generate_video.py"

    module = parse(path)
    text = read(path)

    require(
        "ProductionOrchestrator" in text,
        "generate_video.py missing ProductionOrchestrator.",
    )

    require(
        "ProductionRunner" in text,
        "generate_video.py missing ProductionRunner.",
    )

    require(
        "create_production_plan" in text,
        "generate_video.py does not create production plan.",
    )

    require(
        ".run(" in text,
        "generate_video.py does not execute runner.",
    )

    print("OK   story → production generation entrypoint")


def main() -> None:
    print("=" * 80)
    print("LTX-13B CPU PREFLIGHT")
    print("=" * 80)
    print()
    print("This test does NOT:")
    print("  - install packages")
    print("  - start ComfyUI")
    print("  - load models")
    print("  - require CUDA")
    print("  - consume GPU")
    print()

    check_files()
    check_python()
    check_package_initializers()
    check_import_graph()
    check_workflows()
    check_adapter()
    check_compatibility()
    check_runner_chain()
    check_shot_executor()
    check_resolution_policy()
    check_frontend_policy()
    check_generate_entrypoint()

    print()
    print("=" * 80)
    print("✅ CPU PREFLIGHT PASSED")
    print("=" * 80)
    print()
    print("Repository structure, imports, workflow contracts,")
    print("modern compatibility, blur, 16:9 policy, and")
    print("planner → orchestrator → executor wiring are valid.")
    print()
    print("NEXT: GPU runtime verification.")
    print("=" * 80)


if __name__ == "__main__":
    main()
