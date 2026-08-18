from __future__ import annotations

import ast
import importlib.metadata
import json
import shutil
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCK_FILE = PROJECT_ROOT / "kaggle" / "compatibility_lock.yaml"


def fail(message: str) -> None:
    raise RuntimeError(message)


def source(path: Path) -> str:
    if not path.is_file():
        fail(f"Missing file:\n{path}")
    try:
        return path.read_text(encoding="utf-8")
    except OSError as error:
        fail(f"Unable to read file:\n{path}\n{error}")


def tree(path: Path) -> ast.Module:
    try:
        return ast.parse(source(path), filename=str(path))
    except SyntaxError as error:
        fail(f"Python syntax error:\n{path}\n{error}")


def names(module: ast.AST) -> set[str]:
    result: set[str] = set()
    for node in ast.walk(module):
        if isinstance(node, ast.Name):
            result.add(node.id)
        elif isinstance(node, ast.Attribute):
            result.add(node.attr)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            result.add(node.name)
        elif isinstance(node, ast.alias):
            result.add(node.asname or node.name.rsplit(".", 1)[-1])
    return result


def constants(module: ast.AST) -> set[str]:
    # AST never includes comments.  Exclude docstrings, but only from nodes
    # whose body is actually a statement list (Lambda.body is an expression).
    docstrings: set[int] = set()
    for node in ast.walk(module):
        body = getattr(node, "body", None)
        if not isinstance(body, list) or not body:
            continue
        first = body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            docstrings.add(id(first.value))
    return {
        node.value
        for node in ast.walk(module)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstrings
    }


def classes(module: ast.AST) -> dict[str, ast.ClassDef]:
    return {
        node.name: node
        for node in ast.walk(module)
        if isinstance(node, ast.ClassDef)
    }


def methods(class_node: ast.ClassDef) -> set[str]:
    return {
        node.name
        for node in class_node.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def require_subset(actual: set[str], required: set[str], label: str) -> None:
    missing = required - actual
    if missing:
        fail(f"{label} is missing:\n" + "\n".join(sorted(missing)))


def assigned_template(module: ast.AST, function_name: str, variable: str) -> str:
    function = next(
        (
            node
            for node in ast.walk(module)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == function_name
        ),
        None,
    )
    if function is None:
        fail(f"Missing function: {function_name}")
    for node in ast.walk(function):
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == variable for target in node.targets):
            continue
        value = node.value
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Attribute)
            and value.func.attr == "lstrip"
            and isinstance(value.func.value, ast.Constant)
        ):
            value = value.func.value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            return value.value
    fail(f"{function_name} does not assign string template {variable!r}.")


def has_call(module: ast.AST, receiver: str, method: str) -> bool:
    for node in ast.walk(module):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != method:
            continue
        value = node.func.value
        if isinstance(value, ast.Name) and value.id == receiver:
            return True
        if isinstance(value, ast.Attribute) and value.attr == receiver:
            return True
    return False


def read_lock() -> dict:
    try:
        import yaml
    except ImportError as error:
        fail(f"PyYAML unavailable:\n{error}")
    try:
        data = yaml.safe_load(source(LOCK_FILE))
    except Exception as error:
        fail(f"Invalid compatibility lock:\n{error}")
    if not isinstance(data, dict):
        fail("compatibility_lock.yaml must contain a mapping.")
    require_subset(
        set(data),
        {
            "comfyui",
            "python_runtime",
            "custom_nodes",
            "legacy_ltx_098_compat",
            "detailer_compat",
            "models",
            "validation",
        },
        "compatibility_lock.yaml",
    )
    return data


def validate_files() -> None:
    required = [
        "planner/config.py", "planner/qwen_loader.py", "planner/story_planner.py",
        "planner/character_detector.py", "planner/character_planner.py",
        "planner/scene_planner.py", "planner/shot_planner.py",
        "pipeline/continuity_manager.py", "pipeline/modes.py",
        "pipeline/production_manager.py", "pipeline/production_orchestrator.py",
        "pipeline/reference_manager.py", "execution/checkpoint_manager.py",
        "execution/comfy_client.py", "execution/comfy_workflow_adapter.py",
        "execution/shot_executor.py", "execution/assembly_manager.py",
        "execution/production_runner.py", "scheduler/gpu_scheduler.py",
        "scheduler/shot_queue.py", "schemas/character.py", "schemas/scene.py",
        "schemas/shot.py", "schemas/parser.py", "kaggle/bootstrap.py",
        "kaggle/config.py", "kaggle/launch.py", "kaggle/model_paths.yaml",
        "kaggle/preflight_modern.py", "kaggle/start_comfyui.py",
        "kaggle/start_comfyui_tunnel.py", "compatibility/prepare_modern_ltx.py",
        "scripts/generate_video.py",
        "workflows/baseline/ltxv-13b-dist-i2v-base.json",
        "workflows/detailer/ltxv-13b-098-ic-lora-upscale.json",
    ]
    for relative in required:
        path = PROJECT_ROOT / relative
        source(path)
        print(f"OK   {path}")
    for package in ("execution", "pipeline", "planner", "schemas", "scheduler"):
        source(PROJECT_ROOT / package / "__init__.py")
    if (PROJECT_ROOT / "kaggle" / "runtime_requirements.lock").exists():
        fail("Obsolete kaggle/runtime_requirements.lock must be removed.")


def validate_workflows() -> None:
    checks = {
        "workflows/baseline/ltxv-13b-dist-i2v-base.json": {
            "LTXVBaseSampler", "LTXVConditioning", "STGGuiderAdvanced",
            "FloatToSigmas", "StringToFloatList", "UnetLoaderGGUF",
            "CLIPLoaderGGUF", "VAELoader", "Set VAE Decoder Noise",
            "VHS_VideoCombine",
        },
        "workflows/detailer/ltxv-13b-098-ic-lora-upscale.json": {
            "VHS_LoadVideo", "LTXVLoopingSampler", "LTXVLatentUpsampler",
            "LTXVLatentUpsamplerModelLoader", "LTXVTiledVAEDecode",
            "LTXVFilmGrain", "LoraLoaderModelOnly", "VHS_VideoCombine",
        },
    }
    for relative, required in checks.items():
        path = PROJECT_ROOT / relative
        try:
            data = json.loads(source(path))
        except json.JSONDecodeError as error:
            fail(f"Invalid workflow JSON:\n{path}\n{error}")
        if not isinstance(data, dict):
            fail(f"Workflow root must be an object:\n{path}")
        node_types = {
            node.get("type")
            for node in data.get("nodes", [])
            if isinstance(node, dict)
        }
        require_subset(node_types, required, relative)


def validate_compatibility(lock: dict) -> None:
    module = tree(PROJECT_ROOT / "compatibility" / "prepare_modern_ltx.py")
    require_subset(
        names(module) | constants(module),
        {"load_lock", "get_legacy_commit", "patch_blur", "write_curated_init",
         "build_compat_package", "compatibility_lock.yaml"},
        "compatibility builder",
    )
    initializer = ast.parse(assigned_template(module, "write_curated_init", "init_code"))
    # LTXVConditioning belongs to the modern ComfyUI-LTXVideo installation;
    # it is checked in the BASE workflow/live-node preflight, not this legacy package.
    legacy_nodes = {
        "LTXVBaseSampler", "LTXVLoopingSampler", "LTXVTiledSampler",
        "LTXVTiledVAEDecode", "LTXVLatentUpsampler",
        "LTXVLatentUpsamplerModelLoader", "LTXVFilmGrain",
        "STGGuiderAdvanced", "Set VAE Decoder Noise",
    }
    require_subset(constants(initializer), legacy_nodes, "generated legacy initializer")
    patch = ast.parse(assigned_template(module, "patch_blur", "replacement"))
    blur = next(
        (node for node in ast.walk(patch) if isinstance(node, ast.FunctionDef) and node.name == "blur_internal"),
        None,
    )
    if blur is None or [arg.arg for arg in blur.args.args] != ["image", "blur_radius"]:
        fail("Generated blur patch must define blur_internal(image, blur_radius).")
    if not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "conv2d"
        for node in ast.walk(blur)
    ):
        fail("Generated blur patch does not use native torch conv2d.")


def validate_application() -> None:
    adapter = tree(PROJECT_ROOT / "execution" / "comfy_workflow_adapter.py")
    adapter_class = classes(adapter).get("ComfyWorkflowAdapter")
    if adapter_class is None:
        fail("ComfyWorkflowAdapter class is missing.")
    require_subset(
        methods(adapter_class),
        {"__init__", "to_api_workflow", "apply_modern_compatibility", "set_prompt",
         "set_negative_prompt", "set_seed", "set_filename_prefix", "set_input_image",
         "set_input_video", "validate_modern_detailer", "apply_shot"},
        "ComfyWorkflowAdapter",
    )
    require_subset(
        constants(adapter),
        {"LTXVLatentUpsamplerModelLoader", "LatentUpscaleModelLoader",
         "VHS_LoadVideo", "VHS_VideoCombine", "CLIPTextEncode", "LoadImage"},
        "ComfyWorkflowAdapter node configuration",
    )
    required_classes = {
        "execution/comfy_client.py": ("ComfyClient", {"health_check", "queue_prompt", "get_history", "wait_for_prompt", "download_file", "find_video_outputs"}),
        "execution/production_runner.py": ("ProductionRunner", {"__init__", "prepare", "run", "_run_one_shot", "_dict_to_shot"}),
        "scheduler/gpu_scheduler.py": ("GPUScheduler", {"run"}),
        "execution/checkpoint_manager.py": ("CheckpointManager", {"initialize_shot", "mark_generating", "mark_raw_complete", "mark_detailer_complete", "mark_upscaled_complete", "mark_complete", "mark_failed", "set_assembly_started", "set_assembly_complete", "set_assembly_failed", "get_assembly"}),
        "pipeline/production_orchestrator.py": ("ProductionOrchestrator", {"create_production_plan", "unload_models"}),
        "pipeline/production_manager.py": ("ProductionManager", {"get_pipeline"}),
    }
    for relative, (class_name, required) in required_classes.items():
        module = tree(PROJECT_ROOT / relative)
        class_node = classes(module).get(class_name)
        if class_node is None:
            fail(f"{class_name} class is missing in {relative}.")
        require_subset(methods(class_node), required, class_name)
    runner = tree(PROJECT_ROOT / "execution" / "production_runner.py")
    if not has_call(runner, "scheduler", "run"):
        fail("ProductionRunner does not call self.scheduler.run(...).")
    entry = tree(PROJECT_ROOT / "scripts" / "generate_video.py")
    require_subset(names(entry), {"ProductionOrchestrator", "ProductionRunner", "ProductionManager"}, "generate_video.py")
    require_subset(constants(entry), {"--story", "--mode", "--gpu-url"}, "generate_video.py arguments")
    if not has_call(entry, "orchestrator", "create_production_plan"):
        fail("generate_video.py does not call orchestrator.create_production_plan(...).")
    if not has_call(entry, "runner", "run"):
        fail("generate_video.py does not call runner.run(...).")


def validate_runtime(lock: dict) -> None:
    comfy = PROJECT_ROOT / "ComfyUI"
    if not comfy.exists():
        print("SKIP runtime checks: ComfyUI has not been materialized.")
        return
    expected = lock["python_runtime"]
    try:
        import torch
        import torchvision
    except ImportError as error:
        fail(f"Materialized runtime cannot import torch/torchvision:\n{error}")
    if torch.__version__ != expected["torch"] or torchvision.__version__ != expected["torchvision"]:
        fail("Torch runtime does not match compatibility_lock.yaml.")
    for package in (
        lock["comfyui"]["frontend"]["package"],
        lock["comfyui"]["workflow_templates"]["package"],
        lock["comfyui"]["embedded_docs"]["package"],
    ):
        try:
            importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            fail(f"Materialized runtime is missing package: {package}")
    print("OK   materialized runtime lock")


def main() -> None:
    print("=" * 80)
    print("LTX-13B PROJECT VALIDATION")
    print("=" * 80)
    lock = read_lock()
    validate_files()
    validate_workflows()
    validate_compatibility(lock)
    validate_application()
    validate_runtime(lock)
    if shutil.which("ffmpeg") is None:
        print("WARNING ffmpeg not found in current environment.")
    print("✅ PROJECT VALIDATION PASSED")


if __name__ == "__main__":
    main()
