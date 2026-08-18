#!/usr/bin/env python3
"""Repository-level LTX-13B validation gate."""
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
KAGGLE_DIR = PROJECT_ROOT / "kaggle"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
WORKFLOWS_DIR = PROJECT_ROOT / "workflows"
LOCK_FILE = KAGGLE_DIR / "compatibility_lock.yaml"
CPU_PREFLIGHT = SCRIPTS_DIR / "cpu_preflight.py"
LIVE_RUNTIME = KAGGLE_DIR / "verify_live_runtime.py"
BOOTSTRAP = KAGGLE_DIR / "bootstrap.py"
LAUNCH = KAGGLE_DIR / "launch.py"
PREFLIGHT_MODERN = KAGGLE_DIR / "preflight_modern.py"
TUNNEL = KAGGLE_DIR / "start_comfyui_tunnel.py"
GENERATE_VIDEO = SCRIPTS_DIR / "generate_video.py"
ADAPTER = PROJECT_ROOT / "execution" / "comfy_workflow_adapter.py"
COMPATIBILITY_BUILDER = PROJECT_ROOT / "compatibility" / "prepare_modern_ltx.py"
BASE_WORKFLOW = WORKFLOWS_DIR / "baseline" / "ltxv-13b-dist-i2v-base.json"
DETAILER_WORKFLOW = WORKFLOWS_DIR / "detailer" / "ltxv-13b-098-ic-lora-upscale.json"

CANONICAL_COMFYUI_PORT = 8188
STALE_COMFYUI_PORT = 8219
LEGACY_LATENT_LOADER = "LTXVLatentUpsamplerModelLoader"
MODERN_LATENT_LOADER = "LatentUpscaleModelLoader"

OBSOLETE_FILES = (KAGGLE_DIR / "model_paths.yaml", KAGGLE_DIR / "runtime_requirements.lock")

REQUIRED_FILES = (
    "planner/config.py", "planner/qwen_loader.py", "planner/story_planner.py",
    "planner/character_detector.py", "planner/character_planner.py", "planner/scene_planner.py",
    "planner/shot_planner.py", "pipeline/__init__.py", "pipeline/continuity_manager.py",
    "pipeline/modes.py", "pipeline/production_manager.py", "pipeline/production_orchestrator.py",
    "pipeline/reference_manager.py", "execution/__init__.py", "execution/checkpoint_manager.py",
    "execution/comfy_client.py", "execution/comfy_workflow_adapter.py", "execution/shot_executor.py",
    "execution/assembly_manager.py", "execution/production_runner.py", "scheduler/__init__.py",
    "scheduler/gpu_scheduler.py", "scheduler/shot_queue.py", "schemas/__init__.py", "schemas/character.py",
    "schemas/scene.py", "schemas/shot.py", "schemas/parser.py", "kaggle/compatibility_lock.yaml",
    "kaggle/bootstrap.py", "kaggle/config.py", "kaggle/launch.py", "kaggle/preflight_modern.py",
    "kaggle/start_comfyui.py", "kaggle/start_comfyui_tunnel.py", "kaggle/verify_live_runtime.py",
    "compatibility/prepare_modern_ltx.py", "scripts/cpu_preflight.py", "scripts/generate_video.py",
    "scripts/validate_project.py", "workflows/baseline/ltxv-13b-dist-i2v-base.json",
    "workflows/detailer/ltxv-13b-098-ic-lora-upscale.json",
)


def fail(message: str) -> None:
    raise RuntimeError(message)


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def read_text(path: Path) -> str:
    require(path.is_file(), f"Required file missing:\n{path}")
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        fail(f"Could not read file:\n{path}\n{exc}")


def parse_python(path: Path) -> ast.Module:
    try:
        return ast.parse(read_text(path), filename=str(path))
    except SyntaxError as exc:
        fail(f"Python syntax error:\n{path}\nLine {exc.lineno}: {exc.msg}")


def parse_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(read_text(path))
    except json.JSONDecodeError as exc:
        fail(f"Invalid JSON:\n{path}\n{exc}")
    require(isinstance(value, dict), f"JSON root must be an object:\n{path}")
    return value


def all_python_files() -> list[Path]:
    result = []
    for path in PROJECT_ROOT.rglob("*.py"):
        rel = path.relative_to(PROJECT_ROOT)
        if ".git" in rel.parts or "ComfyUI" in rel.parts or ".runtime_ltx098" in rel.parts or "__pycache__" in rel.parts:
            continue
        result.append(path)
    return sorted(result)


def node_types(workflow: dict[str, Any]) -> set[str]:
    nodes = workflow.get("nodes", [])
    require(isinstance(nodes, list), "Workflow nodes must be a list.")
    return {n.get("type") for n in nodes if isinstance(n, dict) and n.get("type")}


def load_lock() -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        fail(f"PyYAML is required:\n{exc}")
    try:
        data = yaml.safe_load(read_text(LOCK_FILE))
    except Exception as exc:
        fail(f"Invalid compatibility_lock.yaml:\n{exc}")
    require(isinstance(data, dict), "compatibility_lock.yaml must contain a mapping.")
    return data


def validate_files() -> None:
    for rel in REQUIRED_FILES:
        require((PROJECT_ROOT / rel).is_file(), f"Required project file missing:\n{PROJECT_ROOT / rel}")
    for path in OBSOLETE_FILES:
        require(not path.exists(), f"OBSOLETE FILE DETECTED:\n{path}\nRemove it. compatibility_lock.yaml is the single source of truth.")
    print("OK   repository file structure")


def validate_python() -> None:
    files = all_python_files()
    for path in files:
        parse_python(path)
    print(f"OK   Python AST syntax: {len(files)} files")


def validate_lock(lock: dict[str, Any]) -> None:
    required_sections = {"comfyui", "python_runtime", "custom_nodes", "legacy_ltx_098_compat", "detailer_compat", "models", "validation"}
    missing = required_sections - set(lock)
    require(not missing, "compatibility_lock.yaml missing sections:\n" + "\n".join(sorted(missing)))
    comfy = lock["comfyui"]
    models = lock["models"]
    require(isinstance(comfy, dict), "lock.comfyui must be a mapping.")
    require(isinstance(models, dict), "lock.models must be a mapping.")
    commit = str(comfy.get("commit", ""))
    require(len(commit) == 40 and all(c in "0123456789abcdef" for c in commit.lower()), "ComfyUI commit must be a 40-character hexadecimal SHA.")
    required_models = {"ltx_q4", "t5_q4", "vae", "ic_lora", "spatial_upscaler"}
    missing_models = required_models - set(models)
    require(not missing_models, "compatibility_lock.yaml missing models:\n" + "\n".join(sorted(missing_models)))
    for name in sorted(required_models):
        spec = models[name]
        require(isinstance(spec, dict), f"models.{name} must be a mapping.")
        for field in ("dataset", "filename", "target"):
            value = spec.get(field)
            require(isinstance(value, str) and value.strip(), f"models.{name}.{field} is missing.")
    print("OK   compatibility lock")


def validate_model_architecture() -> None:
    source = read_text(BOOTSTRAP)
    require("compatibility_lock.yaml" in source, "bootstrap.py is not lock-driven.")
    require("model_paths.yaml" not in source, "bootstrap.py still references obsolete model_paths.yaml.")
    print("OK   single-source model architecture")


def validate_workflows() -> None:
    base_types = node_types(parse_json(BASE_WORKFLOW))
    detailer_types = node_types(parse_json(DETAILER_WORKFLOW))
    base_required = {"LTXVBaseSampler", "LTXVConditioning", "STGGuiderAdvanced", "FloatToSigmas", "StringToFloatList", "UnetLoaderGGUF", "CLIPLoaderGGUF", "VAELoader", "Set VAE Decoder Noise", "VHS_VideoCombine"}
    missing = base_required - base_types
    require(not missing, "BASE workflow missing nodes:\n" + "\n".join(sorted(missing)))
    detailer_required = {"VHS_LoadVideo", "LTXVConditioning", "STGGuiderAdvanced", "FloatToSigmas", "StringToFloatList", "LTXVLoopingSampler", "LTXVLatentUpsampler", LEGACY_LATENT_LOADER, "LTXVTiledVAEDecode", "LTXVFilmGrain", "LoraLoaderModelOnly", "VHS_VideoCombine"}
    missing = detailer_required - detailer_types
    require(not missing, "DETAILER source workflow missing nodes:\n" + "\n".join(sorted(missing)))
    require(MODERN_LATENT_LOADER not in detailer_types, "DETAILER source workflow already contains modern LatentUpscaleModelLoader. The source must retain the legacy loader.")
    print("OK   BASE workflow")
    print("OK   DETAILER source workflow")


def validate_adapter() -> None:
    module = parse_python(ADAPTER)
    classes = {n.name for n in ast.walk(module) if isinstance(n, ast.ClassDef)}
    require("ComfyWorkflowAdapter" in classes, "ComfyWorkflowAdapter class missing.")
    source = read_text(ADAPTER)
    for text in (LEGACY_LATENT_LOADER, MODERN_LATENT_LOADER, "apply_modern_compatibility", "validate_modern_detailer"):
        require(text in source, f"ComfyWorkflowAdapter missing contract:\n{text}")
    print("OK   ComfyWorkflowAdapter contract")


def validate_real_detailer_conversion() -> None:
    old_path = list(sys.path)
    try:
        root = str(PROJECT_ROOT)
        if root not in sys.path:
            sys.path.insert(0, root)
        module = importlib.import_module("execution.comfy_workflow_adapter")
        adapter_class = getattr(module, "ComfyWorkflowAdapter", None)
        require(adapter_class is not None, "ComfyWorkflowAdapter import failed.")
        adapter = adapter_class(DETAILER_WORKFLOW)
        api = adapter.to_api_workflow()
        require(isinstance(api, dict) and api, "DETAILER adapter produced an empty API workflow.")
        adapter.validate_modern_detailer(api)
        types = {n.get("class_type") for n in api.values() if isinstance(n, dict)}
        require(MODERN_LATENT_LOADER in types, "DETAILER conversion did not produce LatentUpscaleModelLoader.")
        require(LEGACY_LATENT_LOADER not in types, "Legacy latent loader survived DETAILER conversion.")
    except RuntimeError:
        raise
    except Exception as exc:
        fail(f"DETAILER conversion failed: {type(exc).__name__}: {exc}")
    finally:
        sys.path[:] = old_path
    print("OK   DETAILER graph -> API conversion")
    print("OK   legacy latent loader removed")
    print("OK   modern latent loader produced")


def validate_compatibility_builder() -> None:
    source = read_text(COMPATIBILITY_BUILDER)
    for text in ("compatibility_lock.yaml", "legacy_ltx_098_compat", "runtime_package", "LTX098ModernCompat", "blur_internal"):
        require(text in source, f"Compatibility builder missing contract:\n{text}")
    print("OK   compatibility builder contract")


def validate_production_application() -> None:
    source = read_text(GENERATE_VIDEO)
    for text in ("ProductionRunner", "ProductionOrchestrator", "ProductionManager", "BASE_WORKFLOW", "DETAILER_WORKFLOW", "gpu-url"):
        require(text in source, f"generate_video.py missing contract:\n{text}")
    require(str(CANONICAL_COMFYUI_PORT) in source, "generate_video.py does not declare canonical ComfyUI port 8188.")
    # 8219 is allowed only as an explicit stale-endpoint rejection guard.
    require(str(STALE_COMFYUI_PORT) in source, "generate_video.py no longer contains the stale-port rejection guard.")
    try:
        tree = parse_python(GENERATE_VIDEO)
        stale_literals = [n for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str) and str(STALE_COMFYUI_PORT) in n.value]
        require(stale_literals, "generate_video.py stale-port guard is not represented as a string literal.")
        for node in stale_literals:
            value = node.value
            require("8219" in value and ("127.0.0.1:8219" in value or "localhost:8219" in value), "Unexpected 8219 reference in generate_video.py.")
    except RuntimeError:
        raise
    print("OK   production application wiring")
    print("OK   stale ComfyUI endpoint rejection guard")


def validate_kaggle_wiring() -> None:
    launch = read_text(LAUNCH)
    bootstrap = read_text(BOOTSTRAP)
    preflight = read_text(PREFLIGHT_MODERN)
    tunnel = read_text(TUNNEL)
    for name in ("preflight_modern.py", "bootstrap.py", "start_comfyui_tunnel.py"):
        require(name in launch, f"launch.py is not wired to {name}.")
    require("compatibility_lock.yaml" in bootstrap, "bootstrap.py is not lock-driven.")
    require("model_paths.yaml" not in bootstrap, "bootstrap.py references obsolete model_paths.yaml.")
    require("compatibility_lock.yaml" in preflight, "preflight_modern.py is not lock-driven.")
    require(str(STALE_COMFYUI_PORT) not in tunnel, "start_comfyui_tunnel.py contains stale port 8219.")
    require(str(CANONICAL_COMFYUI_PORT) in tunnel, "start_comfyui_tunnel.py does not use canonical port 8188.")
    print("OK   Kaggle launcher/bootstrap wiring")


def validate_cpu_preflight() -> None:
    source = read_text(CPU_PREFLIGHT)
    require("compatibility_lock.yaml" in source, "cpu_preflight.py is not lock-driven.")
    require("model_paths.yaml" in source, "cpu_preflight.py must protect against obsolete model_paths.yaml.")
    require("runtime_requirements.lock" in source, "cpu_preflight.py must protect against obsolete runtime_requirements.lock.")
    print("OK   CPU preflight contract")


def validate_live_runtime_verifier() -> None:
    source = read_text(LIVE_RUNTIME)
    required = {"127.0.0.1:8188", "/system_stats", "/object_info", "LTXVBaseSampler", "LTXVConditioning", "STGGuiderAdvanced", "UnetLoaderGGUF", "CLIPLoaderGGUF", "VAELoader", "VHS_VideoCombine", "LatentUpscaleModelLoader", LEGACY_LATENT_LOADER}
    for text in required:
        require(text in source, f"verify_live_runtime.py missing contract:\n{text}")
    require("8219" not in source, "verify_live_runtime.py still contains stale port 8219.")
    print("OK   live runtime verifier contract")


def validate_port_contract() -> None:
    for path in (TUNNEL, LIVE_RUNTIME):
        source = read_text(path)
        require(str(STALE_COMFYUI_PORT) not in source, f"Stale ComfyUI port detected:\n{path}")
        require(str(CANONICAL_COMFYUI_PORT) in source, f"Canonical ComfyUI port 8188 missing:\n{path}")
    # generate_video.py is checked semantically above because it intentionally contains 8219 as a guard.
    require(str(CANONICAL_COMFYUI_PORT) in read_text(GENERATE_VIDEO), "Canonical ComfyUI port 8188 missing from generate_video.py.")
    print("OK   global ComfyUI port contract: 8188")


def http_json(url: str, timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read()
    except (urllib.error.URLError, urllib.error.HTTPError) as exc:
        fail(f"Live ComfyUI request failed:\n{url}\n{exc}")
    try:
        data = json.loads(payload.decode("utf-8"))
    except json.JSONDecodeError as exc:
        fail(f"Live ComfyUI returned invalid JSON:\n{url}\n{exc}")
    require(isinstance(data, dict), f"Live response is not an object:\n{url}")
    return data


def validate_live_runtime(url: str) -> None:
    print(f"Checking live ComfyUI: {url}")
    http_json(f"{url}/system_stats", 15)
    print("OK   /system_stats")
    object_info = http_json(f"{url}/object_info", 30)
    required = {"LTXVBaseSampler", "LTXVConditioning", "STGGuiderAdvanced", "FloatToSigmas", "StringToFloatList", "UnetLoaderGGUF", "CLIPLoaderGGUF", "VAELoader", "VHS_VideoCombine", "VHS_LoadVideo", "LTXVLoopingSampler", "LTXVLatentUpsampler", "LTXVTiledVAEDecode", "LTXVFilmGrain", "LoraLoaderModelOnly", MODERN_LATENT_LOADER, LEGACY_LATENT_LOADER}
    missing = required - set(object_info)
    require(not missing, "Live ComfyUI missing nodes:\n" + "\n".join(sorted(missing)))
    print(f"OK   live required nodes: {len(required)}")
    print("OK   legacy compatibility node")
    print("OK   modern latent upscaler node")


def validate_cuda() -> None:
    try:
        import torch
    except ImportError as exc:
        fail(f"PyTorch is not importable:\n{exc}")
    require(torch.cuda.is_available(), "CUDA is not available.")
    count = torch.cuda.device_count()
    require(count > 0, "CUDA reports zero devices.")
    print(f"OK   CUDA available: {count} GPU(s)")
    for i in range(count):
        print(f"     GPU {i}: {torch.cuda.get_device_name(i)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the complete LTX-13B repository and runtime contract.")
    parser.add_argument("--require-runtime", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--runtime-url", default="http://127.0.0.1:8188")
    args = parser.parse_args()
    if args.require_cuda:
        args.require_runtime = True
    print("=" * 80)
    print("LTX-13B PROJECT VALIDATION")
    print("=" * 80)
    print(f"Project root: {PROJECT_ROOT}\n")
    validate_files()
    validate_python()
    lock = load_lock()
    validate_lock(lock)
    validate_model_architecture()
    validate_workflows()
    validate_adapter()
    validate_real_detailer_conversion()
    validate_compatibility_builder()
    validate_production_application()
    validate_kaggle_wiring()
    validate_cpu_preflight()
    validate_live_runtime_verifier()
    validate_port_contract()
    if args.require_runtime:
        print("\n" + "=" * 80)
        print("LIVE COMFYUI VALIDATION")
        print("=" * 80)
        validate_live_runtime(args.runtime_url.rstrip("/"))
    if args.require_cuda:
        print("\n" + "=" * 80)
        print("CUDA VALIDATION")
        print("=" * 80)
        validate_cuda()
    print("\n" + "=" * 80)
    print("🎉 LTX-13B VALIDATION PASSED")
    print("=" * 80)
    print("Single source of truth: kaggle/compatibility_lock.yaml")
    print("Canonical ComfyUI port: 8188")
    print("NO VIDEO WAS GENERATED BY THIS VALIDATOR.")


if __name__ == "__main__":
    main()
