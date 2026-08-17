import importlib
import shutil
import sys

from pathlib import Path


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]


def check_file(
    path: Path,
):

    if not path.exists():

        raise RuntimeError(
            f"Missing: {path}"
        )

    print(
        f"OK   {path}"
    )


def check_import(
    module_name: str,
):

    try:

        importlib.import_module(
            module_name
        )

    except Exception as error:

        raise RuntimeError(
            f"Import failed: "
            f"{module_name}\n"
            f"{error}"
        ) from error

    print(
        f"OK   import {module_name}"
    )


def main():

    print("=" * 70)
    print(
        "LTX-13B PROJECT STRUCTURAL VALIDATION"
    )
    print("=" * 70)

    required_files = [
        "planner/config.py",
        "planner/qwen_loader.py",
        "planner/story_planner.py",
        "planner/character_detector.py",
        "planner/character_planner.py",
        "planner/scene_planner.py",
        "planner/shot_planner.py",
        "pipeline/continuity_manager.py",
        "pipeline/production_orchestrator.py",
        "execution/checkpoint_manager.py",
        "execution/comfy_client.py",
        "execution/comfy_workflow_adapter.py",
        "execution/shot_executor.py",
        "execution/assembly_manager.py",
        "execution/production_runner.py",
        "scheduler/gpu_scheduler.py",
        "scheduler/shot_queue.py",
        "workflows/baseline/ltxv-13b-dist-i2v-base.json",
    ]

    print()
    print(
        "FILE CHECK"
    )

    for relative in (
        required_files
    ):

        check_file(
            PROJECT_ROOT
            / relative
        )

    print()
    print(
        "IMPORT CHECK"
    )

    modules = [
        "schemas.character",
        "schemas.scene",
        "schemas.shot",
        "schemas.parser",
        "pipeline.modes",
        "pipeline.reference_manager",
        "pipeline.continuity_manager",
        "planner.config",
        "planner.qwen_loader",
        "planner.story_planner",
        "planner.character_detector",
        "planner.character_planner",
        "planner.scene_planner",
        "planner.shot_planner",
        "pipeline.production_orchestrator",
        "execution.checkpoint_manager",
        "execution.comfy_client",
        "execution.comfy_workflow_adapter",
        "execution.shot_executor",
        "execution.assembly_manager",
        "execution.production_runner",
        "scheduler.gpu_scheduler",
        "scheduler.shot_queue",
    ]

    for module in modules:

        check_import(
            module
        )

    print()
    print(
        "QWEN DATASET CHECK"
    )

    qwen_path = Path(
        "/kaggle/input/"
        "qwen3-4b-instruct-2507"
    )

    check_file(
        qwen_path
        / "config.json"
    )

    check_file(
        qwen_path
        / "model.safetensors.index.json"
    )

    shards = list(
        qwen_path.glob(
            "model-*.safetensors"
        )
    )

    if len(shards) != 3:

        raise RuntimeError(
            "Expected 3 Qwen safetensor shards, "
            f"found {len(shards)}"
        )

    print(
        "OK   Qwen 3 model shards"
    )

    print()
    print(
        "FFMPEG CHECK"
    )

    if (
        shutil.which(
            "ffmpeg"
        )
        is None
    ):

        raise RuntimeError(
            "FFmpeg is not installed."
        )

    print(
        "OK   ffmpeg"
    )

    print()
    print(
        "GPU CHECK"
    )

    try:

        import torch

        count = (
            torch.cuda.device_count()
        )

        print(
            f"CUDA GPUs detected: "
            f"{count}"
        )

        if count < 2:

            print(
                "WARNING: two-GPU execution "
                "cannot be tested on this runtime."
            )

    except Exception as error:

        raise RuntimeError(
            f"PyTorch/CUDA check failed: "
            f"{error}"
        ) from error

    print()
    print(
        "=" * 70
    )

    print(
        "STRUCTURAL VALIDATION PASSED"
    )

    print(
        "No Qwen model was loaded."
    )

    print(
        "No LTX generation was started."
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":

    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

    main()
