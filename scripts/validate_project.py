from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)


REQUIRED = [
    "planner/config.py",
    "planner/qwen_loader.py",
    "planner/story_planner.py",
    "planner/character_detector.py",
    "planner/character_planner.py",
    "planner/scene_planner.py",
    "planner/shot_planner.py",

    "pipeline/continuity_manager.py",
    "pipeline/identity_continuity.py",
    "pipeline/production_orchestrator.py",
    "pipeline/reference_manager.py",

    "execution/assembly_manager.py",
    "execution/checkpoint_manager.py",
    "execution/comfy_client.py",
    "execution/h3_runtime.py",
    "execution/h3_workflow_builder.py",
    "execution/shot_executor.py",
    "execution/production_runner.py",

    "scheduler/gpu_scheduler.py",
    "scheduler/shot_queue.py",

    "schemas/character.py",
    "schemas/parser.py",
    "schemas/scene.py",
    "schemas/shot.py",

    "kaggle/bootstrap.py",
    "kaggle/compatibility_lock.yaml",
    "kaggle/h3_config.yaml",
    "kaggle/preflight_h3.py",
    "kaggle/start_comfyui.py",
    "kaggle/verify_live_runtime.py",

    "scripts/cpu_preflight.py",
    "scripts/generate_video.py",
]


WORKFLOW_ROOT = (
    ROOT
    / "workflows"
    / "MiniMax-H3"
)


REQUIRED_WORKFLOWS = [
    WORKFLOW_ROOT
    / "base"
    / "H3_HardMode_R2V.json",

    WORKFLOW_ROOT
    / "base"
    / "H3_HardMode_Chained.json",
]


FORBIDDEN = {
    "LTX-13B",
    "LTXVLatentUpsamplerModelLoader",
    "ltxv-13b",
    "legacy_ltx_098",
    "detailer_compat",
}


def validate_required_files():

    missing = []

    for relative in REQUIRED:

        if not (
            ROOT
            / relative
        ).is_file():

            missing.append(
                relative
            )

    if missing:
        raise RuntimeError(
            "Missing required project files:\n"
            + "\n".join(
                missing
            )
        )


def validate_python():

    for path in ROOT.rglob(
        "*.py"
    ):

        if ".git" in path.parts:
            continue

        if "ComfyUI" in path.parts:
            continue

        if "__pycache__" in path.parts:
            continue

        source = path.read_text(
            encoding="utf-8"
        )

        ast.parse(
            source,
            filename=str(path),
        )


def validate_json():

    workflow_files = list(
        WORKFLOW_ROOT.rglob(
            "*.json"
        )
    )

    if not workflow_files:
        raise RuntimeError(
            "No H3 workflow JSON files found."
        )

    for path in workflow_files:

        data = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        if not isinstance(
            data,
            dict,
        ):
            raise RuntimeError(
                f"Workflow root is not an object: "
                f"{path}"
            )


def validate_h3_builder():

    path = (
        ROOT
        / "execution"
        / "h3_workflow_builder.py"
    )

    text = path.read_text(
        encoding="utf-8"
    )

    required_terms = [
        "MiniMaxH3ReferenceToVideo",
        "H3MultishotSampler",
        "H3LastFrame",
        "H3ConcatAV",
        "H3ModelLoaderAny",
        "H3ClipLoaderAny",
        "ref_images.ref_image_",
        "ref_videos.ref_video_",
        "ref_video_audios.ref_video_audio_",
        "ref_audios.ref_audio_",
        "/workflow/convert",
    ]

    for term in required_terms:

        if term not in text:
            raise RuntimeError(
                "H3 builder is missing required "
                f"contract term: {term}"
            )


def validate_no_legacy():

    for path in ROOT.rglob("*"):

        if not path.is_file():
            continue

        if ".git" in path.parts:
            continue

        if "ComfyUI" in path.parts:
            continue

        if path.suffix.lower() not in {
            ".py",
            ".json",
            ".yaml",
            ".yml",
            ".txt",
            ".md",
        }:
            continue

        text = path.read_text(
            encoding="utf-8",
            errors="ignore",
        )

        lowered = text.lower()

        for forbidden in FORBIDDEN:

            if forbidden.lower() in lowered:
                raise RuntimeError(
                    "Legacy LTX reference found in "
                    f"{path}: {forbidden}"
                )


def main():

    validate_required_files()
    validate_python()
    validate_json()
    validate_h3_builder()
    validate_no_legacy()

    print(
        "MiniMax H3 project validation PASSED."
    )


if __name__ == "__main__":
    main()
