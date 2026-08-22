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
    "pipeline/modes.py",
    "pipeline/production_manager.py",
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

    "workflows/MiniMax-H3/H3_Ref2VA_Memory_API.json",
    "workflows/MiniMax-H3/H3_Ref2VA_Native_API.json",
]


FORBIDDEN_TEXT = [
    "LTX-13B",
    "LTXVLatentUpsamplerModelLoader",
    "ltxv-13b",
    "legacy_ltx_098",
    "detailer_compat",
]


def validate_files():

    missing = []

    for relative in REQUIRED:

        if not (
            ROOT / relative
        ).is_file():

            missing.append(
                relative
            )

    if missing:

        raise RuntimeError(
            "Missing required files:\n"
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

        if (
            "ComfyUI"
            in path.parts
        ):
            continue

        if (
            "__pycache__"
            in path.parts
        ):
            continue

        source = path.read_text(
            encoding="utf-8"
        )

        ast.parse(
            source,
            filename=str(path),
        )


def validate_json():

    workflow_root = (
        ROOT
        / "workflows"
        / "MiniMax-H3"
    )

    for path in workflow_root.glob(
        "*.json"
    ):

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
                f"Workflow root must be object: "
                f"{path}"
            )


def validate_h3_contract():

    text = (
        ROOT
        / "execution"
        / "h3_workflow_builder.py"
    ).read_text(
        encoding="utf-8"
    )

    required_terms = [
        "MiniMaxH3ReferenceToVideo",
        "H3FreeTextEncoder",
        "ref_images.ref_image_",
        "ref_videos.ref_video_",
        "ref_video_audios.ref_video_audio_",
        "ref_audios.ref_audio_",
        "H3MultishotMemorySampler",
        "VAEDecodeAudio",
    ]

    for term in required_terms:

        if term not in text:

            raise RuntimeError(
                "Missing H3 contract term: "
                + term
            )


def validate_no_legacy_ltx():

    for path in ROOT.rglob("*"):

        if not path.is_file():
            continue

        if ".git" in path.parts:
            continue

        if path.suffix.lower() not in {
            ".py",
            ".yaml",
            ".yml",
            ".json",
            ".md",
            ".txt",
        }:
            continue

        text = path.read_text(
            encoding="utf-8",
            errors="ignore",
        )

        lowered = text.lower()

        for term in FORBIDDEN_TEXT:

            if term.lower() in lowered:

                if (
                    path.name
                    == "validate_project.py"
                ):
                    continue

                raise RuntimeError(
                    f"Legacy LTX text found in "
                    f"{path}: {term}"
                )


def main():

    validate_files()

    validate_python()

    validate_json()

    validate_h3_contract()

    validate_no_legacy_ltx()

    print(
        "MiniMax H3 project validation PASSED."
    )


if __name__ == "__main__":
    main()
