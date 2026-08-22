#!/usr/bin/env python3

from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)


REQUIRED_FILES = [
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
    "execution/h3_workflow_builder.py",
    "execution/shot_executor.py",
    "execution/production_runner.py",

    "schemas/character.py",
    "schemas/parser.py",
    "schemas/scene.py",
    "schemas/shot.py",

    "scheduler/gpu_scheduler.py",
    "scheduler/shot_queue.py",

    "kaggle/bootstrap.py",
    "kaggle/h3_config.yaml",
    "kaggle/preflight_h3.py",
    "kaggle/start_comfyui.py",
    "kaggle/verify_live_runtime.py",

    "scripts/cpu_preflight.py",
    "scripts/generate_video.py",

    "workflows/MiniMax-H3/H3_Ref2VA_Memory_API.json",
]


FORBIDDEN = [
    "ltxv",
    "ic-lora",
    "ltx 0.9.8",
    "ltx-13b",
    "ltxv-13b",
]


def check_python():
    for path in ROOT.rglob("*.py"):

        if (
            ".git"
            in path.parts
        ):
            continue

        try:
            ast.parse(
                path.read_text(
                    encoding="utf-8"
                )
            )
        except SyntaxError as error:
            raise RuntimeError(
                f"Syntax error in {path}: {error}"
            ) from error


def check_files():
    missing = []

    for relative in REQUIRED_FILES:
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


def check_workflow_json():
    workflow = (
        ROOT
        / "workflows"
        / "MiniMax-H3"
        / "H3_Ref2VA_Memory_API.json"
    )

    data = json.loads(
        workflow.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(
        data,
        dict,
    ):
        raise RuntimeError(
            "H3 workflow must be a JSON object."
        )


def check_ltx_removed():
    for path in ROOT.rglob("*"):

        if not path.is_file():
            continue

        if ".git" in path.parts:
            continue

        try:
            text = (
                path.read_text(
                    encoding="utf-8",
                    errors="ignore",
                )
                .lower()
            )
        except Exception:
            continue

        for term in FORBIDDEN:

            if term in text:

                # This validator itself contains the legacy strings.
                if (
                    path
                    == ROOT
                    / "scripts"
                    / "validate_project.py"
                ):
                    continue

                raise RuntimeError(
                    f"Obsolete LTX term "
                    f"found in {path}: "
                    f"{term}"
                )


def main():

    print(
        "=" * 70
    )

    print(
        "MINIMAX H3 PROJECT VALIDATION"
    )

    print(
        "=" * 70
    )

    check_files()
    print(
        "OK  required files"
    )

    check_python()
    print(
        "OK  Python syntax"
    )

    check_workflow_json()
    print(
        "OK  workflow JSON"
    )

    check_ltx_removed()
    print(
        "OK  no obsolete LTX backend"
    )

    print(
        "\nH3 project validation PASSED."
    )


if __name__ == "__main__":
    main()
