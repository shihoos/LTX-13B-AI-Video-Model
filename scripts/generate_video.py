#!/usr/bin/env python3

"""
LTX-13B COMPLETE PRODUCTION VIDEO GENERATOR

Pipeline:

    Story
      ↓
    ProductionOrchestrator
      ↓
    ProductionManager
      ↓
    ProductionRunner
      ↓
    BASE workflow
      ↓
    IC-LoRA / spatial detailer
      ↓
    Final assembly

This script does NOT bootstrap ComfyUI.

Runtime startup belongs to:

    kaggle/launch.py

Canonical local ComfyUI port:

    8188

Example:

    python scripts/generate_video.py \
        --story "A cinematic story..." \
        --mode ai_story \
        --gpu-url 0=http://127.0.0.1:8188
"""

from __future__ import annotations

import argparse
import json
import sys

from pathlib import Path


# ======================================================================
# PROJECT PATH
# ======================================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# ======================================================================
# PROJECT IMPORTS
# ======================================================================

from execution.production_runner import (
    ProductionRunner,
)

from pipeline.production_orchestrator import (
    ProductionOrchestrator,
)

from pipeline.production_manager import (
    ProductionManager,
)


# ======================================================================
# WORKFLOWS
# ======================================================================

BASE_WORKFLOW = (
    PROJECT_ROOT
    / "workflows"
    / "baseline"
    / "ltxv-13b-dist-i2v-base.json"
)

DETAILER_WORKFLOW = (
    PROJECT_ROOT
    / "workflows"
    / "detailer"
    / "ltxv-13b-098-ic-lora-upscale.json"
)


# ======================================================================
# CONSTANTS
# ======================================================================

CANONICAL_COMFYUI_PORT = 8188
STALE_COMFYUI_PORT = 8219


# ======================================================================
# ARGUMENTS
# ======================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the complete LTX-13B production pipeline: "
            "planning → BASE → IC-LoRA/spatial detailer → assembly."
        )
    )

    parser.add_argument(
        "--story",
        required=True,
        help="Story or generation request.",
    )

    parser.add_argument(
        "--mode",
        default="ai_story",
        choices=[
            "ai_story",
            "preserve_user_story",
            "expand_user_story",
        ],
        help="Story planning mode.",
    )

    parser.add_argument(
        "--gpu-url",
        action="append",
        required=True,
        metavar="ID=URL",
        help=(
            "ComfyUI worker URL. Repeat for multiple GPUs. "
            "Canonical local endpoint is port 8188. "
            "Example: "
            "--gpu-url 0=http://127.0.0.1:8188"
        ),
    )

    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path for generation metadata.",
    )

    return parser.parse_args()


# ======================================================================
# GPU URL PARSING
# ======================================================================

def validate_worker_url(url: str) -> None:
    """
    Reject the known stale local ComfyUI port.

    Remote/tunnel URLs are allowed to use their own external
    port, so only the known stale local endpoints are rejected.
    """

    stale_urls = (
        "http://127.0.0.1:8219",
        "http://localhost:8219",
    )

    if url in stale_urls:
        raise ValueError(
            "Stale ComfyUI port detected.\n"
            f"Received: {url}\n"
            f"Canonical local ComfyUI port: "
            f"{CANONICAL_COMFYUI_PORT}"
        )


def parse_gpu_urls(
    values: list[str],
) -> dict[int, str]:

    result: dict[int, str] = {}

    for value in values:

        if "=" not in value:

            raise ValueError(
                "GPU URL must use ID=URL format:\n"
                f"{value}"
            )

        gpu_text, url = value.split(
            "=",
            1,
        )

        gpu_text = gpu_text.strip()
        url = url.strip().rstrip("/")

        try:

            gpu_id = int(
                gpu_text
            )

        except ValueError as error:

            raise ValueError(
                "GPU ID must be an integer:\n"
                f"{gpu_text}"
            ) from error

        if gpu_id < 0:

            raise ValueError(
                "GPU ID must be >= 0."
            )

        if not url.startswith(
            (
                "http://",
                "https://",
            )
        ):

            raise ValueError(
                "GPU URL must begin with "
                "http:// or https://:\n"
                f"{url}"
            )

        validate_worker_url(
            url
        )

        if gpu_id in result:

            raise ValueError(
                f"GPU ID {gpu_id} "
                "was specified more than once."
            )

        result[gpu_id] = url

    if not result:

        raise ValueError(
            "At least one --gpu-url is required."
        )

    return result


# ======================================================================
# ENVIRONMENT VALIDATION
# ======================================================================

def validate_environment() -> None:

    required_files = (
        BASE_WORKFLOW,
        DETAILER_WORKFLOW,
    )

    for path in required_files:

        if not path.is_file():

            raise FileNotFoundError(
                "Required workflow is missing:\n"
                f"{path}"
            )



# ======================================================================
# PRODUCTION PLAN VALIDATION
# ======================================================================

def validate_production_plan(
    production_plan,
) -> list:

    if not isinstance(
        production_plan,
        dict,
    ):

        raise RuntimeError(
            "ProductionOrchestrator returned "
            "an invalid production plan."
        )

    shots = production_plan.get(
        "shots"
    )

    if not isinstance(
        shots,
        list,
    ):

        raise RuntimeError(
            "Production plan does not contain "
            "a valid shots list."
        )

    if not shots:

        raise RuntimeError(
            "Production planner returned zero shots."
        )

    return shots


# ======================================================================
# MAIN
# ======================================================================

def main():

    args = parse_args()

    validate_environment()

    gpu_urls = parse_gpu_urls(
        args.gpu_url
    )

    print(
        "=" * 80
    )

    print(
        "LTX-13B COMPLETE PRODUCTION PIPELINE"
    )

    print(
        "=" * 80
    )

    print(
        f"Project: {PROJECT_ROOT}"
    )

    print(
        f"Story mode: {args.mode}"
    )

    print(
        "GPU workers:"
    )

    for gpu_id, url in sorted(
        gpu_urls.items()
    ):

        print(
            f"  GPU {gpu_id}: {url}"
        )

    # ==============================================================
    # STEP 1 — PLANNING
    # ==============================================================

    print()
    print(
        "=" * 80
    )

    print(
        "STEP 1 — PRODUCTION PLANNING"
    )

    print(
        "=" * 80
    )

    orchestrator = (
        ProductionOrchestrator()
    )

    try:

        production_plan = (
            orchestrator.create_production_plan(
                mode=args.mode,
                user_input=args.story,
            )
        )

    finally:

        orchestrator.unload_models()

    shots = validate_production_plan(
        production_plan
    )

    print(
        f"Production plan created: "
        f"{len(shots)} shots"
    )

    print(
        "Plan:",
        production_plan.get(
            "production_plan_path"
        ),
    )

    # ==============================================================
    # STEP 2 — PIPELINE CONFIGURATION
    # ==============================================================

    print()
    print(
        "=" * 80
    )

    print(
        "STEP 2 — PIPELINE CONFIGURATION"
    )

    print(
        "=" * 80
    )

    manager = (
        ProductionManager()
    )

    pipeline = (
        manager.get_pipeline()
    )

    if not pipeline:

        raise RuntimeError(
            "ProductionManager returned "
            "an empty pipeline."
        )

    print(
        "Configured pipeline:"
    )

    for stage in pipeline:

        print(
            f"  → {stage}"
        )

    # ==============================================================
    # STEP 3 — GPU EXECUTION
    # ==============================================================

    print()
    print(
        "=" * 80
    )

    print(
        "STEP 3 — GPU EXECUTION"
    )

    print(
        "=" * 80
    )

    runner = (
        ProductionRunner(
            project_root=PROJECT_ROOT,
            gpu_urls=gpu_urls,
            workflow_path=BASE_WORKFLOW,
            detailer_workflow_path=(
                DETAILER_WORKFLOW
            ),
        )
    )

    final_video = runner.run(
        production_plan
    )

    # ==============================================================
    # STEP 4 — RESULT VALIDATION
    # ==============================================================

    if final_video is None:

        raise RuntimeError(
            "ProductionRunner returned "
            "no final video path."
        )

    final_video = Path(
        final_video
    )

    if not final_video.exists():

        raise RuntimeError(
            "ProductionRunner reported success "
            "but the final video does not exist:\n"
            f"{final_video}"
        )

    if not final_video.is_file():

        raise RuntimeError(
            "Final video path is not a file:\n"
            f"{final_video}"
        )

    if final_video.stat().st_size <= 0:

        raise RuntimeError(
            "Final video exists but is empty:\n"
            f"{final_video}"
        )

    # ==============================================================
    # RESULT METADATA
    # ==============================================================

    result = {
        "status": "completed",
        "final_video": str(
            final_video
        ),
        "production_plan": (
            production_plan.get(
                "production_plan_path"
            )
        ),
        "shot_count": len(
            shots
        ),
        "gpu_workers": gpu_urls,
        "story_mode": args.mode,
        "comfyui_local_port": (
            CANONICAL_COMFYUI_PORT
        ),
    }

    if args.output_json is not None:

        output_path = Path(
            args.output_json
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path.write_text(
            json.dumps(
                result,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        print(
            f"Metadata: {output_path}"
        )

    # ==============================================================
    # FINAL
    # ==============================================================

    print()
    print(
        "=" * 80
    )

    print(
        "🎉 PRODUCTION COMPLETE"
    )

    print(
        "=" * 80
    )

    print(
        f"Final video:\n{final_video}"
    )

    print(
        f"Size: {final_video.stat().st_size:,} bytes"
    )


if __name__ == "__main__":
    main()
