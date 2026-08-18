from __future__ import annotations

import argparse
import json
import os
import sys

from pathlib import Path


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


from pipeline.production_manager import (
    ProductionManager,
)

from pipeline.production_orchestrator import (
    ProductionOrchestrator,
)

from execution.production_runner import (
    ProductionRunner,
)


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


def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Generate an LTX-13B video from "
            "a story using the existing planner "
            "and production execution pipeline."
        )
    )

    parser.add_argument(
        "--story",
        required=True,
        help=(
            "Story or generation request."
        ),
    )

    parser.add_argument(
        "--mode",
        default="ai_story",
        choices=[
            "ai_story",
            "preserve_user_story",
            "expand_user_story",
        ],
        help=(
            "Story planning mode."
        ),
    )

    parser.add_argument(
        "--gpu-url",
        action="append",
        required=True,
        metavar="ID=URL",
        help=(
            "ComfyUI worker URL. "
            "Repeat for multiple GPUs. "
            "Example: "
            "--gpu-url 0=http://127.0.0.1:8219"
        ),
    )

    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help=(
            "Optional path to save the "
            "final generation result metadata."
        ),
    )

    return parser.parse_args()


def parse_gpu_urls(
    values: list[str],
) -> dict[int, str]:

    result = {}

    for value in values:

        if "=" not in value:

            raise ValueError(
                "GPU URL must use "
                "ID=URL format:\n"
                f"{value}"
            )

        gpu_text, url = (
            value.split(
                "=",
                1,
            )
        )

        gpu_text = (
            gpu_text.strip()
        )

        url = (
            url.strip()
            .rstrip("/")
        )

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

        if gpu_id in result:

            raise ValueError(
                f"GPU ID {gpu_id} "
                "was specified more than once."
            )

        result[
            gpu_id
        ] = url

    if not result:

        raise ValueError(
            "At least one --gpu-url "
            "is required."
        )

    return result


def validate_environment():

    if not BASE_WORKFLOW.is_file():

        raise FileNotFoundError(
            "BASE workflow missing:\n"
            f"{BASE_WORKFLOW}"
        )

    if not DETAILER_WORKFLOW.is_file():

        raise FileNotFoundError(
            "DETAILER workflow missing:\n"
            f"{DETAILER_WORKFLOW}"
        )


def main():

    args = parse_args()

    validate_environment()

    gpu_urls = (
        parse_gpu_urls(
            args.gpu_url
        )
    )

    print(
        "=" * 80
    )

    print(
        "LTX-13B PRODUCTION GENERATION"
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

    for (
        gpu_id,
        url,
    ) in sorted(
        gpu_urls.items()
    ):

        print(
            f"  GPU {gpu_id}: {url}"
        )

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

    planner = (
        ProductionOrchestrator()
    )

    try:

        production_plan = (
            planner.create_production_plan(
                mode=args.mode,
                user_input=args.story,
            )
        )

    finally:

        planner.unload_models()

    shots = production_plan.get(
        "shots",
        [],
    )

    if not shots:

        raise RuntimeError(
            "Production planner returned "
            "zero shots."
        )

    print(
        f"✅ Production plan created: "
        f"{len(shots)} shots"
    )

    print(
        f"Production plan: "
        f"{production_plan.get('production_plan_path')}"
    )

    print()
    print(
        "=" * 80
    )

    print(
        "STEP 2 — EXECUTION"
    )

    print(
        "=" * 80
    )

    manager = (
        ProductionManager()
    )

    print(
        "Configured pipeline:"
    )

    for stage in (
        manager.get_pipeline()
    ):

        print(
            f"  → {stage}"
        )

    print()

    runner = (
        ProductionRunner(
            project_root=(
                PROJECT_ROOT
            ),

            gpu_urls=(
                gpu_urls
            ),

            workflow_path=(
                BASE_WORKFLOW
            ),

            detailer_workflow_path=(
                DETAILER_WORKFLOW
            ),
        )
    )

    final_video = (
        runner.run(
            production_plan
        )
    )

    result = {

        "status":
            "completed",

        "final_video":
            str(
                final_video
            ),

        "production_plan":
            production_plan.get(
                "production_plan_path"
            ),

        "shot_count":
            len(
                shots
            ),

        "gpu_workers":
            gpu_urls,

        "story_mode":
            args.mode,
    }

    if args.output_json is not None:

        output_path = (
            Path(
                args.output_json
            )
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
            f"Result metadata: "
            f"{output_path}"
        )

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


if __name__ == "__main__":
    main()
