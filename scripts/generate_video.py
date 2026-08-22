#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
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
        str(PROJECT_ROOT)
    )

from execution.comfy_client import (
    ComfyClient,
)
from execution.production_runner import (
    ProductionRunner,
)
from pipeline.production_manager import (
    ProductionManager,
)
from pipeline.production_orchestrator import (
    ProductionOrchestrator,
)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "MiniMax H3 Ref2VA "
            "production generator"
        )
    )

    parser.add_argument(
        "--story",
        required=True,
    )

    parser.add_argument(
        "--mode",
        default="ai_story",
        choices=[
            "ai_story",
            "preserve_user_story",
            "expand_user_story",
        ],
    )

    parser.add_argument(
        "--gpu-url",
        default=(
            "http://127.0.0.1:8188"
        ),
    )

    parser.add_argument(
        "--output-json",
        type=Path,
    )

    args = parser.parse_args()

    orchestrator = (
        ProductionOrchestrator()
    )

    try:
        plan = (
            orchestrator
            .create_production_plan(
                mode=args.mode,
                user_input=args.story,
            )
        )
    finally:
        orchestrator.unload_models()

    manager = (
        ProductionManager()
    )

    print(
        "\nPipeline:"
    )

    for stage in (
        manager.get_pipeline()
    ):
        print(
            "  ->",
            stage,
        )

    client = (
        ComfyClient(
            base_url=(
                args.gpu_url.rstrip("/")
            )
        )
    )

    runner = (
        ProductionRunner(
            project_root=PROJECT_ROOT,
            comfy_client=client,
        )
    )

    final_video = (
        runner.run(
            plan
        )
    )

    if (
        not final_video.is_file()
        or final_video.stat().st_size <= 0
    ):
        raise RuntimeError(
            "Generation completed but "
            "the final video is empty."
        )

    result = {
        "status": "completed",
        "backend": (
            "minimax-h3-ref2va-q4"
        ),
        "final_video": str(
            final_video
        ),
        "production_plan": (
            plan.get(
                "production_plan_path"
            )
        ),
        "shot_count": len(
            plan.get(
                "shots",
                [],
            )
        ),
    }

    if args.output_json:
        args.output_json.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        args.output_json.write_text(
            json.dumps(
                result,
                indent=2,
            ),
            encoding="utf-8",
        )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "H3 GENERATION COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        final_video
    )


if __name__ == "__main__":
    main()
