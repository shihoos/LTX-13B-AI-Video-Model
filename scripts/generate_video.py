#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.production_orchestrator import ProductionOrchestrator
from pipeline.production_manager import ProductionManager
from execution.production_runner import ProductionRunner


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "MiniMax H3 Ref2VA production generator: "
            "Qwen planning -> H3 reference-to-video -> 720p assembly."
        )
    )

    parser.add_argument("--story", required=True)
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
        action="append",
        default=None,
        metavar="ID=URL",
        help="Repeatable ComfyUI worker URL, e.g. 0=http://127.0.0.1:8188",
    )
    parser.add_argument("--output-json", type=Path, default=None)

    return parser.parse_args()


def parse_gpu_urls(values):
    if not values:
        return {0: "http://127.0.0.1:8188"}

    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError(
                f"GPU URL must be ID=URL: {value}"
            )

        gpu, url = value.split("=", 1)
        gpu_id = int(gpu)
        if gpu_id < 0:
            raise ValueError("GPU ID must be >= 0.")

        url = url.rstrip("/")
        if not url.startswith(("http://", "https://")):
            raise ValueError(
                f"Invalid GPU URL: {url}"
            )

        result[gpu_id] = url

    return result


def main():
    args = parse_args()
    gpu_urls = parse_gpu_urls(args.gpu_url)

    orchestrator = ProductionOrchestrator()

    try:
        plan = orchestrator.create_production_plan(
            mode=args.mode,
            user_input=args.story,
        )
    finally:
        orchestrator.unload_models()

    manager = ProductionManager()
    print("\nH3 pipeline:")
    for stage in manager.get_pipeline():
        print(f"  -> {stage}")

    runner = ProductionRunner(
        project_root=PROJECT_ROOT,
        gpu_urls=gpu_urls,
    )

    final_video = runner.run(plan)

    if not final_video.is_file() or final_video.stat().st_size <= 0:
        raise RuntimeError(
            f"H3 reported completion but final video is invalid: {final_video}"
        )

    result = {
        "status": "completed",
        "backend": "minimax-h3-ref2va-q4",
        "final_video": str(final_video),
        "production_plan": plan.get("production_plan_path"),
        "shot_count": len(plan.get("shots", [])),
        "gpu_workers": gpu_urls,
    }

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(result, indent=2),
            encoding="utf-8",
        )

    print("\n" + "=" * 80)
    print("H3 PRODUCTION COMPLETE")
    print("=" * 80)
    print(final_video)


if __name__ == "__main__":
    main()
