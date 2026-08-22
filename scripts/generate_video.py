from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

if str(ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT),
    )


def load_clients(
    urls: list[str],
) -> dict:

    from execution.comfy_client import (
        ComfyClient,
    )

    clients = {}

    for index, url in enumerate(
        urls
    ):

        client = (
            ComfyClient(
                base_url=url.rstrip("/")
            )
        )

        if not client.health_check():
            raise RuntimeError(
                f"ComfyUI worker unavailable: "
                f"{url}"
            )

        clients[index] = client

    return clients


def main():

    parser = argparse.ArgumentParser()

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
        "--worker",
        action="append",
        default=[
            "http://127.0.0.1:8188",
        ],
    )

    parser.add_argument(
        "--plan-only",
        action="store_true",
    )

    args = parser.parse_args()

    from pipeline.production_orchestrator import (
        ProductionOrchestrator,
    )

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

    plan_path = (
        plan.get(
            "production_plan_path"
        )
    )

    if plan_path:
        print(
            "PLAN:",
            plan_path,
        )

    if args.plan_only:
        return

    clients = load_clients(
        args.worker
    )

    from execution.production_runner import (
        ProductionRunner,
    )

    runner = (
        ProductionRunner(
            project_root=ROOT,
            comfy_clients=clients,
        )
    )

    final = (
        runner.run(
            plan
        )
    )

    result = {
        "status": "completed",
        "backend": (
            "minimax-h3-ref2va-q4"
        ),
        "video": str(
            final
        ),
        "shots": len(
            plan.get(
                "shots",
                [],
            )
        ),
    }

    print(
        json.dumps(
            result,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
