import json

from pathlib import Path

from planner.qwen_loader import (
    QwenStoryModel,
)

from schemas.shot import (
    Shot,
)

from schemas.parser import (
    extract_json,
)


class ShotPlanner:

    def __init__(self):

        self.model = QwenStoryModel()

    def create_shot_plan(
        self,
        story: str,
        characters: list,
        scene,
        continuity_context: str = "",
    ) -> list:

        project_root = (
            Path(__file__)
            .resolve()
            .parents[1]
        )

        prompt_path = (
            project_root
            / "prompts"
            / "qwen"
            / "shot_plan.txt"
        )

        template = prompt_path.read_text(
            encoding="utf-8"
        )

        character_data = []

        for character in characters:

            if hasattr(
                character,
                "to_dict",
            ):

                character_data.append(
                    character.to_dict()
                )

            elif isinstance(
                character,
                dict,
            ):

                character_data.append(
                    character
                )

        if hasattr(
            scene,
            "to_dict",
        ):

            scene_data = scene.to_dict()

        else:

            scene_data = scene

        prompt = template.format(
            story=story,
            characters=json.dumps(
                character_data,
                indent=2,
            ),
            scene=json.dumps(
                scene_data,
                indent=2,
            ),
            continuity_context=continuity_context,
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a cinematic shot "
                    "planning system. Return only "
                    "valid JSON."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        response = self.model.generate(
            messages
        )

        data = extract_json(
            response
        )

        shots = []

        for item in data.get(
            "shots",
            [],
        ):

            shots.append(
                Shot(
                    shot_id=item.get(
                        "shot_id",
                        f"shot_{len(shots) + 1:03d}",
                    ),
                    scene_id=item.get(
                        "scene_id",
                        scene_data.get(
                            "scene_id",
                            "",
                        ),
                    ),
                    order=item.get(
                        "order",
                        len(shots) + 1,
                    ),
                    duration_seconds=float(
                        item.get(
                            "duration_seconds",
                            5.0,
                        )
                    ),
                    characters=item.get(
                        "characters",
                        [],
                    ),
                    location=item.get(
                        "location",
                        "",
                    ),
                    action=item.get(
                        "action",
                        "",
                    ),
                    camera_shot=item.get(
                        "camera_shot",
                        "",
                    ),
                    camera_movement=item.get(
                        "camera_movement",
                        "",
                    ),
                    lighting=item.get(
                        "lighting",
                        "",
                    ),
                    mood=item.get(
                        "mood",
                        "",
                    ),
                    visual_prompt=item.get(
                        "visual_prompt",
                        "",
                    ),
                    negative_prompt=item.get(
                        "negative_prompt",
                        "",
                    ),
                    previous_shot=item.get(
                        "previous_shot",
                    ),
                    next_shot=item.get(
                        "next_shot",
                    ),
                    continuity_notes=item.get(
                        "continuity_notes",
                        "",
                    ),
                    seed=item.get(
                        "seed",
                    ),
                    reference_images=item.get(
                        "reference_images",
                        [],
                    ),
                )
            )

        return shots

    def unload(self):

        self.model.unload()
