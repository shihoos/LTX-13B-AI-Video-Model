from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from pipeline.continuity_manager import (
    ContinuityManager,
)
from pipeline.reference_manager import (
    ReferenceManager,
)
from planner.character_detector import (
    CharacterDetector,
)
from planner.character_planner import (
    CharacterPlanner,
)
from planner.config import (
    PRODUCTION_DIR,
    ensure_directories,
)
from planner.qwen_loader import (
    QwenStoryModel,
)
from planner.scene_planner import (
    ScenePlanner,
)
from planner.shot_planner import (
    ShotPlanner,
)
from planner.story_planner import (
    StoryPlanner,
)


class ProductionOrchestrator:

    def __init__(self):
        ensure_directories()

        self.project_root = (
            Path(__file__)
            .resolve()
            .parents[1]
        )

        self.model = (
            QwenStoryModel()
        )

        self.story_planner = (
            StoryPlanner(
                model=self.model
            )
        )

        self.character_detector = (
            CharacterDetector(
                model=self.model
            )
        )

        self.character_planner = (
            CharacterPlanner(
                model=self.model
            )
        )

        self.scene_planner = (
            ScenePlanner(
                model=self.model
            )
        )

        self.shot_planner = (
            ShotPlanner(
                model=self.model
            )
        )

        self.continuity_manager = (
            ContinuityManager()
        )

        self.references = (
            ReferenceManager(
                self.project_root
            )
        )

    def create_production_plan(
        self,
        mode: str,
        user_input: str,
    ) -> dict:

        story = (
            self.story_planner.plan(
                mode=mode,
                user_input=user_input,
            )
        )

        names = (
            self.character_detector.detect(
                story=story,
                original_request=user_input,
            )
        )

        characters = (
            self.character_planner
            .create_character_plan(
                story=story,
                character_names=names,
            )
        )

        self.references.resolve_characters(
            characters
        )

        self.references.validate(
            characters,
            require_images=True,
        )

        scenes = (
            self.scene_planner
            .create_scene_plan(
                story=story,
                characters=characters,
            )
        )

        all_shots = []
        previous_shot = None
        shot_index = 1

        for scene in scenes:

            continuity_context = (
                self.continuity_manager
                .build_context(
                    previous_shot
                )
            )

            scene_shots = (
                self.shot_planner
                .create_shot_plan(
                    story=story,
                    characters=characters,
                    scene=scene,
                    continuity_context=(
                        continuity_context
                    ),
                    shot_start_index=(
                        shot_index
                    ),
                )
            )

            scene_shots = (
                self.continuity_manager
                .apply_scene_continuity(
                    shots=scene_shots,
                    previous_shot=(
                        previous_shot
                    ),
                )
            )

            scene.shot_ids = [
                shot.shot_id
                for shot in scene_shots
            ]

            if scene_shots:
                previous_shot = (
                    scene_shots[-1]
                )

            all_shots.extend(
                scene_shots
            )

            shot_index += len(
                scene_shots
            )

        for shot in all_shots:

            if (
                shot.characters
                and not shot.reference_images
            ):
                raise RuntimeError(
                    f"{shot.shot_id}: "
                    "character shot has no "
                    "reference images"
                )

            if len(
                shot.reference_images
            ) > 9:
                raise RuntimeError(
                    f"{shot.shot_id}: "
                    "more than 9 reference images"
                )

            if len(
                shot.reference_videos
            ) > 3:
                raise RuntimeError(
                    f"{shot.shot_id}: "
                    "more than 3 reference videos"
                )

            if len(
                shot.reference_audio_paths
            ) > 3:
                raise RuntimeError(
                    f"{shot.shot_id}: "
                    "more than 3 reference audio files"
                )

        production_plan = {
            "created_at": (
                datetime.now()
                .isoformat()
            ),
            "backend": (
                "minimax-h3-ref2va-q4"
            ),
            "identity_strategy": (
                "structured_identity_profile"
                "+independent_h3_references"
            ),
            "story": story,
            "character_names": names,
            "characters": [
                character.to_dict()
                for character in characters
            ],
            "scenes": [
                scene.to_dict()
                for scene in scenes
            ],
            "shots": [
                shot.to_dict()
                for shot in all_shots
            ],
        }

        output_path = (
            PRODUCTION_DIR
            / "production_plan.json"
        )

        output_path.write_text(
            json.dumps(
                production_plan,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        production_plan[
            "production_plan_path"
        ] = str(
            output_path
        )

        return production_plan

    def unload_models(self):
        try:
            self.model.unload()
        except Exception:
            pass
