from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from planner.config import PRODUCTION_DIR, ensure_directories
from planner.qwen_loader import QwenStoryModel
from planner.story_planner import StoryPlanner
from planner.character_detector import CharacterDetector
from planner.character_planner import CharacterPlanner
from planner.scene_planner import ScenePlanner
from planner.shot_planner import ShotPlanner
from pipeline.continuity_manager import ContinuityManager
from pipeline.reference_manager import ReferenceManager


class ProductionOrchestrator:
    """
    Qwen remains responsible for story/character/scene/shot planning.

    H3 Ref2VA is the only video backend. Character reference discovery happens
    after planning and before the production plan is persisted.
    """

    def __init__(self):
        ensure_directories()

        self.model = QwenStoryModel()
        self.story_planner = StoryPlanner(model=self.model)
        self.character_detector = CharacterDetector(model=self.model)
        self.character_planner = CharacterPlanner(model=self.model)
        self.scene_planner = ScenePlanner(model=self.model)
        self.shot_planner = ShotPlanner(model=self.model)
        self.continuity_manager = ContinuityManager()
        self.references = ReferenceManager(
            Path(__file__).resolve().parents[1]
        )

    def create_production_plan(self, mode: str, user_input: str) -> dict:
        story = self.story_planner.plan(
            mode=mode,
            user_input=user_input,
        )

        names = self.character_detector.detect(
            story=story,
            original_request=user_input,
        )

        characters = self.character_planner.create_character_plan(
            story=story,
            character_names=names,
        )

        self.references.resolve_characters(characters)

        scenes = self.scene_planner.create_scene_plan(
            story=story,
            characters=characters,
        )

        all_shots = []
        previous_shot = None
        shot_index = 1

        for scene in scenes:
            context = self.continuity_manager.build_context(previous_shot)

            scene_shots = self.shot_planner.create_shot_plan(
                story=story,
                characters=characters,
                scene=scene,
                continuity_context=context,
                shot_start_index=shot_index,
            )

            scene_shots = self.continuity_manager.apply_scene_continuity(
                shots=scene_shots,
                previous_shot=previous_shot,
            )

            scene.shot_ids = [shot.shot_id for shot in scene_shots]

            if scene_shots:
                previous_shot = scene_shots[-1]

            all_shots.extend(scene_shots)
            shot_index += len(scene_shots)

        self.references.validate(characters, require_images=True)

        # A scene chain needs at least one image reference. Voice is required only
        # for a speaking character if the user supplied a voice reference.
        for shot in all_shots:
            if shot.characters and not shot.reference_images:
                raise RuntimeError(
                    f"{shot.shot_id} has characters but no H3 reference images."
                )

        production_plan = {
            "created_at": datetime.now().isoformat(),
            "backend": "minimax-h3-ref2va-q4",
            "story": story,
            "character_names": names,
            "characters": [c.to_dict() for c in characters],
            "scenes": [s.to_dict() for s in scenes],
            "shots": [s.to_dict() for s in all_shots],
        }

        output_path = PRODUCTION_DIR / "production_plan.json"
        output_path.write_text(
            json.dumps(
                production_plan,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        production_plan["production_plan_path"] = str(output_path)
        return production_plan

    def unload_models(self):
        try:
            self.model.unload()
        except Exception:
            pass
