import json

from datetime import datetime

from planner.config import (
    PRODUCTION_DIR,
    ensure_directories,
)

from planner.qwen_loader import (
    QwenStoryModel,
)

from planner.story_planner import (
    StoryPlanner,
)

from planner.character_detector import (
    CharacterDetector,
)

from planner.character_planner import (
    CharacterPlanner,
)

from planner.scene_planner import (
    ScenePlanner,
)

from planner.shot_planner import (
    ShotPlanner,
)

from pipeline.continuity_manager import (
    ContinuityManager,
)


class ProductionOrchestrator:

    def __init__(self):

        ensure_directories()

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

    def create_production_plan(
        self,
        mode: str,
        user_input: str,
    ) -> dict:

        print("=" * 60)
        print("STEP 1: Creating story")
        print("=" * 60)

        story = (
            self.story_planner.plan(
                mode=mode,
                user_input=user_input,
            )
        )

        print("Story created.")

        print("=" * 60)
        print(
            "STEP 2: Detecting characters"
        )
        print("=" * 60)

        character_names = (
            self.character_detector
            .detect(
                story=story
            )
        )

        print(
            "Characters detected:"
        )

        for name in character_names:

            print(
                f" - {name}"
            )

        print("=" * 60)
        print(
            "STEP 3: Creating character plan"
        )
        print("=" * 60)

        characters = (
            self.character_planner
            .create_character_plan(
                story=story,
                character_names=(
                    character_names
                ),
            )
        )

        print(
            f"Characters created: "
            f"{len(characters)}"
        )

        print("=" * 60)
        print(
            "STEP 4: Creating scene plan"
        )
        print("=" * 60)

        scenes = (
            self.scene_planner
            .create_scene_plan(
                story=story,
                characters=characters,
            )
        )

        print(
            f"Scenes created: "
            f"{len(scenes)}"
        )

        all_shots = []

        previous_shot = None

        shot_start_index = 1

        for scene in scenes:

            print("=" * 60)

            print(
                f"STEP 5: Planning "
                f"{scene.scene_id}"
            )

            print("=" * 60)

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
                        shot_start_index
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

            shot_start_index += len(
                scene_shots
            )

            print(
                f"{scene.scene_id}: "
                f"{len(scene_shots)} shots"
            )

        production_plan = {
            "created_at": (
                datetime.now()
                .isoformat()
            ),

            "story": story,

            "character_names": (
                character_names
            ),

            "characters": [
                character.to_dict()
                for character
                in characters
            ],

            "scenes": [
                scene.to_dict()
                for scene
                in scenes
            ],

            "shots": [
                shot.to_dict()
                for shot
                in all_shots
            ],
        }

        output_path = (
            self.save_production_plan(
                production_plan
            )
        )

        production_plan[
            "production_plan_path"
        ] = str(
            output_path
        )

        return production_plan

    def save_production_plan(
        self,
        production_plan: dict,
    ):

        output_path = (
            PRODUCTION_DIR
            / "production_plan.json"
        )

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                production_plan,
                file,
                indent=2,
                ensure_ascii=False,
            )

        print("=" * 60)

        print(
            "Production plan saved:"
        )

        print(
            output_path
        )

        print("=" * 60)

        return output_path

    def unload_models(self):

        self.model.unload()
