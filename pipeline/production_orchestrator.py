import json

from datetime import datetime
from pathlib import Path

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

    # ============================================================
    # PLANNING PREVIEW
    # ============================================================

    @staticmethod
    def _print_story_preview(
        story: str,
    ) -> None:

        print()
        print("=" * 80)
        print("STORY PLAN")
        print("=" * 80)
        print(story)

    @staticmethod
    def _print_character_preview(
        characters: list,
    ) -> None:

        print()
        print("=" * 80)
        print("CHARACTER PLAN")
        print("=" * 80)

        if not characters:

            print("No characters detected.")
            return

        for character in characters:

            print()
            print(
                f"Character: {character.name}"
            )

            print(
                f"  Role: {character.role}"
            )

            print(
                f"  Description: "
                f"{character.description}"
            )

            print(
                f"  Personality: "
                f"{character.personality}"
            )

            print(
                "  Appearance: "
                + json.dumps(
                    character.appearance,
                    ensure_ascii=False,
                )
            )

            print(
                "  Clothing: "
                + json.dumps(
                    character.clothing,
                    ensure_ascii=False,
                )
            )

            print(
                "  Distinctive features: "
                + json.dumps(
                    character.distinctive_features,
                    ensure_ascii=False,
                )
            )

            if character.reference_path:

                print(
                    "  Reference: "
                    f"{character.reference_mode.upper()} "
                    f"→ {character.reference_path}"
                )

            else:

                print(
                    "  Reference: "
                    f"{character.reference_mode.upper()}"
                )

            if character.continuity_rules:

                print(
                    "  Continuity: "
                    + json.dumps(
                        character.continuity_rules,
                        ensure_ascii=False,
                    )
                )

    @staticmethod
    def _print_scene_preview(
        scenes: list,
    ) -> None:

        print()
        print("=" * 80)
        print("SCENE PLAN")
        print("=" * 80)

        if not scenes:

            print("No scenes created.")
            return

        for scene in scenes:

            print()
            print(
                f"{scene.scene_id}"
            )

            print(
                f"  Order: {scene.order}"
            )

            print(
                f"  Location: "
                f"{scene.location}"
            )

            print(
                f"  Time of day: "
                f"{scene.time_of_day}"
            )

            print(
                f"  Description: "
                f"{scene.description}"
            )

            print(
                f"  Mood: "
                f"{scene.mood}"
            )

            print(
                f"  Lighting: "
                f"{scene.lighting}"
            )

            print(
                f"  Characters: "
                f"{', '.join(scene.characters)}"
            )

            print(
                f"  Story summary: "
                f"{scene.story_summary}"
            )

            if scene.continuity_notes:

                print(
                    f"  Continuity: "
                    f"{scene.continuity_notes}"
                )

            print(
                f"  Shots planned: "
                f"{len(scene.shot_ids)}"
            )

    @staticmethod
    def _print_shot_preview(
        shots: list,
    ) -> None:

        print()
        print("=" * 80)
        print("SHOT PLAN")
        print("=" * 80)

        if not shots:

            print("No shots created.")
            return

        for shot in shots:

            print()
            print(
                f"{shot.shot_id}"
            )

            print(
                f"  Scene: "
                f"{shot.scene_id}"
            )

            print(
                f"  Order: "
                f"{shot.order}"
            )

            print(
                f"  Duration: "
                f"{shot.duration_seconds}s"
            )

            print(
                f"  Characters: "
                f"{', '.join(shot.characters)}"
            )

            print(
                f"  Location: "
                f"{shot.location}"
            )

            print(
                f"  Action: "
                f"{shot.action}"
            )

            print(
                f"  Camera: "
                f"{shot.camera_shot}"
            )

            print(
                f"  Camera movement: "
                f"{shot.camera_movement}"
            )

            print(
                f"  Lighting: "
                f"{shot.lighting}"
            )

            print(
                f"  Mood: "
                f"{shot.mood}"
            )

            if shot.visual_prompt:

                print(
                    "  Visual prompt:"
                )

                print(
                    f"    {shot.visual_prompt}"
                )

            if shot.negative_prompt:

                print(
                    "  Negative prompt:"
                )

                print(
                    f"    {shot.negative_prompt}"
                )

            if shot.seed is not None:

                print(
                    f"  Seed: "
                    f"{shot.seed}"
                )

            if shot.reference_images:

                print(
                    "  Reference images:"
                )

                for image in (
                    shot.reference_images
                ):

                    print(
                        f"    → {image}"
                    )

            else:

                print(
                    "  Reference images: NONE"
                )

            if shot.previous_shot:

                print(
                    f"  Previous shot: "
                    f"{shot.previous_shot}"
                )

            if shot.next_shot:

                print(
                    f"  Next shot: "
                    f"{shot.next_shot}"
                )

            if shot.continuity_notes:

                print(
                    "  Continuity:"
                )

                print(
                    f"    {shot.continuity_notes}"
                )

    # ============================================================
    # REFERENCE VALIDATION
    # ============================================================

    @staticmethod
    def _validate_references(
        characters: list,
        shots: list,
    ) -> None:

        missing = []

        for character in characters:

            if not character.reference_path:

                continue

            path = Path(
                character.reference_path
            )

            if not path.is_file():

                missing.append(
                    (
                        character.name,
                        str(path),
                    )
                )

        for shot in shots:

            if not shot.reference_images:

                missing.append(
                    (
                        shot.shot_id,
                        "NO_REFERENCE_IMAGE",
                    )
                )

                continue

            for image_path in (
                shot.reference_images
            ):

                path = Path(
                    image_path
                )

                if not path.is_file():

                    missing.append(
                        (
                            shot.shot_id,
                            str(path),
                        )
                    )

        print()
        print("=" * 80)
        print("REFERENCE VALIDATION")
        print("=" * 80)

        if missing:

            for name, path in missing:

                print(
                    f"Missing reference: "
                    f"{name} → {path}"
                )

            raise RuntimeError(
                "Reference validation failed. "
                "The current BASE workflow is I2V "
                "and requires a valid input image "
                "for every shot."
            )

        print(
            "All character and shot reference "
            "images are valid."
        )

    # ============================================================
    # PRODUCTION PLAN
    # ============================================================

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

        self._print_story_preview(
            story
        )

        print()
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

        if character_names:

            for name in character_names:

                print(
                    f" - {name}"
                )

        else:

            print(
                " - None"
            )

        print()
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

        self._print_character_preview(
            characters
        )

        print()
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

        self._print_scene_preview(
            scenes
        )

        all_shots = []

        previous_shot = None

        shot_start_index = 1

        for scene in scenes:

            print()
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

        self._print_shot_preview(
            all_shots
        )

        self._validate_references(
            characters,
            all_shots,
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

    # ============================================================
    # SAVE PRODUCTION PLAN
    # ============================================================

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

        print()
        print("=" * 60)

        print(
            "Production plan saved:"
        )

        print(
            output_path
        )

        print("=" * 60)

        return output_path

    # ============================================================
    # UNLOAD
    # ============================================================

    def unload_models(self):

        self.model.unload()
