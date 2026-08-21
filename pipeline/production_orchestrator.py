import json

from datetime import datetime
from pathlib import Path

from execution.comfy_client import (
    ComfyClient,
)

from execution.reference_image_generator import (
    ReferenceImageGenerator,
)

from planner.config import (
    PRODUCTION_DIR,
    REFERENCE_IMAGE_HOST,
    REFERENCE_IMAGE_PORT,
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

        self.reference_generator = (
            ReferenceImageGenerator(
                client=ComfyClient(
                    base_url=(
                        "http://"
                        f"{REFERENCE_IMAGE_HOST}:"
                        f"{REFERENCE_IMAGE_PORT}"
                    )
                )
            )
        )

    # ============================================================
    # PREVIEWS
    # ============================================================

    @staticmethod
    def _print_story_preview(
        story: str,
    ):

        print()
        print("=" * 80)
        print("STORY PLAN")
        print("=" * 80)
        print(story)

    @staticmethod
    def _print_character_preview(
        characters: list,
    ):

        print()
        print("=" * 80)
        print("CHARACTER PLAN")
        print("=" * 80)

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

            print(
                "  Character state: "
                + json.dumps(
                    character.character_state,
                    ensure_ascii=False,
                )
            )

            print(
                f"  Reference: "
                f"{character.reference_mode.upper()}"
                + (
                    f" → {character.reference_path}"
                    if character.reference_path
                    else ""
                )
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
    ):

        print()
        print("=" * 80)
        print("SCENE PLAN")
        print("=" * 80)

        for scene in scenes:

            print()
            print(
                f"{scene.scene_id}"
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
                f"  Weather: "
                f"{scene.weather}"
            )

            print(
                f"  Atmosphere: "
                f"{scene.atmosphere}"
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
                "  Environment details: "
                + json.dumps(
                    scene.environment_details,
                    ensure_ascii=False,
                )
            )

            print(
                "  Key props: "
                + json.dumps(
                    scene.key_props,
                    ensure_ascii=False,
                )
            )

            print(
                f"  Scene objective: "
                f"{scene.scene_objective}"
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
    ):

        print()
        print("=" * 80)
        print("SHOT PLAN")
        print("=" * 80)

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
                f"  Duration: "
                f"{shot.duration_seconds}s"
            )

            print(
                f"  Characters: "
                f"{', '.join(shot.characters)}"
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

            print(
                f"  Visual prompt: "
                f"{shot.visual_prompt}"
            )

            if shot.negative_prompt:

                print(
                    f"  Negative prompt: "
                    f"{shot.negative_prompt}"
                )

            if shot.seed is not None:

                print(
                    f"  Seed: "
                    f"{shot.seed}"
                )

            print(
                "  Reference images: "
                + (
                    ", ".join(
                        shot.reference_images
                    )
                    if shot.reference_images
                    else "NONE"
                )
            )

            if shot.continuity_notes:

                print(
                    f"  Continuity: "
                    f"{shot.continuity_notes}"
                )

    # ============================================================
    # CHARACTER REFERENCES
    # ============================================================

    def _resolve_character_references(
        self,
        characters: list,
        shots: list,
    ) -> None:

        missing_characters = [
            character
            for character
            in characters
            if (
                character.reference_mode
                == "missing"
            )
        ]

        if not missing_characters:

            return

        print()
        print("=" * 80)
        print("GENERATING MISSING CHARACTER REFERENCES")
        print("=" * 80)

        # Qwen is no longer needed after all planning is complete.
        self.unload_models()

        for character in (
            missing_characters
        ):

            print()
            print(
                f"Generating reference: "
                f"{character.name}"
            )

            generated_path = (
                self.reference_generator
                .generate(
                    character_name=(
                        character.name
                    ),
                    description=(
                        character.description
                    ),
                    personality=(
                        character.personality
                    ),
                    appearance=(
                        character.appearance
                    ),
                    clothing=(
                        character.clothing
                    ),
                    distinctive_features=(
                        character.distinctive_features
                    ),
                    character_state=(
                        character.character_state
                    ),
                )
            )

            character.reference_mode = (
                "generated"
            )

            character.reference_path = (
                str(generated_path)
            )

            print(
                f"Generated reference: "
                f"{generated_path}"
            )

        # ------------------------------------------------------------
        # Attach the final reference path to every relevant shot.
        # ------------------------------------------------------------

        reference_by_character = {
            character.name.lower():
                character.reference_path
            for character in characters
            if character.reference_path
        }

        for shot in shots:

            shot.reference_images = []

            for character_name in (
                shot.characters
            ):

                reference = (
                    reference_by_character.get(
                        character_name.lower()
                    )
                )

                if reference:

                    shot.reference_images.append(
                        reference
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

                missing.append(
                    (
                        character.name,
                        "NO_REFERENCE",
                    )
                )

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

            characters = (
                getattr(
                    shot,
                    "characters",
                    []
                )
                or []
            )
        
            reference_images = (
                getattr(
                    shot,
                    "reference_images",
                    []
                )
                or []
            )
        
            # Environment / character-free shots do not require
            # character reference images.
            if not characters:
        
                if reference_images:
        
                    for image_path in reference_images:
        
                        if not Path(
                            image_path
                        ).is_file():
        
                            missing.append(
                                (
                                    shot.shot_id,
                                    str(image_path),
                                )
                            )
        
                continue
        
            # Character-containing shots must have references.
            if not reference_images:
        
                missing.append(
                    (
                        shot.shot_id,
                        "NO_REFERENCE_IMAGE",
                    )
                )
        
                continue
        
            for image_path in reference_images:
        
                if not Path(
                    image_path
                ).is_file():
        
                    missing.append(
                        (
                            shot.shot_id,
                            str(image_path)
                        )
                    )

        if missing:

            print()
            print("=" * 80)
            print("REFERENCE VALIDATION")
            print("=" * 80)

            for name, path in missing:

                print(
                    f"Missing reference: "
                    f"{name} → {path}"
                )

            raise RuntimeError(
                "Reference validation failed."
            )

        print()
        print("=" * 80)
        print("REFERENCE VALIDATION")
        print("=" * 80)
        print(
            "All character and shot references are valid."
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

        print("=" * 60)
        print(
            "STEP 2: Detecting characters"
        )
        print("=" * 60)

        character_names = (
            self.character_detector
            .detect(
                story=story,
                original_request=user_input,
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

        self._print_character_preview(
            characters
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

        self._print_scene_preview(
            scenes
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

        # ------------------------------------------------------------
        # Qwen work is finished.
        # Generate missing references only now.
        # ------------------------------------------------------------

        self._resolve_character_references(
            characters,
            all_shots,
        )

        self._validate_references(
            characters,
            all_shots,
        )

        self._print_shot_preview(
            all_shots
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

    def unload_models(self):

        self.model.unload()
