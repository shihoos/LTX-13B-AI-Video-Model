from pipeline.modes import (
    ReferenceMode,
    StoryMode,
)


class ProductionManager:
    """
    H3-only production configuration.

    The old LTX IC-LoRA/detailer and LTX spatial-upscale stages
    are intentionally removed.
    """

    def __init__(
        self,
        story_mode=StoryMode.AI_STORY,
        reference_mode=ReferenceMode.AUTO,
        use_multigpu=True,
    ):
        self.story_mode = story_mode
        self.reference_mode = reference_mode
        self.use_multigpu = use_multigpu

    def get_pipeline(self):
        return [
            "story_planning",
            "character_detection",
            "character_planning",
            "identity_profile_generation",
            "scene_planning",
            "shot_planning",
            "reference_validation",
            "h3_generation",
            "scene_continuity",
            "video_assembly",
            "720p_delivery",
        ]

    def get_story_mode(self):
        if isinstance(
            self.story_mode,
            StoryMode,
        ):
            return self.story_mode.value

        return str(
            self.story_mode
        )

    def get_reference_mode(self):
        if isinstance(
            self.reference_mode,
            ReferenceMode,
        ):
            return self.reference_mode.value

        return str(
            self.reference_mode
        )
