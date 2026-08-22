from pipeline.modes import ReferenceMode, StoryMode


class ProductionManager:
    """H3-only high-level production configuration."""

    def __init__(
        self,
        story_mode=StoryMode.AI_STORY,
        reference_mode=ReferenceMode.AUTO,
        use_multigpu=False,
    ):
        self.story_mode = story_mode
        self.reference_mode = reference_mode
        self.use_multigpu = use_multigpu

    def get_pipeline(self):
        return [
            "story_planning",
            "character_detection",
            "character_planning",
            "scene_planning",
            "shot_planning",
            "h3_reference_resolution",
            "h3_ref2va_generation",
            "h3_720p_export",
            "video_assembly",
        ]

    def get_story_mode(self):
        return (
            self.story_mode.value
            if isinstance(self.story_mode, StoryMode)
            else str(self.story_mode)
        )

    def get_reference_mode(self):
        return (
            self.reference_mode.value
            if isinstance(self.reference_mode, ReferenceMode)
            else str(self.reference_mode)
        )
