from pipeline.modes import (
    ReferenceMode,
    StoryMode,
)


class ProductionManager:

    """
    High-level production configuration.

    This class describes what the production should do.
    Execution itself is handled by ProductionRunner.
    """

    def __init__(
        self,
        story_mode=StoryMode.AI_STORY,
        reference_mode=(
            ReferenceMode.AUTO
        ),
        use_detailer=True,
        use_upscaler=True,
        use_multigpu=True,
    ):

        self.story_mode = (
            story_mode
        )

        self.reference_mode = (
            reference_mode
        )

        self.use_detailer = (
            use_detailer
        )

        self.use_upscaler = (
            use_upscaler
        )

        self.use_multigpu = (
            use_multigpu
        )

    def get_pipeline(self):

        pipeline = [
            "story_planning",
            "character_detection",
            "character_planning",
            "scene_planning",
            "shot_planning",
            "checkpoint_initialization",
            "ltx_generation",
        ]

        if self.use_detailer:

            pipeline.append(
                "ic_lora_detailer"
            )

        if self.use_upscaler:

            pipeline.append(
                "ltx_spatial_upscaler"
            )

        pipeline.extend(
            [
                "checkpoint_validation",
                "video_assembly",
                "final_720p_export",
            ]
        )

        return pipeline

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

            return (
                self.reference_mode.value
            )

        return str(
            self.reference_mode
        )
