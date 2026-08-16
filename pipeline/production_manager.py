from pipeline.modes import (
    ReferenceMode,
)


class ProductionManager:
    """
    Controls the high-level production decisions.
    """

    def __init__(
        self,
        story_mode,
        reference_mode=ReferenceMode.AUTO,
        use_detailer=True,
        use_upscaler=True,
        use_multigpu=True,
    ):

        self.story_mode = story_mode

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
            "character_planning",
            "scene_planning",
            "shot_planning",
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
                "video_assembly",
                "final_export",
            ]
        )

        return pipeline
