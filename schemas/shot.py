from dataclasses import (
    dataclass,
    field,
)
from typing import Optional


@dataclass
class Shot:

    shot_id: str

    scene_id: str

    order: int

    duration_seconds: float

    characters: list = field(
        default_factory=list
    )

    location: str = ""

    action: str = ""

    camera_shot: str = ""

    camera_movement: str = ""

    lighting: str = ""

    mood: str = ""

    visual_prompt: str = ""

    negative_prompt: str = ""

    previous_shot: Optional[str] = None

    next_shot: Optional[str] = None

    continuity_notes: str = ""

    seed: Optional[int] = None

    reference_images: list = field(
        default_factory=list
    )

    def to_dict(self):

        return {
            "shot_id":
                self.shot_id,

            "scene_id":
                self.scene_id,

            "order":
                self.order,

            "duration_seconds":
                self.duration_seconds,

            "characters":
                self.characters,

            "location":
                self.location,

            "action":
                self.action,

            "camera_shot":
                self.camera_shot,

            "camera_movement":
                self.camera_movement,

            "lighting":
                self.lighting,

            "mood":
                self.mood,

            "visual_prompt":
                self.visual_prompt,

            "negative_prompt":
                self.negative_prompt,

            "previous_shot":
                self.previous_shot,

            "next_shot":
                self.next_shot,

            "continuity_notes":
                self.continuity_notes,

            "seed":
                self.seed,

            "reference_images":
                self.reference_images,
        }
