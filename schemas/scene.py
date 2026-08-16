from dataclasses import (
    dataclass,
    field,
)


@dataclass
class Scene:

    scene_id: str

    order: int

    location: str = ""

    time_of_day: str = ""

    description: str = ""

    characters: list = field(
        default_factory=list
    )

    story_summary: str = ""

    continuity_notes: str = ""

    shot_ids: list = field(
        default_factory=list
    )

    def to_dict(self):

        return {
            "scene_id":
                self.scene_id,

            "order":
                self.order,

            "location":
                self.location,

            "time_of_day":
                self.time_of_day,

            "description":
                self.description,

            "characters":
                self.characters,

            "story_summary":
                self.story_summary,

            "continuity_notes":
                self.continuity_notes,

            "shot_ids":
                self.shot_ids,
        }
