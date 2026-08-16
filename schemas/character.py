from dataclasses import (
    dataclass,
    field,
)
from typing import Optional


@dataclass
class Character:

    character_id: str

    name: str

    role: str

    description: str

    personality: str

    appearance: dict = field(
        default_factory=dict
    )

    clothing: dict = field(
        default_factory=dict
    )

    distinctive_features: list = field(
        default_factory=list
    )

    reference_mode: str = "auto"

    reference_path: Optional[str] = None

    continuity_rules: list = field(
        default_factory=list
    )

    def to_dict(self):

        return {
            "character_id":
                self.character_id,

            "name":
                self.name,

            "role":
                self.role,

            "description":
                self.description,

            "personality":
                self.personality,

            "appearance":
                self.appearance,

            "clothing":
                self.clothing,

            "distinctive_features":
                self.distinctive_features,

            "reference_mode":
                self.reference_mode,

            "reference_path":
                self.reference_path,

            "continuity_rules":
                self.continuity_rules,
        }
