from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Character:
    character_id: str
    name: str
    role: str
    description: str
    personality: str

    appearance: dict = field(default_factory=dict)
    clothing: dict = field(default_factory=dict)
    distinctive_features: list = field(default_factory=list)
    character_state: dict = field(default_factory=dict)
    continuity_rules: list = field(default_factory=list)

    # H3 Ref2VA reference package.
    reference_mode: str = "missing"
    reference_paths: list[str] = field(default_factory=list)
    reference_audio_path: Optional[str] = None
    reference_video_path: Optional[str] = None

    # Kept for backward-compatible JSON/planner inputs.
    reference_path: Optional[str] = None
    reference_mask_path: Optional[str] = None

    def normalized_reference_paths(self) -> list[str]:
        values = list(self.reference_paths or [])
        if self.reference_path and self.reference_path not in values:
            values.insert(0, self.reference_path)
        return [str(p) for p in values if str(p).strip()]

    def to_dict(self) -> dict:
        refs = self.normalized_reference_paths()

        return {
            "character_id": self.character_id,
            "name": self.name,
            "role": self.role,
            "description": self.description,
            "personality": self.personality,
            "appearance": self.appearance,
            "clothing": self.clothing,
            "distinctive_features": self.distinctive_features,
            "character_state": self.character_state,
            "continuity_rules": self.continuity_rules,
            "reference_mode": self.reference_mode,
            "reference_paths": refs,
            "reference_path": refs[0] if refs else self.reference_path,
            "reference_mask_path": self.reference_mask_path,
            "reference_audio_path": self.reference_audio_path,
            "reference_video_path": self.reference_video_path,
        }
