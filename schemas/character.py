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

    # H3 reference package.
    reference_mode: str = "missing"

    reference_paths: list[str] = field(default_factory=list)
    reference_video_paths: list[str] = field(default_factory=list)
    reference_audio_paths: list[str] = field(default_factory=list)

    # Backward-compatible fields.
    reference_path: Optional[str] = None
    reference_video_path: Optional[str] = None
    reference_audio_path: Optional[str] = None

    # Kept only for backward-compatible JSON.
    reference_mask_path: Optional[str] = None

    identity_profile: dict = field(default_factory=dict)

    def normalized_reference_paths(self) -> list[str]:
        values = list(self.reference_paths or [])

        if self.reference_path and self.reference_path not in values:
            values.insert(0, self.reference_path)

        return [
            str(path)
            for path in values
            if str(path).strip()
        ]

    def normalized_video_paths(self) -> list[str]:
        values = list(self.reference_video_paths or [])

        if (
            self.reference_video_path
            and self.reference_video_path not in values
        ):
            values.insert(
                0,
                self.reference_video_path,
            )

        return [
            str(path)
            for path in values
            if str(path).strip()
        ]

    def normalized_audio_paths(self) -> list[str]:
        values = list(self.reference_audio_paths or [])

        if (
            self.reference_audio_path
            and self.reference_audio_path not in values
        ):
            values.insert(
                0,
                self.reference_audio_path,
            )

        return [
            str(path)
            for path in values
            if str(path).strip()
        ]

    def build_identity_profile(self) -> dict:
        self.identity_profile = {
            "name": self.name,
            "facial_features": self.appearance.get(
                "facial_features",
                "",
            ),
            "hair": self.appearance.get(
                "hair",
                "",
            ),
            "body_build": self.appearance.get(
                "body_build",
                "",
            ),
            "skin_tone": self.appearance.get(
                "skin_tone",
                "",
            ),
            "age_range": self.appearance.get(
                "age_range",
                "",
            ),
            "accessories": self.appearance.get(
                "accessories",
                [],
            ),
            "clothing": self.clothing,
            "distinctive_features": (
                self.distinctive_features
            ),
            "character_state": (
                self.character_state
            ),
            "continuity_rules": (
                self.continuity_rules
            ),
        }

        return self.identity_profile

    def identity_lock_text(self) -> str:
        profile = self.build_identity_profile()

        def stringify(value) -> str:
            if isinstance(value, list):
                return ", ".join(
                    str(item)
                    for item in value
                )

            if isinstance(value, dict):
                return ", ".join(
                    f"{key}: {item}"
                    for key, item in value.items()
                )

            return str(value or "").strip()

        return " ".join(
            [
                f"CHARACTER IDENTITY LOCK: {self.name}.",
                (
                    f"Facial features: "
                    f"{stringify(profile['facial_features'])}."
                ),
                f"Hair: {stringify(profile['hair'])}.",
                (
                    f"Body build: "
                    f"{stringify(profile['body_build'])}."
                ),
                (
                    f"Skin tone: "
                    f"{stringify(profile['skin_tone'])}."
                ),
                (
                    f"Age range: "
                    f"{stringify(profile['age_range'])}."
                ),
                (
                    f"Accessories: "
                    f"{stringify(profile['accessories'])}."
                ),
                (
                    f"Clothing continuity: "
                    f"{stringify(profile['clothing'])}."
                ),
                (
                    f"Distinctive features: "
                    f"{stringify(profile['distinctive_features'])}."
                ),
                (
                    f"Current state: "
                    f"{stringify(profile['character_state'])}."
                ),
                (
                    f"Continuity rules: "
                    f"{stringify(profile['continuity_rules'])}."
                ),
                (
                    "Preserve the same face, hairstyle, "
                    "hairline, body proportions, skin tone, "
                    "distinctive marks and identity-bearing "
                    "accessories throughout the shot. "
                    "Only change clothing or physical state "
                    "when the story explicitly requires it."
                ),
            ]
        )

    def to_dict(self) -> dict:
        images = self.normalized_reference_paths()
        videos = self.normalized_video_paths()
        audios = self.normalized_audio_paths()

        self.build_identity_profile()

        return {
            "character_id": self.character_id,
            "name": self.name,
            "role": self.role,
            "description": self.description,
            "personality": self.personality,
            "appearance": self.appearance,
            "clothing": self.clothing,
            "distinctive_features": (
                self.distinctive_features
            ),
            "character_state": self.character_state,
            "continuity_rules": (
                self.continuity_rules
            ),
            "reference_mode": self.reference_mode,
            "reference_paths": images,
            "reference_video_paths": videos,
            "reference_audio_paths": audios,
            "reference_path": (
                images[0]
                if images
                else self.reference_path
            ),
            "reference_video_path": (
                videos[0]
                if videos
                else self.reference_video_path
            ),
            "reference_audio_path": (
                audios[0]
                if audios
                else self.reference_audio_path
            ),
            "reference_mask_path": (
                self.reference_mask_path
            ),
            "identity_profile": (
                self.identity_profile
            ),
            "identity_lock": (
                self.identity_lock_text()
            ),
        }
