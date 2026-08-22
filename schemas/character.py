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

    # Qwen's story-designed information.
    appearance: dict = field(
        default_factory=dict
    )

    clothing: dict = field(
        default_factory=dict
    )

    distinctive_features: list = field(
        default_factory=list
    )

    character_state: dict = field(
        default_factory=dict
    )

    continuity_rules: list = field(
        default_factory=list
    )

    # --------------------------------------------------------
    # H3 references
    # --------------------------------------------------------

    reference_mode: str = "missing"

    reference_paths: list[str] = field(
        default_factory=list
    )

    reference_video_paths: list[str] = field(
        default_factory=list
    )

    reference_audio_paths: list[str] = field(
        default_factory=list
    )

    reference_path: Optional[str] = None
    reference_video_path: Optional[str] = None
    reference_audio_path: Optional[str] = None

    # Backward compatibility only.
    reference_mask_path: Optional[str] = None

    # --------------------------------------------------------
    # Identity profile
    # --------------------------------------------------------

    identity_profile: dict = field(
        default_factory=dict
    )

    story_state_profile: dict = field(
        default_factory=dict
    )

    def normalized_reference_paths(self):
        values = list(
            self.reference_paths or []
        )

        if (
            self.reference_path
            and self.reference_path not in values
        ):
            values.insert(
                0,
                self.reference_path,
            )

        return [
            str(path)
            for path in values
            if str(path).strip()
        ]

    def normalized_video_paths(self):
        values = list(
            self.reference_video_paths or []
        )

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

    def normalized_audio_paths(self):
        values = list(
            self.reference_audio_paths or []
        )

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

    def build_identity_profile(self):
        appearance = (
            self.appearance or {}
        )

        self.identity_profile = {
            "name": self.name,

            "facial_features": appearance.get(
                "facial_features",
                "",
            ),

            "hair": appearance.get(
                "hair",
                "",
            ),

            "body_build": appearance.get(
                "body_build",
                "",
            ),

            "body_proportions": appearance.get(
                "body_proportions",
                "",
            ),

            "skin_tone": appearance.get(
                "skin_tone",
                "",
            ),

            "age_range": appearance.get(
                "age_range",
                "",
            ),

            "stable_identity_marks": (
                appearance.get(
                    "stable_identity_marks",
                    [],
                )
            ),
        }

        return self.identity_profile

    def build_story_state_profile(self):
        self.story_state_profile = {
            "clothing": self.clothing,
            "distinctive_features": (
                self.distinctive_features
            ),
            "character_state": (
                self.character_state
            ),
        }

        return self.story_state_profile

    def identity_lock_text(self):

        identity = (
            self.build_identity_profile()
        )

        def stringify(value):
            if isinstance(
                value,
                list,
            ):
                return ", ".join(
                    str(item)
                    for item in value
                )

            if isinstance(
                value,
                dict,
            ):
                return ", ".join(
                    f"{key}: {item}"
                    for key, item
                    in value.items()
                )

            return str(
                value or ""
            ).strip()

        return " ".join(
            [
                (
                    f"IMMUTABLE CHARACTER IDENTITY: "
                    f"{self.name}."
                ),
                (
                    "Preserve face geometry and "
                    "facial proportions."
                ),
                (
                    "Preserve hairstyle and hairline."
                ),
                (
                    "Preserve body structure and "
                    "body proportions."
                ),
                (
                    "Preserve skin tone."
                ),
                (
                    "Preserve stable identity-bearing "
                    "features."
                ),
                (
                    "Facial features: "
                    f"{stringify(identity['facial_features'])}."
                ),
                (
                    "Hair: "
                    f"{stringify(identity['hair'])}."
                ),
                (
                    "Body: "
                    f"{stringify(identity['body_build'])}."
                ),
                (
                    "Body proportions: "
                    f"{stringify(identity['body_proportions'])}."
                ),
                (
                    "Skin tone: "
                    f"{stringify(identity['skin_tone'])}."
                ),
                (
                    "Age range: "
                    f"{stringify(identity['age_range'])}."
                ),
                (
                    "Stable identity marks: "
                    f"{stringify(identity['stable_identity_marks'])}."
                ),
            ]
        )

    def story_state_text(self):

        state = (
            self.build_story_state_profile()
        )

        return (
            f"CURRENT STORY STATE FOR {self.name}: "
            f"clothing={state['clothing']}; "
            f"distinctive_features="
            f"{state['distinctive_features']}; "
            f"character_state="
            f"{state['character_state']}."
        )

    def to_dict(self):

        images = (
            self.normalized_reference_paths()
        )

        videos = (
            self.normalized_video_paths()
        )

        audios = (
            self.normalized_audio_paths()
        )

        self.build_identity_profile()
        self.build_story_state_profile()

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

            "character_state": (
                self.character_state
            ),

            "continuity_rules": (
                self.continuity_rules
            ),

            "reference_mode": (
                self.reference_mode
            ),

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

            "story_state_profile": (
                self.story_state_profile
            ),

            "identity_lock": (
                self.identity_lock_text()
            ),

            "story_state_lock": (
                self.story_state_text()
            ),
        }
