from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Shot:
    shot_id: str
    scene_id: str
    order: int
    duration_seconds: float

    characters: list = field(default_factory=list)

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

    # H3 media references.
    reference_images: list[str] = field(
        default_factory=list
    )
    reference_videos: list[str] = field(
        default_factory=list
    )
    reference_audio: Optional[str] = None

    reference_audio_paths: list[str] = field(
        default_factory=list
    )

    reference_audio_by_character: dict = field(
        default_factory=dict
    )

    reference_video_by_character: dict = field(
        default_factory=dict
    )

    speaking_characters: list[str] = field(
        default_factory=list
    )

    speech_text: str = ""

    reference_bindings: list[str] = field(
        default_factory=list
    )

    identity_locks: list[str] = field(
        default_factory=list
    )

    width: int = 960
    height: int = 544
    fps: int = 24
    frames_per_shot: int = 124
    steps: int = 14

    def to_dict(self) -> dict:
        return {
            "shot_id": self.shot_id,
            "scene_id": self.scene_id,
            "order": self.order,
            "duration_seconds": self.duration_seconds,
            "characters": self.characters,
            "location": self.location,
            "action": self.action,
            "camera_shot": self.camera_shot,
            "camera_movement": self.camera_movement,
            "lighting": self.lighting,
            "mood": self.mood,
            "visual_prompt": self.visual_prompt,
            "negative_prompt": self.negative_prompt,
            "previous_shot": self.previous_shot,
            "next_shot": self.next_shot,
            "continuity_notes": self.continuity_notes,
            "seed": self.seed,
            "reference_images": self.reference_images,
            "reference_videos": self.reference_videos,
            "reference_audio": self.reference_audio,
            "reference_audio_paths": (
                self.reference_audio_paths
            ),
            "reference_audio_by_character": (
                self.reference_audio_by_character
            ),
            "reference_video_by_character": (
                self.reference_video_by_character
            ),
            "speaking_characters": (
                self.speaking_characters
            ),
            "speech_text": self.speech_text,
            "reference_bindings": (
                self.reference_bindings
            ),
            "identity_locks": (
                self.identity_locks
            ),
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "frames_per_shot": (
                self.frames_per_shot
            ),
            "steps": self.steps,
        }
