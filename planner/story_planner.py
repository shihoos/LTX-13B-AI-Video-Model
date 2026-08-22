from __future__ import annotations

import json
from pathlib import Path

from planner.config import QWEN_SHOT_PLAN_TEMPERATURE
from planner.qwen_loader import QwenStoryModel
from schemas.parser import extract_json
from schemas.shot import Shot


class ShotPlanner:
    """Creates H3-ready shot contracts while preserving the Qwen planner."""

    H3_WIDTH = 960
    H3_HEIGHT = 544
    H3_FPS = 24
    H3_FRAMES = 124  # 5.17 s, on H3's 17k+5 frame grid.

    def __init__(self, model=None):
        self.model = model if model is not None else QwenStoryModel()

    def _read_prompt(self) -> str:
        path = (
            Path(__file__).resolve().parents[1]
            / "prompts"
            / "qwen"
            / "shot_plan.txt"
        )
        return path.read_text(encoding="utf-8")

    @staticmethod
    def _reference_map(characters: list) -> dict:
        result = {}
        for character in characters:
            name = getattr(character, "name", "") or ""
            refs = list(getattr(character, "reference_paths", []) or [])
            if getattr(character, "reference_path", None):
                refs.insert(0, character.reference_path)

            unique = []
            for ref in refs:
                if ref and ref not in unique:
                    unique.append(str(ref))

            if name:
                result[name.lower()] = unique
        return result

    @staticmethod
    def _audio_map(characters: list) -> dict:
        return {
            getattr(c, "name", "").lower(): getattr(c, "reference_audio_path", None)
            for c in characters
            if getattr(c, "name", "")
        }

    @staticmethod
    def _video_map(characters: list) -> dict:
        return {
            getattr(c, "name", "").lower(): getattr(c, "reference_video_path", None)
            for c in characters
            if getattr(c, "name", "")
        }

    @staticmethod
    def _clean_characters(values) -> list[str]:
        if not isinstance(values, list):
            return []
        return [str(x).strip() for x in values if str(x).strip()]

    def create_shot_plan(
        self,
        story: str,
        characters: list,
        scene,
        continuity_context: str = "",
        shot_start_index: int = 1,
    ) -> list[Shot]:
        template = self._read_prompt()

        character_data = [
            c.to_dict() if hasattr(c, "to_dict") else c
            for c in characters
        ]
        scene_data = scene.to_dict() if hasattr(scene, "to_dict") else scene

        prompt = template.format(
            story=story,
            characters=json.dumps(
                character_data,
                indent=2,
                ensure_ascii=False,
            ),
            scene=json.dumps(
                scene_data,
                indent=2,
                ensure_ascii=False,
            ),
            continuity_context=continuity_context,
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a cinematic shot planning system. "
                    "Return only valid JSON matching the requested schema."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        response = self.model.generate(
            messages,
            temperature=QWEN_SHOT_PLAN_TEMPERATURE,
        )

        data = extract_json(response)

        refs = self._reference_map(characters)
        audio = self._audio_map(characters)
        videos = self._video_map(characters)

        shots = []

        for item in data.get("shots", []):
            shot_characters = self._clean_characters(
                item.get("characters", [])
            )

            # Keep one fixed frame count inside one H3 scene chain.
            # The production runner groups scenes, so per-scene references stay stable.
            reference_images = []
            reference_bindings = []
            reference_video = None
            reference_audio = None
            speaking = self._clean_characters(
                item.get("speaking_characters", [])
            )

            for character_name in shot_characters:
                character_refs = refs.get(character_name.lower(), [])
                for ref in character_refs:
                    if ref not in reference_images:
                        reference_images.append(ref)

                if reference_images:
                    start = len(reference_images) - len(character_refs)
                    added = reference_images[start:] if character_refs else []
                    if added:
                        tags = ", ".join(
                            f"<Picture {reference_images.index(p) + 1}>"
                            for p in added
                        )
                        reference_bindings.append(
                            f"{tags} are reference images of {character_name}."
                        )

                if not reference_video:
                    reference_video = videos.get(character_name.lower())

            # H3 ref2va currently exposes one voice_ref lane in the
            # multishot memory sampler. Prefer the first speaking character.
            for speaker in speaking:
                candidate = audio.get(speaker.lower())
                if candidate:
                    reference_audio = candidate
                    break

            if len(reference_images) > 9:
                reference_images = reference_images[:9]

            speech_text = str(item.get("speech_text", "") or "").strip()
            if speaking and not reference_audio:
                # Dialogue may still be generated, but the voice will not be reference-anchored.
                pass

            shot_prompt = str(
                item.get("visual_prompt", "")
            ).strip()

            binding_text = " ".join(reference_bindings)
            if binding_text:
                shot_prompt = f"{binding_text} {shot_prompt}".strip()

            shot = Shot(
                shot_id=f"shot_{shot_start_index + len(shots):03d}",
                scene_id=scene_data.get("scene_id", ""),
                order=len(shots) + 1,
                duration_seconds=5.0,
                characters=shot_characters,
                location=item.get("location", scene_data.get("location", "")),
                action=item.get("action", ""),
                camera_shot=item.get("camera_shot", ""),
                camera_movement=item.get("camera_movement", ""),
                lighting=item.get("lighting", ""),
                mood=item.get("mood", ""),
                visual_prompt=shot_prompt,
                negative_prompt=item.get("negative_prompt", ""),
                previous_shot=None,
                next_shot=None,
                continuity_notes=item.get("continuity_notes", ""),
                seed=item.get("seed"),
                reference_images=reference_images,
                reference_videos=[reference_video] if reference_video else [],
                reference_audio=reference_audio,
                reference_audio_by_character={
                    name: audio.get(name.lower())
                    for name in speaking
                    if audio.get(name.lower())
                },
                reference_video_by_character={
                    name: videos.get(name.lower())
                    for name in shot_characters
                    if videos.get(name.lower())
                },
                speaking_characters=speaking,
                speech_text=speech_text,
                reference_bindings=reference_bindings,
                width=self.H3_WIDTH,
                height=self.H3_HEIGHT,
                fps=self.H3_FPS,
                frames_per_shot=self.H3_FRAMES,
            )
            shots.append(shot)

        return shots

    def unload(self):
        self.model.unload()
