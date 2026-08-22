from __future__ import annotations

import re
from pathlib import Path

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".ogg", ".flac"}


class ReferenceManager:
    """
    H3 identity media manager.

    Preferred layout:

        assets/references/<character>/
            images/
                01.png
                02.png
                03.png
            videos/
                scene.mp4
            audio/
                voice.wav

    Legacy single-file layout is also supported:
        assets/characters/<character>.png
        assets/audio/<character>.wav
        assets/references/<character>.mp4

    No reference images are composited together. H3 receives independent
    reference slots so identity information is not mixed into one image.
    """

    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)

        self.assets = self.project_root / "assets"
        self.characters_dir = self.assets / "characters"
        self.references_dir = self.assets / "references"
        self.audio_dir = self.assets / "audio"

        for path in (
            self.characters_dir,
            self.references_dir,
            self.audio_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _key(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")

    @staticmethod
    def _sorted_files(directory: Path, extensions: set[str]) -> list[Path]:
        if not directory.is_dir():
            return []
        return sorted(
            p for p in directory.iterdir()
            if p.is_file() and p.suffix.lower() in extensions
        )

    def resolve_character(self, character_name: str) -> dict:
        key = self._key(character_name)

        images: list[Path] = []
        videos: list[Path] = []
        audio: list[Path] = []

        # Preferred directory structure.
        ref_root = self.references_dir / key
        images_dir = ref_root / "images"
        videos_dir = ref_root / "videos"
        audio_dir = ref_root / "audio"

        images.extend(self._sorted_files(images_dir, IMAGE_EXTENSIONS))
        videos.extend(self._sorted_files(videos_dir, VIDEO_EXTENSIONS))
        audio.extend(self._sorted_files(audio_dir, AUDIO_EXTENSIONS))

        # Legacy image files in assets/characters.
        for p in self._sorted_files(self.characters_dir, IMAGE_EXTENSIONS):
            if self._key(p.stem) == key:
                if p not in images:
                    images.append(p)

        # Legacy references/<character>.<ext>.
        for p in self._sorted_files(self.references_dir, IMAGE_EXTENSIONS | VIDEO_EXTENSIONS):
            if self._key(p.stem) == key:
                if p.suffix.lower() in IMAGE_EXTENSIONS and p not in images:
                    images.append(p)
                if p.suffix.lower() in VIDEO_EXTENSIONS and p not in videos:
                    videos.append(p)

        # Legacy assets/audio/<character>.<ext>.
        for p in self._sorted_files(self.audio_dir, AUDIO_EXTENSIONS):
            if self._key(p.stem) == key:
                if p not in audio:
                    audio.append(p)

        return {
            "reference_paths": [str(p) for p in images],
            "reference_video_path": str(videos[0]) if videos else None,
            "reference_audio_path": str(audio[0]) if audio else None,
        }

    def resolve_characters(self, characters: list) -> list:
        for character in characters:
            name = getattr(character, "name", "")
            result = self.resolve_character(name)

            refs = result["reference_paths"]

            # Explicit character values win.
            existing = list(getattr(character, "reference_paths", []) or [])
            if getattr(character, "reference_path", None):
                existing.insert(0, character.reference_path)

            merged = []
            for path in existing + refs:
                if path and path not in merged:
                    merged.append(path)

            character.reference_paths = merged
            character.reference_path = merged[0] if merged else None

            if not character.reference_audio_path:
                character.reference_audio_path = result["reference_audio_path"]

            if not character.reference_video_path:
                character.reference_video_path = result["reference_video_path"]

            character.reference_mode = "provided" if merged else "missing"

        return characters

    def validate(self, characters: list, *, require_images: bool = True) -> None:
        errors = []

        for character in characters:
            refs = list(getattr(character, "reference_paths", []) or [])
            if require_images and not refs:
                errors.append(
                    f"{character.name}: no reference image found"
                )

            if len(refs) > 9:
                errors.append(
                    f"{character.name}: {len(refs)} images available; "
                    "the H3 render lane accepts at most 9 per generation."
                )

            for path in refs:
                if not Path(path).is_file():
                    errors.append(
                        f"{character.name}: missing reference image: {path}"
                    )

            audio = getattr(character, "reference_audio_path", None)
            if audio and not Path(audio).is_file():
                errors.append(
                    f"{character.name}: missing reference audio: {audio}"
                )

            video = getattr(character, "reference_video_path", None)
            if video and not Path(video).is_file():
                errors.append(
                    f"{character.name}: missing reference video: {video}"
                )

        if errors:
            raise RuntimeError(
                "H3 reference validation failed:\n- "
                + "\n- ".join(errors)
            )
