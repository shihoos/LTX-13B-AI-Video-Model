from __future__ import annotations

from pathlib import Path


class H3ReferenceBinding:
    """
    Converts the application's character/reference model into
    H3's explicit reference-label language.

    H3 uses 0-based API slots but 1-based prompt labels:
        ref_image_0 -> <Picture 1>
        ref_video_0 -> <Video 1>
        ref_audio_0 -> <Audio 1>
    """

    MAX_IMAGES = 9
    MAX_VIDEOS = 3
    MAX_AUDIO = 3

    @classmethod
    def collect(
        cls,
        characters: list,
        shot_character_names: list[str],
    ) -> dict:
        lookup = {
            character.name.lower(): character
            for character in characters
        }

        images: list[str] = []
        videos: list[str] = []
        audio: list[str] = []

        image_owners: dict[str, list[str]] = {}
        video_owners: dict[str, list[str]] = {}
        audio_owners: dict[str, list[str]] = {}

        for name in shot_character_names:
            character = lookup.get(
                str(name).lower()
            )

            if character is None:
                continue

            for path in character.normalized_reference_paths():
                if (
                    path not in images
                    and len(images) < cls.MAX_IMAGES
                ):
                    images.append(path)

                image_owners.setdefault(
                    path,
                    [],
                ).append(character.name)

            for path in character.normalized_video_paths():
                if (
                    path not in videos
                    and len(videos) < cls.MAX_VIDEOS
                ):
                    videos.append(path)

                video_owners.setdefault(
                    path,
                    [],
                ).append(character.name)

            for path in character.normalized_audio_paths():
                if (
                    path not in audio
                    and len(audio) < cls.MAX_AUDIO
                ):
                    audio.append(path)

                audio_owners.setdefault(
                    path,
                    [],
                ).append(character.name)

        picture_lines = [
            (
                f"<Picture {index}> is the canonical "
                f"visual identity reference for "
                f"{', '.join(image_owners.get(path, []))}."
            )
            for index, path in enumerate(
                images,
                start=1,
            )
        ]

        video_lines = [
            (
                f"<Video {index}> is the motion/camera "
                f"reference for "
                f"{', '.join(video_owners.get(path, []))}."
            )
            for index, path in enumerate(
                videos,
                start=1,
            )
        ]

        audio_lines = [
            (
                f"<Audio {index}> is the voice/audio "
                f"identity reference for "
                f"{', '.join(audio_owners.get(path, []))}."
            )
            for index, path in enumerate(
                audio,
                start=1,
            )
        ]

        return {
            "images": images,
            "videos": videos,
            "audio": audio,
            "prompt_lines": (
                picture_lines
                + video_lines
                + audio_lines
            ),
        }

    @classmethod
    def prompt_block(
        cls,
        characters: list,
        shot_character_names: list[str],
    ) -> tuple[str, dict]:

        data = cls.collect(
            characters=characters,
            shot_character_names=shot_character_names,
        )

        return (
            "\n".join(
                data["prompt_lines"]
            ),
            data,
        )
