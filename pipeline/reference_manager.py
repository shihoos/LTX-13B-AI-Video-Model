from pathlib import Path

from execution.character_reference_processor import (
    CharacterReferenceProcessor,
)


PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)

ASSETS_DIR = (
    PROJECT_ROOT / "assets"
)

CHARACTERS_DIR = (
    ASSETS_DIR / "characters"
)

REFERENCES_DIR = (
    ASSETS_DIR / "references"
)

AUDIO_DIR = (
    ASSETS_DIR / "audio"
)

MUSIC_DIR = (
    ASSETS_DIR / "music"
)

GENERATED_CHARACTERS_DIR = (
    PROJECT_ROOT
    / "data"
    / "characters"
    / "generated"
)

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
}

AUDIO_EXTENSIONS = {
    ".mp3",
    ".wav",
    ".m4a",
    ".ogg",
}


class ReferenceManager:

    def __init__(self):

        self.processor = (
            CharacterReferenceProcessor(
                PROJECT_ROOT
            )
        )

        self.ensure_directories()

    def ensure_directories(self):

        directories = [
            CHARACTERS_DIR,
            REFERENCES_DIR,
            AUDIO_DIR,
            MUSIC_DIR,
            GENERATED_CHARACTERS_DIR,
        ]

        for directory in directories:

            directory.mkdir(
                parents=True,
                exist_ok=True,
            )

    def find_character(
        self,
        character_name: str,
    ):

        provided = (
            self.find_provided_character(
                character_name
            )
        )

        if provided is not None:

            return provided

        return (
            self.find_generated_character(
                character_name
            )
        )

    def find_provided_character(
        self,
        character_name: str,
    ):

        return self._find_file(
            CHARACTERS_DIR,
            character_name,
            IMAGE_EXTENSIONS,
        )

    def find_generated_character(
        self,
        character_name: str,
    ):

        return self._find_file(
            GENERATED_CHARACTERS_DIR,
            character_name,
            IMAGE_EXTENSIONS,
        )

    def find_reference(
        self,
        reference_name: str,
    ):

        return self._find_file(
            REFERENCES_DIR,
            reference_name,
            IMAGE_EXTENSIONS,
        )

    def find_audio(
        self,
        audio_name: str,
    ):

        return self._find_file(
            AUDIO_DIR,
            audio_name,
            AUDIO_EXTENSIONS,
        )

    def find_music(
        self,
        music_name: str,
    ):

        return self._find_file(
            MUSIC_DIR,
            music_name,
            AUDIO_EXTENSIONS,
        )

    def _find_file(
        self,
        directory: Path,
        name: str,
        extensions: set,
    ):

        if not directory.is_dir():

            return None

        normalized_name = (
            str(name)
            .strip()
            .lower()
            .replace(" ", "_")
        )

        for file_path in (
            directory.iterdir()
        ):

            if not file_path.is_file():

                continue

            if (
                file_path.suffix.lower()
                not in extensions
            ):

                continue

            file_name = (
                file_path.stem
                .strip()
                .lower()
                .replace(" ", "_")
            )

            if (
                file_name
                == normalized_name
            ):

                return file_path

        return None

    def character_exists(
        self,
        character_name: str,
    ) -> bool:

        return (
            self.find_character(
                character_name
            )
            is not None
        )

    def character_asset_names(
        self,
    ) -> list[str]:

        if not CHARACTERS_DIR.is_dir():

            return []

        return sorted(
            file_path.stem
            for file_path
            in CHARACTERS_DIR.iterdir()
            if (
                file_path.is_file()
                and
                file_path.suffix.lower()
                in IMAGE_EXTENSIONS
            )
        )

    def _prepare_identity_reference(
        self,
        character_name: str,
        source_path: Path,
    ) -> dict:

        return self.processor.process(
            character_name=character_name,
            source_path=source_path,
        )

    def get_character_source(
        self,
        character_name: str,
    ) -> dict:

        provided = (
            self.find_provided_character(
                character_name
            )
        )

        if provided is not None:

            processed = (
                self._prepare_identity_reference(
                    character_name,
                    provided,
                )
            )

            return {
                "mode":
                    "provided",

                "path":
                    processed["path"],

                "mask_path":
                    processed["mask_path"],

                "metadata_path":
                    processed[
                        "metadata_path"
                    ],

                "source_path":
                    str(provided),

                "message": (
                    "Use the provided character "
                    "reference after face-first "
                    "identity preprocessing."
                ),
            }

        generated = (
            self.find_generated_character(
                character_name
            )
        )

        if generated is not None:

            processed = (
                self._prepare_identity_reference(
                    character_name,
                    generated,
                )
            )

            return {
                "mode":
                    "generated",

                "path":
                    processed["path"],

                "mask_path":
                    processed["mask_path"],

                "metadata_path":
                    processed[
                        "metadata_path"
                    ],

                "source_path":
                    str(generated),

                "message": (
                    "Reuse the generated character "
                    "reference after face-first "
                    "identity preprocessing."
                ),
            }

        return {
            "mode":
                "missing",

            "path":
                None,

            "mask_path":
                None,

            "metadata_path":
                None,

            "source_path":
                None,

            "message":
                "No character reference exists.",
        }
