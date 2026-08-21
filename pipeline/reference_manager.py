from pathlib import Path


PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)


ASSETS_DIR = (
    PROJECT_ROOT
    / "assets"
)

CHARACTERS_DIR = (
    ASSETS_DIR
    / "characters"
)

REFERENCES_DIR = (
    ASSETS_DIR
    / "references"
)

AUDIO_DIR = (
    ASSETS_DIR
    / "audio"
)

MUSIC_DIR = (
    ASSETS_DIR
    / "music"
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

    """
    Resolves character and media assets.

    Character resolution policy:

        assets/characters/<character-name>.*

    Existing assets are always preferred.

    Automatic image generation is intentionally not performed here
    because the current repository does not contain a configured
    still-image generation model/workflow.
    """

    def __init__(self):

        self.ensure_directories()

    def ensure_directories(self):

        directories = [
            CHARACTERS_DIR,
            REFERENCES_DIR,
            AUDIO_DIR,
            MUSIC_DIR,
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

        return self._find_file(
            directory=CHARACTERS_DIR,
            name=character_name,
            extensions=IMAGE_EXTENSIONS,
        )

    def find_reference(
        self,
        reference_name: str,
    ):

        return self._find_file(
            directory=REFERENCES_DIR,
            name=reference_name,
            extensions=IMAGE_EXTENSIONS,
        )

    def find_audio(
        self,
        audio_name: str,
    ):

        return self._find_file(
            directory=AUDIO_DIR,
            name=audio_name,
            extensions=AUDIO_EXTENSIONS,
        )

    def find_music(
        self,
        music_name: str,
    ):

        return self._find_file(
            directory=MUSIC_DIR,
            name=music_name,
            extensions=AUDIO_EXTENSIONS,
        )

    def character_asset_names(self) -> list[str]:

        names = []

        if not CHARACTERS_DIR.exists():

            return names

        for file_path in sorted(
            CHARACTERS_DIR.iterdir()
        ):

            if not file_path.is_file():

                continue

            if (
                file_path.suffix.lower()
                not in IMAGE_EXTENSIONS
            ):

                continue

            names.append(
                file_path.stem
            )

        return names

    def _find_file(
        self,
        directory: Path,
        name: str,
        extensions: set,
    ):

        if not directory.exists():

            return None

        normalized_name = (
            name
            .strip()
            .lower()
            .replace(" ", "_")
        )

        for file_path in sorted(
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

    def get_character_source(
        self,
        character_name: str,
    ) -> dict:

        reference = (
            self.find_character(
                character_name
            )
        )

        if reference is not None:

            return {
                "mode": "provided",
                "path": str(
                    reference
                ),
                "message": (
                    "Use the provided "
                    "character reference."
                ),
            }

        return {
            "mode": "missing",
            "path": None,
            "message": (
                "No character reference exists "
                f"for '{character_name}'. "
                "A still-image generation workflow "
                "must be configured before this "
                "character can be used with the "
                "current I2V pipeline."
            ),
        }
