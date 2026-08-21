from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


class CharacterReferenceProcessor:

    """
    Prepare a user-provided character image for LTX I2V.

    The original image is never modified.

    The processed identity reference:
      - detects the largest/most central face,
      - keeps face, hair, neck and limited shoulder context,
      - suppresses the original scene/background,
      - preserves aspect ratio,
      - outputs exactly 768x432,
      - never stretches the person.
    """

    WIDTH = 768
    HEIGHT = 432
    BACKGROUND = (128, 128, 128)

    def __init__(self, project_root: Path):

        self.root = (
            Path(project_root)
            / "data"
            / "characters"
            / "processed"
        )

        self.root.mkdir(
            parents=True,
            exist_ok=True,
        )

    @staticmethod
    def _safe_name(name: str) -> str:

        value = "".join(
            character
            if character.isalnum()
            else "_"
            for character in name.strip().lower()
        )

        value = "_".join(
            part
            for part in value.split("_")
            if part
        )

        if not value:
            raise ValueError(
                "Character name cannot be empty."
            )

        return value

    def output_path(
        self,
        character_name: str,
    ) -> Path:

        return (
            self.root
            / self._safe_name(character_name)
            / "identity_reference.png"
        )

    def metadata_path(
        self,
        character_name: str,
    ) -> Path:

        return (
            self.root
            / self._safe_name(character_name)
            / "metadata.json"
        )

    @staticmethod
    def _detect_face(
        image: Image.Image,
    ) -> tuple[int, int, int, int]:

        array = np.asarray(
            image.convert("RGB")
        )

        gray = cv2.cvtColor(
            array,
            cv2.COLOR_RGB2GRAY,
        )

        detector = cv2.CascadeClassifier(
            cv2.data.haarcascades
            + "haarcascade_frontalface_default.xml"
        )

        if detector.empty():
            raise RuntimeError(
                "OpenCV face detector could not be loaded."
            )

        faces = detector.detectMultiScale(
            gray,
            scaleFactor=1.08,
            minNeighbors=5,
            minSize=(
                max(32, image.width // 20),
                max(32, image.height // 20),
            ),
        )

        if len(faces) == 0:
            raise RuntimeError(
                "No face was detected in the provided "
                "character reference. Use a clear image "
                "with the character's face visible."
            )

        center_x = image.width / 2
        center_y = image.height / 2

        def score(face):

            x, y, width, height = face

            area = width * height

            distance = (
                (
                    x + width / 2 - center_x
                ) ** 2
                + (
                    y + height / 2 - center_y
                ) ** 2
            ) ** 0.5

            return area - distance * 0.15

        x, y, width, height = max(
            faces,
            key=score,
        )

        return (
            int(x),
            int(y),
            int(width),
            int(height),
        )

    @staticmethod
    def _crop_identity(
        image: Image.Image,
        face_box: tuple[int, int, int, int],
    ) -> tuple[
        Image.Image,
        tuple[int, int, int, int],
    ]:

        x, y, width, height = face_box

        left = max(
            0,
            int(x - width * 1.25),
        )

        top = max(
            0,
            int(y - height * 1.15),
        )

        right = min(
            image.width,
            int(x + width * 2.25),
        )

        bottom = min(
            image.height,
            int(y + height * 2.70),
        )

        if right <= left or bottom <= top:
            raise RuntimeError(
                "Detected face produced an invalid "
                "identity crop."
            )

        crop = image.crop(
            (
                left,
                top,
                right,
                bottom,
            )
        )

        local_face = (
            x - left,
            y - top,
            width,
            height,
        )

        return crop, local_face

    @classmethod
    def _neutralize_background(
        cls,
        crop: Image.Image,
        face_box: tuple[int, int, int, int],
    ) -> Image.Image:

        width, height = crop.size

        mask = Image.new(
            "L",
            (
                width,
                height,
            ),
            0,
        )

        draw = ImageDraw.Draw(mask)

        x, y, face_width, face_height = face_box

        center_x = x + face_width / 2
        center_y = y + face_height * 1.25

        radius_x = face_width * 1.75
        radius_y = face_height * 2.45

        draw.ellipse(
            (
                center_x - radius_x,
                center_y - radius_y,
                center_x + radius_x,
                center_y + radius_y,
            ),
            fill=255,
        )

        mask = mask.filter(
            ImageFilter.GaussianBlur(
                radius=max(
                    4,
                    int(
                        min(width, height)
                        * 0.025
                    ),
                )
            )
        )

        background = Image.new(
            "RGB",
            (
                width,
                height,
            ),
            cls.BACKGROUND,
        )

        return Image.composite(
            crop.convert("RGB"),
            background,
            mask,
        )

    @classmethod
    def _letterbox(
        cls,
        image: Image.Image,
    ) -> Image.Image:

        source = image.convert("RGB")

        scale = min(
            cls.WIDTH / source.width,
            cls.HEIGHT / source.height,
        )

        width = max(
            1,
            round(source.width * scale),
        )

        height = max(
            1,
            round(source.height * scale),
        )

        resized = source.resize(
            (
                width,
                height,
            ),
            Image.Resampling.LANCZOS,
        )

        canvas = Image.new(
            "RGB",
            (
                cls.WIDTH,
                cls.HEIGHT,
            ),
            cls.BACKGROUND,
        )

        canvas.paste(
            resized,
            (
                (cls.WIDTH - width) // 2,
                (cls.HEIGHT - height) // 2,
            ),
        )

        return canvas

    def process(
        self,
        character_name: str,
        source_path: str | Path,
    ) -> Path:

        source = Path(source_path)

        if not source.is_file():
            raise FileNotFoundError(
                "Character reference does not exist: "
                f"{source}"
            )

        destination = self.output_path(
            character_name
        )

        metadata_path = self.metadata_path(
            character_name
        )

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        stat = source.stat()

        if (
            destination.is_file()
            and destination.stat().st_size > 0
            and metadata_path.is_file()
        ):

            try:

                metadata = json.loads(
                    metadata_path.read_text(
                        encoding="utf-8"
                    )
                )

                if (
                    metadata.get("source_path")
                    == str(source.resolve())
                    and metadata.get("source_size")
                    == stat.st_size
                    and metadata.get("source_mtime_ns")
                    == stat.st_mtime_ns
                ):

                    return destination

            except (
                OSError,
                ValueError,
                TypeError,
            ):
                pass

        with Image.open(source) as opened:

            image = opened.convert("RGB")

            face_box = self._detect_face(
                image
            )

            crop, local_face = (
                self._crop_identity(
                    image,
                    face_box,
                )
            )

            processed = (
                self._neutralize_background(
                    crop,
                    local_face,
                )
            )

            output = self._letterbox(
                processed
            )

            output.save(
                destination,
                format="PNG",
            )

        metadata_path.write_text(
            json.dumps(
                {
                    "character_name":
                        character_name,
                    "source_path":
                        str(source.resolve()),
                    "source_size":
                        stat.st_size,
                    "source_mtime_ns":
                        stat.st_mtime_ns,
                    "target_width":
                        self.WIDTH,
                    "target_height":
                        self.HEIGHT,
                    "identity_priority":
                        "face_first",
                    "background":
                        "neutralized",
                    "aspect_ratio_preserved":
                        True,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        return destination
