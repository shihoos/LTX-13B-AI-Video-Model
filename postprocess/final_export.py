from __future__ import annotations

import subprocess
from pathlib import Path

from planner.config import (
    FINAL_HEIGHT,
    FINAL_WIDTH,
)


class FinalExporter:

    @staticmethod
    def export_720p(
        source: Path,
        destination: Path,
    ) -> Path:

        source = Path(source)
        destination = Path(destination)

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        command = [
            "ffmpeg",
            "-y",

            "-i",
            str(source),

            "-vf",
            (
                f"scale={FINAL_WIDTH}:"
                f"{FINAL_HEIGHT}:"
                "flags=lanczos,"
                "setsar=1"
            ),

            "-c:v",
            "libx264",

            "-preset",
            "medium",

            "-crf",
            "17",

            "-pix_fmt",
            "yuv420p",

            "-c:a",
            "aac",

            "-b:a",
            "192k",

            "-movflags",
            "+faststart",

            str(destination),
        ]

        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(
                "Final 720p export failed:\n"
                + result.stderr[-5000:]
            )

        if not destination.is_file():
            raise RuntimeError(
                "FFmpeg completed but final "
                "video was not created."
            )

        return destination
