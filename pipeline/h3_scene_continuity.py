from __future__ import annotations

import subprocess
from pathlib import Path


class H3SceneContinuity:

    def __init__(
        self,
        project_root: Path,
    ):
        self.project_root = Path(
            project_root
        )

        self.root = (
            self.project_root
            / "data"
            / "production"
            / "continuity"
        )

        self.root.mkdir(
            parents=True,
            exist_ok=True,
        )

    def extract_last_frame(
        self,
        video_path: Path,
        scene_id: str,
        shot_id: str,
    ) -> Path:

        destination = (
            self.root
            / str(scene_id)
            / f"{shot_id}_last_frame.png"
        )

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        command = [
            "ffmpeg",
            "-y",
            "-sseof",
            "-0.05",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-update",
            "1",
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
                "Could not extract H3 last frame:\n"
                + result.stderr[-4000:]
            )

        if not destination.is_file():
            raise RuntimeError(
                "FFmpeg completed but did not create "
                f"the last frame: {destination}"
            )

        return destination

    def prepare_next_shot(
        self,
        video_path: Path,
        scene_id: str,
        shot_id: str,
    ) -> Path:

        return self.extract_last_frame(
            video_path=video_path,
            scene_id=scene_id,
            shot_id=shot_id,
        )
