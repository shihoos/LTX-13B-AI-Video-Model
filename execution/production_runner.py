from __future__ import annotations

import subprocess
from pathlib import Path

from execution.shot_executor import (
    ShotExecutor,
)


class ProductionRunner:

    def __init__(
        self,
        project_root: Path,
        comfy_client,
    ):
        self.project_root = (
            Path(project_root)
        )

        self.client = (
            comfy_client
        )

        self.comfy_input_dir = (
            self.project_root
            / "ComfyUI"
            / "input"
        )

        self.output_dir = (
            self.project_root
            / "data"
            / "production"
            / "h3"
        )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    @staticmethod
    def _concat(
        files: list[Path],
        destination: Path,
    ) -> Path:

        manifest = (
            destination.with_suffix(
                ".txt"
            )
        )

        manifest.write_text(
            "\n".join(
                f"file '{path.resolve()}'"
                for path in files
            )
            + "\n",
            encoding="utf-8",
        )

        command = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(manifest),
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

        manifest.unlink(
            missing_ok=True
        )

        if result.returncode != 0:
            raise RuntimeError(
                "FFmpeg assembly failed:\n"
                + result.stderr[-4000:]
            )

        return destination

    @staticmethod
    def _deliver_720p(
        source: Path,
        destination: Path,
    ) -> Path:

        command = [
            "ffmpeg",
            "-y",
            "-i",
            str(source),
            "-vf",
            (
                "scale=1280:720:"
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
                "720p delivery failed:\n"
                + result.stderr[-4000:]
            )

        return destination

    def run(
        self,
        production_plan: dict,
    ) -> Path:

        if not self.client.health_check():
            raise RuntimeError(
                "Cannot connect to ComfyUI at "
                f"{self.client.base_url}"
            )

        shots = production_plan.get(
            "shots",
            [],
        )

        if not shots:
            raise ValueError(
                "Production plan contains no shots."
            )

        executor = (
            ShotExecutor(
                comfy_client=self.client,
                project_root=self.project_root,
                comfy_input_dir=(
                    self.comfy_input_dir
                ),
            )
        )

        scene_outputs = {}

        for shot in shots:
            scene_id = shot[
                "scene_id"
            ]

            print(
                f"Generating "
                f"{shot['shot_id']} "
                f"with existing H3 workflow..."
            )

            video = executor.execute(
                shot,
                self.output_dir,
            )

            scene_outputs.setdefault(
                scene_id,
                [],
            ).append(
                video
            )

        mastered_scenes = []

        for scene_id, files in (
            scene_outputs.items()
        ):
            if len(files) == 1:
                master = files[0]
            else:
                master = (
                    self.output_dir
                    / f"{scene_id}_master.mp4"
                )

                self._concat(
                    files,
                    master,
                )

            mastered_scenes.append(
                master
            )

        if len(
            mastered_scenes
        ) == 1:
            master = mastered_scenes[0]
        else:
            master = (
                self.output_dir
                / "h3_master.mp4"
            )

            self._concat(
                mastered_scenes,
                master,
            )

        final = (
            self.project_root
            / "data"
            / "production"
            / "final_h3_720p.mp4"
        )

        return self._deliver_720p(
            master,
            final,
        )
