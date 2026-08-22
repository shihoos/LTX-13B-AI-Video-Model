from __future__ import annotations

import subprocess
from pathlib import Path

from execution.shot_executor import (
    ShotExecutor,
)
from scheduler.gpu_scheduler import (
    GPUScheduler,
)


class ProductionRunner:

    def __init__(
        self,
        project_root: Path,
        comfy_clients: dict[int, object],
    ):
        self.project_root = Path(
            project_root
        )

        self.clients = dict(
            comfy_clients
        )

        self.comfy_input_root = (
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

    def _generate_one(
        self,
        gpu_id: int,
        shot: dict,
    ) -> Path:

        client = self.clients[
            gpu_id
        ]

        executor = (
            ShotExecutor(
                comfy_client=client,
                project_root=(
                    self.project_root
                ),
                comfy_input_dir=(
                    self.comfy_input_root
                    / f"gpu_{gpu_id}"
                ),
            )
        )

        native = (
            int(
                shot.get(
                    "order",
                    1,
                )
            )
            == 1
        )

        return executor.execute(
            shot=shot,
            output_dir=(
                self.output_dir
                / f"gpu_{gpu_id}"
            ),
            native_ref2va=native,
        )

    @staticmethod
    def _concat(
        videos: list[Path],
        destination: Path,
    ) -> Path:

        manifest = (
            destination
            .with_suffix(".txt")
        )

        manifest.write_text(
            "\n".join(
                "file "
                f"'{video.resolve()}'"
                for video in videos
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
            "-c",
            "copy",
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
                "FFmpeg concat failed:\n"
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

        if not self.clients:
            raise RuntimeError(
                "No ComfyUI GPU workers "
                "were configured."
            )

        shots = list(
            production_plan.get(
                "shots",
                [],
            )
        )

        if not shots:
            raise ValueError(
                "No shots in production plan."
            )

        failures = []

        if len(
            self.clients
        ) == 1:

            gpu_id = next(
                iter(
                    self.clients
                )
            )

            for shot in shots:
                try:
                    self._generate_one(
                        gpu_id,
                        shot,
                    )
                except Exception as error:
                    failures.append(
                        (
                            gpu_id,
                            shot["shot_id"],
                            str(error),
                        )
                    )

        else:

            scheduler = (
                GPUScheduler(
                    gpu_ids=list(
                        self.clients
                    )
                )
            )

            failures = (
                scheduler.run(
                    shots,
                    self._generate_one,
                )
            )

        if failures:
            raise RuntimeError(
                "H3 generation failures:\n"
                + "\n".join(
                    str(item)
                    for item in failures
                )
            )

        generated = sorted(
            self.output_dir.rglob(
                "*.mp4"
            )
        )

        if not generated:
            raise RuntimeError(
                "No H3 output videos were found."
            )

        master = (
            self.output_dir
            / "master.mp4"
        )

        self._concat(
            generated,
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
