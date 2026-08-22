from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from execution.shot_executor import (
    ShotExecutor,
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

        self.input_root = (
            self.project_root
            / "ComfyUI"
            / "input"
        )

        self.output_root = (
            self.project_root
            / "data"
            / "production"
            / "h3"
        )

        self.output_root.mkdir(
            parents=True,
            exist_ok=True,
        )

    def _executor(
        self,
        gpu_id: int,
        scene_id: str,
    ) -> ShotExecutor:

        input_dir = (
            self.input_root
            / f"gpu_{gpu_id}"
            / str(scene_id)
        )

        input_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        return ShotExecutor(
            comfy_client=self.clients[
                gpu_id
            ],
            project_root=self.project_root,
            comfy_input_dir=input_dir,
        )

    def run_scene(
        self,
        gpu_id: int,
        scene_id: str,
        shots: list[dict[str, Any]],
    ) -> Path:

        if gpu_id not in self.clients:
            raise RuntimeError(
                f"GPU worker {gpu_id} is not configured."
            )

        if not shots:
            raise ValueError(
                f"Scene {scene_id} has no shots."
            )

        shots = sorted(
            shots,
            key=lambda item: int(
                item.get(
                    "order",
                    0,
                )
            ),
        )

        output_dir = (
            self.output_root
            / f"gpu_{gpu_id}"
            / str(scene_id)
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        executor = self._executor(
            gpu_id,
            scene_id,
        )

        if len(shots) == 1:
            return (
                executor.execute_native_ref2va(
                    shots[0],
                    output_dir,
                )
            )

        return (
            executor.execute_hardmode_chained(
                shots,
                output_dir,
            )
        )

    @staticmethod
    def concat(
        videos: list[Path],
        destination: Path,
    ) -> Path:

        if not videos:
            raise ValueError(
                "No videos supplied for concat."
            )

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        manifest = (
            destination.with_suffix(
                ".txt"
            )
        )

        lines = []

        for path in videos:
            escaped = (
                str(
                    path.resolve()
                )
                .replace(
                    "'",
                    "'\\''"
                )
            )

            lines.append(
                f"file '{escaped}'"
            )

        manifest.write_text(
            "\n".join(lines)
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
                + result.stderr[-5000:]
            )

        if not destination.is_file():
            raise RuntimeError(
                "FFmpeg reported success but "
                "the concatenated video does not exist."
            )

        return destination

    @staticmethod
    def upscale_720p(
        source: Path,
        destination: Path,
    ) -> Path:

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
            "scale=1280:720:flags=lanczos,setsar=1",
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
                "720p export failed:\n"
                + result.stderr[-5000:]
            )

        return destination

    def run(
        self,
        production_plan: dict[str, Any],
    ) -> Path:

        shots = production_plan.get(
            "shots",
            []
        )

        if not shots:
            raise RuntimeError(
                "Production plan contains no shots."
            )

        scenes: dict[
            str,
            list[dict[str, Any]]
        ] = {}

        for shot in shots:
            scene_id = str(
                shot.get(
                    "scene_id",
                    "",
                )
            )

            if not scene_id:
                raise RuntimeError(
                    "Shot is missing scene_id."
                )

            scenes.setdefault(
                scene_id,
                []
            ).append(
                shot
            )

        gpu_ids = sorted(
            self.clients.keys()
        )

        if not gpu_ids:
            raise RuntimeError(
                "No ComfyUI GPU workers configured."
            )

        ordered_scenes = sorted(
            scenes.items(),
            key=lambda item: min(
                int(
                    shot.get(
                        "order",
                        0,
                    )
                )
                for shot in item[1]
            ),
        )

        scene_masters = []

        for index, (
            scene_id,
            scene_shots,
        ) in enumerate(
            ordered_scenes
        ):

            gpu_id = gpu_ids[
                index % len(gpu_ids)
            ]

            master = self.run_scene(
                gpu_id=gpu_id,
                scene_id=scene_id,
                shots=scene_shots,
            )

            scene_masters.append(
                master
            )

        native_master = (
            self.output_root
            / "master_native.mp4"
        )

        self.concat(
            scene_masters,
            native_master,
        )

        final = (
            self.project_root
            / "data"
            / "production"
            / "final_h3_720p.mp4"
        )

        return self.upscale_720p(
            source=native_master,
            destination=final,
        )
