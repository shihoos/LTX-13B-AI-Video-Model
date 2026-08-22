from __future__ import annotations

import subprocess
from pathlib import Path

from execution.shot_executor import (
    ShotExecutor,
)
from pipeline.h3_scene_continuity import (
    H3SceneContinuity,
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

        self.continuity = (
            H3SceneContinuity(
                self.project_root
            )
        )

    def run_scene(
        self,
        gpu_id: int,
        scene_id: str,
        shots: list[dict],
    ) -> Path:

        if gpu_id not in self.clients:
            raise RuntimeError(
                f"GPU worker {gpu_id} is not configured."
            )

        client = self.clients[gpu_id]

        input_dir = (
            self.input_root
            / f"gpu_{gpu_id}"
            / str(scene_id)
        )

        output_dir = (
            self.output_root
            / f"gpu_{gpu_id}"
            / str(scene_id)
        )

        input_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        rendered: list[Path] = []

        previous_last_frame = None

        for index, shot in enumerate(
            shots
        ):

            # The first shot is the identity-casting
            # native Ref2VA shot.
            if index != 0:
                # The next-stage interface keeps the previous
                # frame available to the scene manager.
                # The actual H3 chained workflow consumes it.
                if previous_last_frame:
                    shot = dict(shot)
                    shot[
                        "continuity_start_image"
                    ] = str(
                        previous_last_frame
                    )

            executor = (
                ShotExecutor(
                    comfy_client=client,
                    project_root=self.project_root,
                    comfy_input_dir=input_dir,
                )
            )

            result = (
                executor
                .execute_native_ref2va(
                    shot=shot,
                    output_dir=output_dir,
                )
            )

            rendered.append(
                result
            )

            previous_last_frame = (
                self.continuity
                .prepare_next_shot(
                    video_path=result,
                    scene_id=scene_id,
                    shot_id=shot[
                        "shot_id"
                    ],
                )
            )

        scene_master = (
            output_dir
            / f"{scene_id}.mp4"
        )

        self.concat(
            rendered,
            scene_master,
        )

        return scene_master

    @staticmethod
    def concat(
        videos: list[Path],
        destination: Path,
    ) -> Path:

        if not videos:
            raise ValueError(
                "No video files supplied."
            )

        manifest = (
            destination.with_suffix(
                ".txt"
            )
        )

        manifest.write_text(
            "\n".join(
                f"file '{path.resolve()}'"
                for path in videos
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
    def upscale_720p(
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
                "720p export failed:\n"
                + result.stderr[-4000:]
            )

        return destination

    def run(
        self,
        production_plan: dict,
    ) -> Path:

        shots = production_plan.get(
            "shots",
            [],
        )

        if not shots:
            raise RuntimeError(
                "Production plan contains no shots."
            )

        scenes: dict[str, list[dict]] = {}

        for shot in shots:
            scenes.setdefault(
                str(
                    shot["scene_id"]
                ),
                [],
            ).append(
                shot
            )

        gpu_ids = list(
            self.clients
        )

        if not gpu_ids:
            raise RuntimeError(
                "No GPU workers configured."
            )

        scene_masters = []

        for scene_index, (
            scene_id,
            scene_shots,
        ) in enumerate(
            scenes.items()
        ):

            gpu_id = gpu_ids[
                scene_index
                % len(gpu_ids)
            ]

            master = (
                self.run_scene(
                    gpu_id=gpu_id,
                    scene_id=scene_id,
                    shots=scene_shots,
                )
            )

            scene_masters.append(
                master
            )

        master = (
            self.output_root
            / "master_native.mp4"
        )

        self.concat(
            scene_masters,
            master,
        )

        final = (
            self.project_root
            / "data"
            / "production"
            / "final_h3_720p.mp4"
        )

        return self.upscale_720p(
            source=master,
            destination=final,
        )
