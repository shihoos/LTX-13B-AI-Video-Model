from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from execution.shot_executor import (
    ShotExecutor,
)

from postprocess.final_export import (
    FinalExporter,
)

from postprocess.h3_regenerate_2k import (
    H3Regenerate2K,
)

from planner.config import (
    H3_REGENERATE_2K_ENABLED,
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
        gpu_id,
        scene_id,
    ):

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
        gpu_id,
        scene_id,
        shots,
    ):

        executor = self._executor(
            gpu_id,
            scene_id,
        )

        shots = sorted(
            shots,
            key=lambda item:
                int(
                    item.get(
                        "order",
                        0,
                    )
                ),
        )

        output_dir = (
            self.output_root
            / f"gpu_{gpu_id}"
            / scene_id
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # At the moment we use Ref2VA directly
        # for each shot.
        #
        # This guarantees correct reference routing
        # before adding a custom multishot node.
        shot_outputs = []

        for shot in shots:

            result = (
                executor
                .execute_native_ref2va(
                    shot,
                    output_dir,
                )
            )

            shot_outputs.append(
                result
            )

        return shot_outputs

    @staticmethod
    def concat(
        videos,
        destination,
    ):

        import subprocess

        if not videos:
            raise ValueError(
                "No videos supplied."
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

        for video in videos:

            path = (
                Path(video)
                .resolve()
            )

            escaped = str(
                path
            ).replace(
                "'",
                "'\\''",
            )

            lines.append(
                f"file '{escaped}'"
            )

        manifest.write_text(
            "\n".join(lines)
            + "\n",
            encoding="utf-8",
        )

        result = subprocess.run(
            [
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
            ],
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

        return destination

    def run(
        self,
        production_plan: dict[str, Any],
    ):

        shots = production_plan.get(
            "shots",
            [],
        )

        if not shots:
            raise RuntimeError(
                "Production plan contains no shots."
            )

        scenes = {}

        for shot in shots:

            scene_id = str(
                shot.get(
                    "scene_id",
                    "",
                )
            )

            if not scene_id:
                raise RuntimeError(
                    "Shot missing scene_id."
                )

            scenes.setdefault(
                scene_id,
                [],
            ).append(
                shot
            )

        scene_jobs = []

        for scene_id, scene_shots in (
            scenes.items()
        ):

            scene_jobs.append(
                SimpleNamespace(
                    scene_id=scene_id,
                    shots=scene_shots,
                )
            )

        scheduler = GPUScheduler(
            gpu_ids=sorted(
                self.clients.keys()
            )
        )

        def worker(
            gpu_id,
            job,
        ):

            return self.run_scene(
                gpu_id=gpu_id,
                scene_id=job.scene_id,
                shots=job.shots,
            )

        results = scheduler.run(
            scene_jobs,
            worker,
        )

        # Restore narrative scene order.
        results.sort(
            key=lambda item:
                next(
                    int(
                        shot.get(
                            "order",
                            0,
                        )
                    )
                    for shot in scenes[
                        item[1]
                    ]
                )
        )

        scene_videos = []

        for _, scene_id, paths in results:

            scene_videos.extend(
                paths
            )

        native_master = (
            self.output_root
            / "master_native.mp4"
        )

        self.concat(
            scene_videos,
            native_master,
        )

        master_for_export = (
            native_master
        )

        # Official H3 regeneration.
        if H3_REGENERATE_2K_ENABLED:

            regenerated = (
                self.output_root
                / "master_h3_2k.mp4"
            )

            regenerater = (
                H3Regenerate2K()
            )

            master_for_export = (
                regenerater.regenerate(
                    native_master,
                    regenerated,
                    prompt=production_plan.get(
                        "story",
                        "",
                    ),
                )
            )

        final = (
            self.project_root
            / "data"
            / "production"
            / "final_h3_720p.mp4"
        )

        return FinalExporter.export_720p(
            master_for_export,
            final,
        )
