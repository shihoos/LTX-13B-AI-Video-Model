from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from execution.shot_executor import ShotExecutor
from pipeline.h3_scene_continuity import H3SceneContinuity


class ProductionRunner:
    """
    Scene-level H3 production runner.

    First shot only:
        H3_HardMode_R2V

    Multi-shot scene:
        H3_HardMode_Chained

    This prevents the old bug where later shots were still submitted to
    Ref2VA while merely carrying an unused `continuity_start_image` field.
    """

    def __init__(
        self,
        project_root: Path,
        comfy_clients: dict[int, object],
    ):
        self.project_root = Path(project_root)
        self.clients = dict(comfy_clients)

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

        self.continuity = H3SceneContinuity(
            self.project_root
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
            comfy_client=self.clients[gpu_id],
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
                f"Scene {scene_id} contains no shots."
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

        # A single-shot scene uses native Ref2VA.
        if len(shots) == 1:
            result = executor.execute_native_ref2va(
                shots[0],
                output_dir,
            )

            self.continuity.prepare_next_shot(
                result,
                scene_id,
                shots[0]["shot_id"],
            )

            return result

        # Multi-shot scenes use the complete upstream Hard Mode Chained graph.
        master = executor.execute_hardmode_chained(
            shots,
            output_dir,
        )

        # Persist one scene checkpoint. The chained workflow itself is the
        # source of last-frame continuity between individual shots.
        self.continuity.prepare_next_shot(
            master,
            scene_id,
            f"{scene_id}_master",
        )

        return master

    @staticmethod
    def concat(
        videos: list[Path],
        destination: Path,
    ) -> Path:
        if not videos:
            raise ValueError("No videos supplied.")

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        manifest = destination.with_suffix(".txt")
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
                + result.stderr[-4000:]
            )

        return destination

    def run(
        self,
        production_plan: dict[str, Any],
    ) -> Path:
        shots = production_plan.get("shots", [])
        if not shots:
            raise RuntimeError(
                "Production plan contains no shots."
            )

        scenes: dict[str, list[dict[str, Any]]] = {}

        for shot in shots:
            scenes.setdefault(
                str(shot["scene_id"]),
                [],
            ).append(shot)

        gpu_ids = list(self.clients)
        if not gpu_ids:
            raise RuntimeError(
                "No GPU workers configured."
            )

        scene_masters: list[Path] = []

        for index, (scene_id, scene_shots) in enumerate(
            sorted(
                scenes.items(),
                key=lambda item: (
                    min(
                        int(shot.get("order", 0))
                        for shot in item[1]
                    ),
                    item[0],
                ),
            )
        ):
            # Scene-parallel scheduling remains across independent scenes.
            gpu_id = gpu_ids[index % len(gpu_ids)]

            master = self.run_scene(
                gpu_id=gpu_id,
                scene_id=scene_id,
                shots=scene_shots,
            )

            scene_masters.append(master)

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
