from __future__ import annotations

import subprocess
from pathlib import Path

from execution.comfy_client import ComfyClient
from execution.shot_executor import ShotExecutor


class ProductionRunner:
    """
    H3-only production runner.

    Shots are grouped by scene because H3's memory sampler keeps one stable
    reference bank and one reference-video lane across a chain. Different
    scenes can therefore use different casts/references safely.
    """

    def __init__(
        self,
        project_root: Path,
        gpu_urls: dict[int, str],
    ):
        self.project_root = Path(project_root)
        self.gpu_urls = {
            int(k): str(v).rstrip("/")
            for k, v in gpu_urls.items()
        }
        if not self.gpu_urls:
            raise ValueError("At least one H3 ComfyUI worker is required.")

        self.client = ComfyClient(
            base_url=self.gpu_urls[sorted(self.gpu_urls)[0]]
        )

        self.comfy_input_dir = (
            self.project_root / "ComfyUI" / "input"
        )
        self.output_dir = (
            self.project_root / "data" / "production" / "h3"
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _ffmpeg_scale(
        source: Path,
        destination: Path,
        width: int = 1280,
        height: int = 720,
    ) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)

        command = [
            "ffmpeg",
            "-y",
            "-i",
            str(source),
            "-vf",
            (
                f"scale={width}:{height}:"
                "flags=lanczos,"
                "setsar=1"
            ),
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "16",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            str(destination),
        ]

        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if completed.returncode != 0:
            raise RuntimeError(
                "FFmpeg 720p export failed:\n"
                + completed.stderr[-4000:]
            )

        return destination

    def _object_info(self) -> dict:
        info = self.client.get_object_info()

        required = [
            "H3ModelLoaderAny",
            "H3ClipLoaderAny",
            "H3MultishotMemorySampler",
            "H3ReferenceAudio",
        ]

        missing = [
            name
            for name in required
            if name not in info
        ]

        if missing:
            raise RuntimeError(
                "H3 runtime is incomplete. Missing nodes:\n- "
                + "\n- ".join(missing)
            )

        return info

    def run(self, production_plan: dict) -> Path:
        if not self.client.health_check():
            raise RuntimeError(
                f"Cannot connect to H3 ComfyUI: {self.client.base_url}"
            )

        object_info = self._object_info()

        shots = production_plan.get("shots") or []
        if not shots:
            raise ValueError("Production plan has no shots.")

        scenes: dict[str, list[dict]] = {}
        for shot in shots:
            scenes.setdefault(
                str(shot["scene_id"]),
                []
            ).append(shot)

        executor = ShotExecutor(
            comfy_client=self.client,
            project_root=self.project_root,
            comfy_input_dir=self.comfy_input_dir,
        )

        scene_outputs: list[Path] = []

        for scene_id, scene_shots in scenes.items():
            print(f"\n=== H3 SCENE: {scene_id} ===")
            output = executor.execute_scene(
                scene_id=scene_id,
                shots=scene_shots,
                object_info=object_info,
                output_dir=self.output_dir,
            )

            final = self.output_dir / f"{scene_id}_720p.mp4"
            self._ffmpeg_scale(output, final)
            scene_outputs.append(final)

        if len(scene_outputs) == 1:
            return scene_outputs[0]

        concat_file = self.output_dir / "concat.txt"
        concat_file.write_text(
            "\n".join(
                f"file '{path.resolve()}'"
                for path in scene_outputs
            ) + "\n",
            encoding="utf-8",
        )

        final = (
            self.project_root
            / "data"
            / "production"
            / "final_h3_720p.mp4"
        )

        command = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_file),
            "-c",
            "copy",
            str(final),
        ]

        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if completed.returncode != 0:
            raise RuntimeError(
                "FFmpeg scene assembly failed:\n"
                + completed.stderr[-4000:]
            )

        return final
