from __future__ import annotations

import shutil
from pathlib import Path

from execution.h3_workflow_builder import H3WorkflowBuilder


class ShotExecutor:
    """
    H3-only scene/shot executor.

    For production, ProductionRunner groups shots by scene and calls H3's
    multishot-memory sampler once per scene. This class is also usable for a
    single-shot smoke test.
    """

    def __init__(
        self,
        comfy_client,
        project_root: Path,
        comfy_input_dir: Path,
    ):
        self.client = comfy_client
        self.project_root = Path(project_root)
        self.comfy_input_dir = Path(comfy_input_dir)
        self.comfy_input_dir.mkdir(parents=True, exist_ok=True)

    def _copy_input(self, source: str, shot_id: str) -> str:
        src = Path(source)
        if not src.is_file():
            raise FileNotFoundError(src)

        safe = "".join(
            c if c.isalnum() or c in "._-" else "_"
            for c in src.name
        )
        destination = self.comfy_input_dir / f"{shot_id}_{safe}"
        shutil.copy2(src, destination)
        return destination.name

    @staticmethod
    def _combined_prompt(shots: list[dict]) -> str:
        chunks = []

        for index, shot in enumerate(shots):
            if index:
                chunks.append("---")

            chunks.append(
                " ".join(
                    x for x in (
                        shot.get("visual_prompt", ""),
                        shot.get("action", ""),
                        shot.get("camera_shot", ""),
                        shot.get("camera_movement", ""),
                        shot.get("lighting", ""),
                        shot.get("mood", ""),
                        shot.get("continuity_notes", ""),
                    )
                    if str(x).strip()
                ).strip()
            )

            speech = str(shot.get("speech_text") or "").strip()
            if speech:
                chunks.append(
                    f"Dialogue for this shot: {speech}"
                )

        return "\n".join(chunks)

    @staticmethod
    def _unique(values) -> list[str]:
        result = []
        for value in values or []:
            if value and str(value) not in result:
                result.append(str(value))
        return result

    def execute_scene(
        self,
        scene_id: str,
        shots: list[dict],
        object_info: dict,
        output_dir: Path,
    ) -> Path:
        if not shots:
            raise ValueError("No shots supplied for H3 scene execution.")

        images = self._unique(
            path
            for shot in shots
            for path in shot.get("reference_images", [])
        )[:9]

        videos = self._unique(
            path
            for shot in shots
            for path in shot.get("reference_videos", [])
        )

        voice = next(
            (
                shot.get("reference_audio")
                for shot in shots
                if shot.get("reference_audio")
            ),
            None,
        )

        copied_images = [
            self._copy_input(path, scene_id)
            for path in images
        ]

        copied_voice = (
            self._copy_input(voice, scene_id)
            if voice
            else None
        )

        copied_video = (
            self._copy_input(videos[0], scene_id)
            if videos
            else None
        )

        # The current H3 multishot sampler has one reference-video lane and
        # one voice-ref lane. Application-level N references remain stored;
        # this scene executor selects the supported primary lanes.
        script = self._combined_prompt(shots)

        first_shot = shots[0]
        builder = H3WorkflowBuilder(
            self.project_root,
            object_info,
        )

        workflow = builder.build(
            script=script,
            image_files=copied_images,
            voice_audio=copied_voice,
            reference_video=copied_video,
            width=int(first_shot.get("width", 960)),
            height=int(first_shot.get("height", 544)),
            frames_per_shot=int(first_shot.get("frames_per_shot", 124)),
            steps=int(first_shot.get("steps", 14)),
            output_prefix=f"h3/{scene_id}",
        )

        prompt_id = self.client.queue_prompt(workflow)
        history = self.client.wait_for_prompt(prompt_id, timeout=7200)

        outputs = self.client.find_video_outputs(history)
        if not outputs:
            raise RuntimeError(
                f"H3 completed without a video output for scene {scene_id}."
            )

        output_dir.mkdir(parents=True, exist_ok=True)
        destination = output_dir / f"{scene_id}_h3.mp4"

        self.client.download_file(
            filename=outputs[0]["filename"],
            subfolder=outputs[0]["subfolder"],
            file_type=outputs[0]["type"],
            destination=destination,
        )

        return destination
