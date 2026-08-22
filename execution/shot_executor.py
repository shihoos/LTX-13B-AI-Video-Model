from __future__ import annotations

import json
import shutil
from pathlib import Path


class ShotExecutor:

    WORKFLOW_PATH = (
        Path(__file__)
        .resolve()
        .parents[1]
        / "workflows"
        / "MiniMax-H3"
        / "H3_Ref2VA_Memory_API.json"
    )

    def __init__(
        self,
        comfy_client,
        project_root: Path,
        comfy_input_dir: Path,
    ):
        self.client = comfy_client

        self.project_root = Path(
            project_root
        )

        self.comfy_input_dir = (
            Path(comfy_input_dir)
        )

        self.comfy_input_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def _copy_input(
        self,
        source: str,
        prefix: str,
    ) -> str:

        source_path = Path(
            source
        )

        if not source_path.is_file():
            raise FileNotFoundError(
                source_path
            )

        safe_name = "".join(
            character
            if (
                character.isalnum()
                or character in "._-"
            )
            else "_"
            for character
            in source_path.name
        )

        destination = (
            self.comfy_input_dir
            / f"{prefix}_{safe_name}"
        )

        shutil.copy2(
            source_path,
            destination,
        )

        return destination.name

    def _load_workflow(self) -> dict:
        if not self.WORKFLOW_PATH.is_file():
            raise FileNotFoundError(
                "Required H3 workflow is missing:\n"
                f"{self.WORKFLOW_PATH}"
            )

        return json.loads(
            self.WORKFLOW_PATH.read_text(
                encoding="utf-8"
            )
        )

    @staticmethod
    def _make_prompt(
        shot: dict,
    ) -> str:

        sections = [
            "CHARACTER / REFERENCE CONDITIONING:",
            *shot.get(
                "reference_bindings",
                [],
            ),
            *shot.get(
                "identity_locks",
                [],
            ),
            "",
            "SHOT:",
            shot.get(
                "visual_prompt",
                "",
            ),
            "",
            "ACTION:",
            shot.get(
                "action",
                "",
            ),
            "",
            "CAMERA:",
            shot.get(
                "camera_shot",
                "",
            ),
            shot.get(
                "camera_movement",
                "",
            ),
            "",
            "LIGHTING:",
            shot.get(
                "lighting",
                "",
            ),
            "",
            "MOOD:",
            shot.get(
                "mood",
                "",
            ),
            "",
            "CONTINUITY:",
            shot.get(
                "continuity_notes",
                "",
            ),
        ]

        speech = str(
            shot.get(
                "speech_text",
                "",
            )
            or ""
        ).strip()

        if speech:
            sections.extend(
                [
                    "",
                    "DIALOGUE:",
                    speech,
                ]
            )

        return "\n".join(
            str(value)
            for value in sections
            if str(value).strip()
        )

    def build_memory_workflow(
        self,
        shot: dict,
    ) -> dict:

        workflow = (
            self._load_workflow()
        )

        # These node ids are present in the actual workflow
        # stored in the current repository.
        sampler = workflow.get(
            "70"
        )

        if not isinstance(
            sampler,
            dict,
        ):
            raise RuntimeError(
                "Current H3 workflow does not "
                "contain node 70."
            )

        inputs = sampler.setdefault(
            "inputs",
            {},
        )

        inputs["script"] = (
            self._make_prompt(
                shot
            )
        )

        inputs["width"] = int(
            shot.get(
                "width",
                960,
            )
        )

        inputs["height"] = int(
            shot.get(
                "height",
                544,
            )
        )

        inputs[
            "frames_per_shot"
        ] = int(
            shot.get(
                "frames_per_shot",
                124,
            )
        )

        inputs["steps"] = int(
            shot.get(
                "steps",
                14,
            )
        )

        return workflow

    def execute(
        self,
        shot: dict,
        output_dir: Path,
    ) -> Path:

        workflow = (
            self.build_memory_workflow(
                shot
            )
        )

        prompt_id = (
            self.client
            .queue_prompt(
                workflow
            )
        )

        history = (
            self.client
            .wait_for_prompt(
                prompt_id,
                timeout=7200,
            )
        )

        videos = (
            self.client
            .find_video_outputs(
                history
            )
        )

        if not videos:
            raise RuntimeError(
                f"H3 returned no video for "
                f"{shot['shot_id']}."
            )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        destination = (
            output_dir
            / f"{shot['shot_id']}.mp4"
        )

        result = videos[-1]

        self.client.download_file(
            filename=result["filename"],
            subfolder=result["subfolder"],
            file_type=result["type"],
            destination=destination,
        )

        return destination
