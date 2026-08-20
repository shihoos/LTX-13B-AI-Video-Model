from __future__ import annotations

import os
from pathlib import Path
import shutil


class ShotExecutor:

    """
    Executes one shot through the complete pipeline:

        Base workflow
            ↓
        raw MP4
            ↓
        IC-LoRA/detail/upscale workflow
            ↓
        detailed high-resolution MP4

    The raw artifact is persisted before the second stage starts.
    Therefore a restart can reuse an existing raw clip instead of
    regenerating it.
    """

    def __init__(
        self,
        comfy_client,
        workflow_adapter,
        detailer_workflow_adapter,
        checkpoint_manager,
        project_root: Path,
        comfy_input_dir: Path,
    ):

        self.client = comfy_client

        self.workflow_adapter = workflow_adapter

        self.detailer_workflow_adapter = (
            detailer_workflow_adapter
        )

        self.checkpoints = checkpoint_manager

        self.project_root = Path(
            project_root
        )

        self.comfy_input_dir = Path(
            comfy_input_dir
        )

        self.comfy_input_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Convert both graph workflows to API prompts once.
        self.api_workflow = (
            self.workflow_adapter
            .to_api_workflow()
        )

        self.api_detailer_workflow = (
            self.detailer_workflow_adapter
            .to_api_workflow()
        )

    # ========================================================
    # REFERENCE IMAGE
    # ========================================================

    def _copy_reference(
        self,
        reference_path: str,
        shot_id: str,
    ) -> str:

        source = Path(
            reference_path
        )

        if not source.exists():

            raise FileNotFoundError(
                f"Reference image does not exist: "
                f"{source}"
            )

        destination = (
            self.comfy_input_dir
            / (
                f"{shot_id}_"
                f"{source.name}"
            )
        )

        shutil.copy2(
            source,
            destination,
        )

        return destination.name

    # ========================================================
    # VIDEO INPUT FOR DETAILER
    # ========================================================

    def _copy_raw_to_comfy_input(
        self,
        raw_path: Path,
        shot_id: str,
    ) -> str:

        if not raw_path.exists():

            raise FileNotFoundError(
                f"Raw video does not exist: "
                f"{raw_path}"
            )

        destination = (
            self.comfy_input_dir
            / (
                f"{shot_id}_raw.mp4"
            )
        )

        # A hard link avoids copying the entire raw video when
        # source and destination are on the same filesystem.
        #
        # If hard linking is unavailable, for example because
        # the paths are on different filesystems, fall back to
        # a normal metadata-preserving copy.
        if destination.exists():

            shutil.copy2(
                raw_path,
                destination,
            )

        else:

            try:

                os.link(
                    raw_path,
                    destination,
                )

            except OSError:

                shutil.copy2(
                    raw_path,
                    destination,
                )

        return destination.name

    # ========================================================
    # RAW GENERATION
    # ========================================================

    def execute_raw(
        self,
        shot,
        gpu_id: int,
        output_dir: Path,
    ) -> Path:

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.checkpoints.mark_generating(
            shot.shot_id,
            gpu_id,
        )

        workflow = (
            self.workflow_adapter
            .apply_shot(
                self.api_workflow,
                shot,
            )
        )

        # ----------------------------------------------------
        # Make output name unique per shot.
        # ----------------------------------------------------

        self.workflow_adapter.set_filename_prefix(
            workflow,
            f"ltx_raw/{shot.shot_id}",
        )

        # ----------------------------------------------------
        # Use the shot seed when supplied.
        # ----------------------------------------------------

        if shot.seed is not None:

            self.workflow_adapter.set_seed(
                workflow,
                int(shot.seed),
            )

        # ----------------------------------------------------
        # Copy and bind first reference image.
        # ----------------------------------------------------

        if shot.reference_images:

            reference_name = (
                self._copy_reference(
                    shot.reference_images[0],
                    shot.shot_id,
                )
            )

            self.workflow_adapter.set_input_image(
                workflow,
                reference_name,
            )

        print(
            f"[{shot.shot_id}] "
            f"GPU {gpu_id}: starting base generation"
        )

        # ----------------------------------------------------
        # Queue base workflow.
        # ----------------------------------------------------

        prompt_id = (
            self.client.queue_prompt(
                workflow
            )
        )

        history = (
            self.client.wait_for_prompt(
                prompt_id
            )
        )

        video_outputs = (
            self.client.find_video_outputs(
                history
            )
        )

        if not video_outputs:

            raise RuntimeError(
                f"[{shot.shot_id}] "
                "Base workflow completed but "
                "no video output was found."
            )

        video = video_outputs[0]

        destination = (
            output_dir
            / "shots"
            / shot.scene_id
            / "raw"
            / (
                f"{shot.shot_id}.mp4"
            )
        )

        self.client.download_file(
            filename=video[
                "filename"
            ],
            subfolder=video[
                "subfolder"
            ],
            file_type=video[
                "type"
            ],
            destination=destination,
        )

        if not destination.exists():

            raise RuntimeError(
                f"[{shot.shot_id}] "
                "Raw output download failed."
            )

        self.checkpoints.mark_raw_complete(
            shot.shot_id,
            str(destination),
        )

        print(
            f"[{shot.shot_id}] "
            f"Raw generation complete:"
            f"\n    {destination}"
        )

        return destination

    # ========================================================
    # IC-LORA + DETAIL + UPSCALE
    # ========================================================

    def execute_detailer(
        self,
        shot,
        raw_path: Path,
        gpu_id: int,
        output_dir: Path,
    ) -> Path:

        if not raw_path.exists():

            raise FileNotFoundError(
                f"Raw clip required for detailer "
                f"does not exist: {raw_path}"
            )

        workflow = (
            self.detailer_workflow_adapter
            .apply_shot(
                self.api_detailer_workflow,
                shot,
            )
        )

        # ----------------------------------------------------
        # Seed detail generation using the same shot seed.
        # ----------------------------------------------------

        if shot.seed is not None:

            self.detailer_workflow_adapter.set_seed(
                workflow,
                int(shot.seed),
            )

        # ----------------------------------------------------
        # Copy raw generated video into ComfyUI input.
        # ----------------------------------------------------

        comfy_video_name = (
            self._copy_raw_to_comfy_input(
                raw_path,
                shot.shot_id,
            )
        )

        # ----------------------------------------------------
        # Bind raw video to VHS_LoadVideo.
        # ----------------------------------------------------

        self.detailer_workflow_adapter.set_input_video(
            workflow,
            comfy_video_name,
        )

        # ----------------------------------------------------
        # Detailer/upscale output.
        # This workflow already performs:
        #
        # raw video
        #    ↓
        # IC-LoRA 0.9.8
        #    ↓
        # spatial upscale 2x
        #    ↓
        # 1536x864 master
        # ----------------------------------------------------

        self.detailer_workflow_adapter.set_filename_prefix(
            workflow,
            f"ltx_master/{shot.shot_id}",
        )

        print(
            f"[{shot.shot_id}] "
            f"GPU {gpu_id}: starting "
            "IC-LoRA detail + spatial upscale"
        )

        prompt_id = (
            self.client.queue_prompt(
                workflow
            )
        )

        history = (
            self.client.wait_for_prompt(
                prompt_id
            )
        )

        video_outputs = (
            self.client.find_video_outputs(
                history
            )
        )

        if not video_outputs:

            raise RuntimeError(
                f"[{shot.shot_id}] "
                "Detailer workflow completed "
                "but no video output was found."
            )

        video = video_outputs[0]

        destination = (
            output_dir
            / "shots"
            / shot.scene_id
            / "upscaled"
            / (
                f"{shot.shot_id}.mp4"
            )
        )

        self.client.download_file(
            filename=video[
                "filename"
            ],
            subfolder=video[
                "subfolder"
            ],
            file_type=video[
                "type"
            ],
            destination=destination,
        )

        if not destination.exists():

            raise RuntimeError(
                f"[{shot.shot_id}] "
                "Detailer/upscale output "
                "download failed."
            )

        self.checkpoints.mark_upscaled_complete(
            shot.shot_id,
            str(destination),
        )

        print(
            f"[{shot.shot_id}] "
            f"IC-LoRA + upscale complete:"
            f"\n    {destination}"
        )

        return destination

    # ========================================================
    # COMPLETE SHOT
    # ========================================================

    def execute_shot(
        self,
        shot,
        gpu_id: int,
        output_dir: Path,
    ):

        try:

            state = (
                self.checkpoints.get_shot(
                    shot.shot_id
                )
            )

            # ------------------------------------------------
            # COMPLETED SHOT
            # ------------------------------------------------

            if (
                state
                and state.get(
                    "upscaled_path"
                )
                and Path(
                    state["upscaled_path"]
                ).exists()
            ):

                print(
                    f"[{shot.shot_id}] "
                    "already has completed "
                    "upscaled output; skipping."
                )

                if state.get(
                    "status"
                ) != "completed":

                    self.checkpoints.mark_complete(
                        shot.shot_id
                    )

                return Path(
                    state["upscaled_path"]
                )

            # ------------------------------------------------
            # RAW ARTIFACT
            # ------------------------------------------------

            raw_path = None

            if (
                state
                and state.get(
                    "raw_path"
                )
                and Path(
                    state["raw_path"]
                ).exists()
            ):

                raw_path = Path(
                    state["raw_path"]
                )

                print(
                    f"[{shot.shot_id}] "
                    "Existing raw artifact found."
                )

                print(
                    f"[{shot.shot_id}] "
                    "Skipping base generation."
                )

            else:

                raw_path = (
                    self.execute_raw(
                        shot,
                        gpu_id,
                        output_dir,
                    )
                )

            # ------------------------------------------------
            # DETAILER / UPSCALE
            # ------------------------------------------------

            state = (
                self.checkpoints.get_shot(
                    shot.shot_id
                )
            )

            if (
                state
                and state.get(
                    "upscaled_path"
                )
                and Path(
                    state["upscaled_path"]
                ).exists()
            ):

                final_path = Path(
                    state["upscaled_path"]
                )

                self.checkpoints.mark_complete(
                    shot.shot_id
                )

                return final_path

            final_path = (
                self.execute_detailer(
                    shot,
                    raw_path,
                    gpu_id,
                    output_dir,
                )
            )

            # ------------------------------------------------
            # Only now is the shot truly complete.
            # ------------------------------------------------

            self.checkpoints.mark_complete(
                shot.shot_id
            )

            print(
                f"[{shot.shot_id}] "
                "✅ COMPLETE"
            )

            return final_path

        except Exception as error:

            self.checkpoints.mark_failed(
                shot.shot_id,
                str(error),
            )

            print(
                f"[{shot.shot_id}] "
                f"❌ FAILED: {error}"
            )

            raise
