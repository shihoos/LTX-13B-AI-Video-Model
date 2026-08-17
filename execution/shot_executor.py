import shutil

from pathlib import Path


class ShotExecutor:

    def __init__(
        self,
        comfy_client,
        workflow_adapter,
        checkpoint_manager,
        project_root: Path,
        comfy_input_dir: Path,
    ):

        self.client = (
            comfy_client
        )

        self.adapter = (
            workflow_adapter
        )

        self.checkpoints = (
            checkpoint_manager
        )

        self.project_root = (
            Path(project_root)
        )

        self.comfy_input_dir = (
            Path(comfy_input_dir)
        )

        self.comfy_input_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.api_workflow = (
            self.adapter
            .to_api_workflow()
        )

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

    def execute_raw(
        self,
        shot,
        gpu_id: int,
        output_dir: Path,
    ):

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.checkpoints.mark_generating(
            shot.shot_id,
            gpu_id,
        )

        workflow = (
            self.adapter.apply_shot(
                self.api_workflow,
                shot,
            )
        )

        self.adapter.set_filename_prefix(
            workflow,
            (
                f"ltx/"
                f"{shot.shot_id}"
            ),
        )

        if shot.seed is not None:

            self.adapter.set_seed(
                workflow,
                shot.seed,
            )

        # Copy available references into ComfyUI input.
        #
        # The canonical I2V workflow determines whether
        # the image input node is required.
        for index, reference in enumerate(
            shot.reference_images
        ):

            if index >= 1:

                break

            self._copy_reference(
                reference,
                shot.shot_id,
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
                "ComfyUI completed the prompt "
                "but no video output was found."
            )

        video = video_outputs[0]

        destination = (
            output_dir
            / "raw"
            / shot.scene_id
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

        self.checkpoints.mark_raw_complete(
            shot.shot_id,
            str(destination),
        )

        return destination

    def execute_shot(
        self,
        shot,
        gpu_id: int,
        output_dir: Path,
    ):

        state = (
            self.checkpoints.get_shot(
                shot.shot_id
            )
        )

        if (
            state
            and state[
                "status"
            ]
            == "completed"
        ):

            print(
                f"[{shot.shot_id}] "
                "already completed; skipping."
            )

            return state[
                "upscaled_path"
            ] or state[
                "detailer_path"
            ] or state[
                "raw_path"
            ]

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
                "raw output already exists."
            )

        else:

            raw_path = self.execute_raw(
                shot,
                gpu_id,
                output_dir,
            )

        # Detailer and external upscale are deliberately
        # not guessed here. Their workflows will be added
        # once their concrete ComfyUI graphs exist.
        #
        # For now raw output is considered the resumable
        # production artifact.

        self.checkpoints.mark_complete(
            shot.shot_id
        )

        return raw_path
