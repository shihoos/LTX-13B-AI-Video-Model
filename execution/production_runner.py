from __future__ import annotations

from pathlib import Path

from execution.assembly_manager import (
    AssemblyManager,
)

from execution.checkpoint_manager import (
    CheckpointManager,
)

from execution.comfy_client import (
    ComfyClient,
)

from execution.comfy_workflow_adapter import (
    ComfyWorkflowAdapter,
)

from execution.shot_executor import (
    ShotExecutor,
)

from planner.config import (
    DATA_DIR,
)

from scheduler.gpu_scheduler import (
    GPUScheduler,
)


class ProductionRunner:

    """
    Executes a complete production plan.

    Pipeline:

        Production plan
              ↓
        GPU scheduler
              ↓
        ShotExecutor
              ↓
        BASE LTX generation
              ↓
        raw MP4
              ↓
        IC-LoRA + spatial detailer
              ↓
        final shot MP4
              ↓
        FFmpeg assembly

    The runner is responsible for:

    - validating the production plan
    - resuming valid completed shots
    - rejecting invalid/incomplete checkpoint artifacts
    - scheduling pending shots across GPU workers
    - validating completed shot artifacts
    - assembling the final video
    - validating the final assembled artifact
    """

    def __init__(
        self,
        project_root: Path,
        gpu_urls: dict[int, str],
        workflow_path: Path,
        detailer_workflow_path: Path | None = None,
    ):

        self.project_root = Path(
            project_root
        )

        if not gpu_urls:

            raise ValueError(
                "At least one ComfyUI GPU URL is required."
            )

        self.gpu_urls = {
            int(gpu_id): str(url).rstrip("/")
            for gpu_id, url
            in gpu_urls.items()
        }

        self.production_dir = (
            DATA_DIR
            / "production"
        )

        if detailer_workflow_path is None:

            detailer_workflow_path = (
                self.project_root
                / "workflows"
                / "detailer"
                / "ltxv-13b-098-ic-lora-upscale.json"
            )

        self.workflow_path = Path(
            workflow_path
        )

        self.detailer_workflow_path = Path(
            detailer_workflow_path
        )

        if not self.workflow_path.is_file():

            raise FileNotFoundError(
                "Base workflow not found:\n"
                f"{self.workflow_path}"
            )

        if not self.detailer_workflow_path.is_file():

            raise FileNotFoundError(
                "Detailer workflow not found:\n"
                f"{self.detailer_workflow_path}"
            )

        self.checkpoints = (
            CheckpointManager(
                self.production_dir
            )
        )

        self.workflow_adapter = (
            ComfyWorkflowAdapter(
                self.workflow_path
            )
        )

        self.detailer_workflow_adapter = (
            ComfyWorkflowAdapter(
                self.detailer_workflow_path
            )
        )

        self.clients = {
            gpu_id: ComfyClient(
                base_url=url
            )
            for gpu_id, url
            in self.gpu_urls.items()
        }

        self.scheduler = (
            GPUScheduler(
                gpu_ids=list(
                    self.gpu_urls.keys()
                )
            )
        )

        self.output_dir = (
            self.production_dir
        )

        self.assembly = (
            AssemblyManager(
                self.output_dir
            )
        )

    # ========================================================
    # ARTIFACT VALIDATION
    # ========================================================

    @staticmethod
    def _validate_artifact(
        path: str | Path,
        description: str,
    ) -> Path:

        artifact = Path(
            path
        )

        if not artifact.exists():

            raise RuntimeError(
                f"{description} does not exist:\n"
                f"{artifact}"
            )

        if not artifact.is_file():

            raise RuntimeError(
                f"{description} is not a file:\n"
                f"{artifact}"
            )

        try:

            size = artifact.stat().st_size

        except OSError as error:

            raise RuntimeError(
                f"Could not inspect {description}:\n"
                f"{artifact}\n"
                f"{error}"
            ) from error

        if size <= 0:

            raise RuntimeError(
                f"{description} is empty:\n"
                f"{artifact}"
            )

        return artifact

    # ========================================================
    # PREPARE
    # ========================================================

    def prepare(
        self,
        production_plan: dict,
    ) -> None:

        if not isinstance(
            production_plan,
            dict,
        ):

            raise TypeError(
                "production_plan must be a dict."
            )

        shots = production_plan.get(
            "shots"
        )

        if not isinstance(
            shots,
            list,
        ):

            raise ValueError(
                "production_plan['shots'] must be a list."
            )

        if not shots:

            raise ValueError(
                "Production plan contains no shots."
            )

        for shot in shots:

            if not isinstance(
                shot,
                dict,
            ):

                raise ValueError(
                    "Every production-plan shot must be a dict."
                )

            try:

                shot_id = shot[
                    "shot_id"
                ]

                scene_id = shot[
                    "scene_id"
                ]

            except KeyError as error:

                raise ValueError(
                    "Production-plan shot is missing "
                    f"required field: {error}"
                ) from error

            self.checkpoints.initialize_shot(
                shot_id=shot_id,
                scene_id=scene_id,
            )

    # ========================================================
    # RUN COMPLETE PRODUCTION
    # ========================================================

    def run(
        self,
        production_plan: dict,
    ) -> Path:

        self.prepare(
            production_plan
        )

        shots = production_plan[
            "shots"
        ]

        pending_shots = []

        for shot_data in shots:

            shot_id = shot_data[
                "shot_id"
            ]

            state = (
                self.checkpoints.get_shot(
                    shot_id
                )
            )

            if (
                state
                and state.get(
                    "status"
                ) == "completed"
                and state.get(
                    "upscaled_path"
                )
            ):

                try:

                    self._validate_artifact(
                        state[
                            "upscaled_path"
                        ],
                        f"Completed artifact for {shot_id}",
                    )

                except RuntimeError as error:

                    print(
                        f"{shot_id}: "
                        "checkpoint artifact is invalid; "
                        "shot will be regenerated."
                    )

                    print(
                        f"  Reason: {error}"
                    )

                else:

                    print(
                        f"{shot_id}: "
                        "already complete."
                    )

                    continue

            pending_shots.append(
                self._dict_to_shot(
                    shot_data
                )
            )

        print(
            f"Pending shots: "
            f"{len(pending_shots)}"
        )

        if pending_shots:

            failures = (
                self.scheduler.run(
                    pending_shots,
                    self._run_one_shot,
                )
            )

            if failures:

                details = "\n".join(
                    (
                        f"- GPU {gpu_id} "
                        f"{shot_id}: {error}"
                    )
                    for (
                        gpu_id,
                        shot_id,
                        error,
                    )
                    in failures
                )

                raise RuntimeError(
                    "One or more shots failed:\n"
                    f"{details}"
                )

        completed = []

        for shot_data in shots:

            shot_id = shot_data[
                "shot_id"
            ]

            state = (
                self.checkpoints.get_shot(
                    shot_id
                )
            )

            if not state:

                raise RuntimeError(
                    f"No checkpoint state exists "
                    f"for {shot_id}."
                )

            if state.get(
                "status"
            ) != "completed":

                raise RuntimeError(
                    "Shot did not reach final "
                    f"completed state: {shot_id}"
                )

            final_path = (
                state.get(
                    "upscaled_path"
                )
            )

            if not final_path:

                raise RuntimeError(
                    "Completed shot is missing "
                    f"upscaled_path: {shot_id}"
                )

            validated_path = (
                self._validate_artifact(
                    final_path,
                    f"Completed shot artifact for {shot_id}",
                )
            )

            completed.append(
                str(
                    validated_path
                )
            )

        existing = (
            self.checkpoints
            .get_assembly()
        )

        if (
            existing
            and existing.get(
                "status"
            ) == "completed"
            and existing.get(
                "path"
            )
        ):

            try:

                existing_path = (
                    self._validate_artifact(
                        existing["path"],
                        "Existing final video",
                    )
                )

            except RuntimeError as error:

                print(
                    "Existing assembly checkpoint "
                    "is invalid; rebuilding final video."
                )

                print(
                    f"  Reason: {error}"
                )

            else:

                print(
                    "Final video already exists. "
                    "Skipping assembly."
                )

                return existing_path

        self.checkpoints.set_assembly_started()

        try:

            final_path = (
                self.assembly
                .assemble(
                    completed
                )
            )

            validated_final_path = (
                self._validate_artifact(
                    final_path,
                    "Final assembled video",
                )
            )

            self.checkpoints.set_assembly_complete(
                str(
                    validated_final_path
                )
            )

            print(
                "✅ Production assembly complete:"
            )

            print(
                validated_final_path
            )

            return validated_final_path

        except Exception as error:

            self.checkpoints.set_assembly_failed(
                str(error)
            )

            raise

    # ========================================================
    # ONE SHOT / ONE GPU
    # ========================================================

    def _run_one_shot(
        self,
        gpu_id: int,
        shot,
    ) -> None:

        if gpu_id not in self.clients:

            raise RuntimeError(
                f"Unknown GPU worker ID: {gpu_id}"
            )

        client = self.clients[
            gpu_id
        ]

        if not client.health_check():

            raise RuntimeError(
                f"ComfyUI worker for GPU "
                f"{gpu_id} is unavailable at "
                f"{self.gpu_urls[gpu_id]}"
            )

        executor = ShotExecutor(
            comfy_client=client,
            workflow_adapter=(
                self.workflow_adapter
            ),
            detailer_workflow_adapter=(
                self.detailer_workflow_adapter
            ),
            checkpoint_manager=(
                self.checkpoints
            ),
            project_root=(
                self.project_root
            ),
            comfy_input_dir=(
                self.project_root
                / "ComfyUI"
                / "input"
            ),
        )

        executor.execute_shot(
            shot=shot,
            gpu_id=gpu_id,
            output_dir=self.output_dir,
        )

    # ========================================================
    # DICT → Shot
    # ========================================================

    @staticmethod
    def _dict_to_shot(
        data: dict,
    ):

        from schemas.shot import (
            Shot,
        )

        required = [
            "shot_id",
            "scene_id",
            "order",
            "duration_seconds",
        ]

        missing = [
            key
            for key in required
            if key not in data
        ]

        if missing:

            raise ValueError(
                "Shot is missing required fields: "
                + ", ".join(
                    missing
                )
            )

        return Shot(
            shot_id=data[
                "shot_id"
            ],

            scene_id=data[
                "scene_id"
            ],

            order=int(
                data[
                    "order"
                ]
            ),

            duration_seconds=float(
                data[
                    "duration_seconds"
                ]
            ),

            characters=data.get(
                "characters",
                [],
            ),

            location=data.get(
                "location",
                "",
            ),

            action=data.get(
                "action",
                "",
            ),

            camera_shot=data.get(
                "camera_shot",
                "",
            ),

            camera_movement=data.get(
                "camera_movement",
                "",
            ),

            lighting=data.get(
                "lighting",
                "",
            ),

            mood=data.get(
                "mood",
                "",
            ),

            visual_prompt=data.get(
                "visual_prompt",
                "",
            ),

            negative_prompt=data.get(
                "negative_prompt",
                "",
            ),

            previous_shot=data.get(
                "previous_shot"
            ),

            next_shot=data.get(
                "next_shot"
            ),

            continuity_notes=data.get(
                "continuity_notes",
                "",
            ),

            seed=data.get(
                "seed"
            ),

            reference_images=data.get(
                "reference_images",
                [],
            ),
        )


if __name__ == "__main__":
    pass
