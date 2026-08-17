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
    Production orchestrator.

    Pipeline:

        Shot
          ↓
        Base ComfyUI workflow
          ↓
        raw clip
          ↓
        IC-LoRA/detail/upscale ComfyUI workflow
          ↓
        high-resolution master clip
          ↓
        FFmpeg assembly

    Two GPU workers can process independent shots concurrently.
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

        self.production_dir = (
            DATA_DIR
            / "production"
        )

        # ----------------------------------------------------
        # Detailer workflow path
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Validate workflows immediately.
        # Fail before GPU work starts.
        # ----------------------------------------------------

        if not self.workflow_path.is_file():

            raise FileNotFoundError(
                f"Base workflow not found:\n"
                f"{self.workflow_path}"
            )

        if not self.detailer_workflow_path.is_file():

            raise FileNotFoundError(
                f"IC-LoRA detailer workflow not found:\n"
                f"{self.detailer_workflow_path}"
            )

        # ----------------------------------------------------
        # Persistent state
        # ----------------------------------------------------

        self.checkpoints = (
            CheckpointManager(
                self.production_dir
            )
        )

        # ----------------------------------------------------
        # Separate adapters for separate workflows.
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # One ComfyUI client per GPU.
        # ----------------------------------------------------

        self.clients = {
            gpu_id: ComfyClient(
                base_url=url
            )
            for gpu_id, url in gpu_urls.items()
        }

        self.scheduler = (
            GPUScheduler(
                gpu_ids=list(
                    gpu_urls.keys()
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
    # PREPARE
    # ========================================================

    def prepare(
        self,
        production_plan: dict,
    ):

        for shot in (
            production_plan[
                "shots"
            ]
        ):

            self.checkpoints.initialize_shot(
                shot_id=shot[
                    "shot_id"
                ],
                scene_id=shot[
                    "scene_id"
                ],
            )

    # ========================================================
    # RUN COMPLETE PRODUCTION
    # ========================================================

    def run(
        self,
        production_plan: dict,
    ):

        self.prepare(
            production_plan
        )

        pending_shots = []

        for shot_data in (
            production_plan[
                "shots"
            ]
        ):

            shot_id = (
                shot_data[
                    "shot_id"
                ]
            )

            state = (
                self.checkpoints.get_shot(
                    shot_id
                )
            )

            # ------------------------------------------------
            # Truly complete only when final upscaled
            # artifact exists.
            # ------------------------------------------------

            if (
                state
                and state.get(
                    "status"
                ) == "completed"
                and state.get(
                    "upscaled_path"
                )
                and Path(
                    state["upscaled_path"]
                ).exists()
            ):

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

        # ----------------------------------------------------
        # Two GPU workers execute independent shots in parallel.
        # ----------------------------------------------------

        if pending_shots:

            self.scheduler.run(
                pending_shots,
                self._run_one_shot,
            )

        # ----------------------------------------------------
        # Verify EVERY shot has final output.
        # ----------------------------------------------------

        completed = []

        for shot_data in (
            production_plan[
                "shots"
            ]
        ):

            shot_id = (
                shot_data[
                    "shot_id"
                ]
            )

            state = (
                self.checkpoints.get_shot(
                    shot_id
                )
            )

            if not state:
                continue

            final_path = (
                state.get(
                    "upscaled_path"
                )
            )

            if (
                state.get(
                    "status"
                ) == "completed"
                and final_path
                and Path(
                    final_path
                ).exists()
            ):

                completed.append(
                    final_path
                )

        if len(completed) != len(
            production_plan[
                "shots"
            ]
        ):

            raise RuntimeError(
                "Not all shots are complete. "
                "FFmpeg assembly was not started."
            )

        # ----------------------------------------------------
        # Avoid re-running FFmpeg if the final movie exists.
        # ----------------------------------------------------

        existing = (
            self.checkpoints.state[
                "assembly"
            ]
        )

        if (
            existing.get(
                "status"
            ) == "completed"
            and existing.get(
                "path"
            )
            and Path(
                existing["path"]
            ).exists()
        ):

            print(
                "Final video already exists. "
                "Skipping assembly."
            )

            return Path(
                existing["path"]
            )

        # ----------------------------------------------------
        # Assemble final movie.
        # ----------------------------------------------------

        self.checkpoints.set_assembly_started()

        try:

            final_path = (
                self.assembly
                .assemble(
                    completed
                )
            )

            self.checkpoints.set_assembly_complete(
                str(
                    final_path
                )
            )

            return final_path

        except Exception as error:

            self.checkpoints.set_assembly_failed(
                str(error)
            )

            raise

    # ========================================================
    # ONE SHOT ON ONE GPU
    # ========================================================

    def _run_one_shot(
        self,
        gpu_id: int,
        shot,
    ):

        client = (
            self.clients[gpu_id]
        )

        if not client.health_check():

            raise RuntimeError(
                f"ComfyUI worker for GPU "
                f"{gpu_id} is unavailable."
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
            output_dir=(
                self.output_dir
            ),
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

        return Shot(
            shot_id=data[
                "shot_id"
            ],
            scene_id=data[
                "scene_id"
            ],
            order=data[
                "order"
            ],
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
