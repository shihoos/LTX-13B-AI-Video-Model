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

    def __init__(
        self,
        project_root: Path,
        gpu_urls: dict[int, str],
        workflow_path: Path,
    ):

        self.project_root = (
            Path(project_root)
        )

        self.production_dir = (
            DATA_DIR
            / "production"
        )

        self.checkpoints = (
            CheckpointManager(
                self.production_dir
            )
        )

        self.workflow_adapter = (
            ComfyWorkflowAdapter(
                workflow_path
            )
        )

        self.clients = {
            gpu_id: ComfyClient(
                base_url=url
            )
            for gpu_id, url
            in gpu_urls.items()
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

            if self.checkpoints.is_complete(
                shot_id
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

        if pending_shots:

            self.scheduler.run(
                pending_shots,
                self._run_one_shot,
            )

        completed = []

        for shot_data in (
            production_plan[
                "shots"
            ]
        ):

            state = (
                self.checkpoints
                .get_shot(
                    shot_data[
                        "shot_id"
                    ]
                )
            )

            if not state:
                continue

            if (
                state["status"]
                == "completed"
            ):

                path = (
                    state.get(
                        "upscaled_path"
                    )
                    or state.get(
                        "detailer_path"
                    )
                    or state.get(
                        "raw_path"
                    )
                )

                if (
                    path
                    and Path(
                        path
                    ).exists()
                ):

                    completed.append(
                        path
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
