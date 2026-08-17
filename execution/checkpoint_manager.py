import json

from datetime import datetime

from pathlib import Path


class CheckpointManager:

    """
    Persistent production state.

    A shot is never considered complete merely because
    a directory exists. Completion is explicitly recorded
    in the manifest.
    """

    def __init__(
        self,
        production_dir: Path,
    ):

        self.production_dir = (
            Path(production_dir)
        )

        self.shots_dir = (
            self.production_dir
            / "shots"
        )

        self.state_path = (
            self.production_dir
            / "production_state.json"
        )

        self.production_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.shots_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.state = (
            self._load_state()
        )

    def _load_state(self) -> dict:

        if not self.state_path.exists():

            return {
                "created_at": (
                    datetime.now()
                    .isoformat()
                ),
                "updated_at": (
                    datetime.now()
                    .isoformat()
                ),
                "status": "not_started",
                "shots": {},
                "assembly": {
                    "status": "not_started",
                    "path": None,
                },
            }

        with self.state_path.open(
            "r",
            encoding="utf-8",
        ) as file:

            return json.load(file)

    def save(self):

        self.state[
            "updated_at"
        ] = datetime.now().isoformat()

        temp_path = (
            self.state_path.with_suffix(
                ".tmp"
            )
        )

        with temp_path.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                self.state,
                file,
                indent=2,
                ensure_ascii=False,
            )

        temp_path.replace(
            self.state_path
        )

    def initialize_shot(
        self,
        shot_id: str,
        scene_id: str,
    ):

        if shot_id not in self.state[
            "shots"
        ]:

            self.state["shots"][
                shot_id
            ] = {
                "shot_id": shot_id,
                "scene_id": scene_id,
                "status": "pending",
                "gpu_id": None,
                "raw_path": None,
                "detailer_path": None,
                "upscaled_path": None,
                "error": None,
                "updated_at": (
                    datetime.now()
                    .isoformat()
                ),
            }

            self.save()

    def get_shot(
        self,
        shot_id: str,
    ) -> dict | None:

        return self.state[
            "shots"
        ].get(
            shot_id
        )

    def is_complete(
        self,
        shot_id: str,
    ) -> bool:

        shot = self.get_shot(
            shot_id
        )

        if shot is None:

            return False

        return (
            shot["status"]
            == "completed"
        )

    def mark_generating(
        self,
        shot_id: str,
        gpu_id: int,
    ):

        shot = self.state[
            "shots"
        ][shot_id]

        shot["status"] = (
            "generating"
        )

        shot["gpu_id"] = gpu_id

        shot["error"] = None

        shot["updated_at"] = (
            datetime.now()
            .isoformat()
        )

        self.save()

    def mark_raw_complete(
        self,
        shot_id: str,
        path: str,
    ):

        shot = self.state[
            "shots"
        ][shot_id]

        shot["status"] = (
            "raw_complete"
        )

        shot["raw_path"] = path

        shot["updated_at"] = (
            datetime.now()
            .isoformat()
        )

        self.save()

    def mark_detailer_complete(
        self,
        shot_id: str,
        path: str,
    ):

        shot = self.state[
            "shots"
        ][shot_id]

        shot["status"] = (
            "detailer_complete"
        )

        shot["detailer_path"] = path

        shot["updated_at"] = (
            datetime.now()
            .isoformat()
        )

        self.save()

    def mark_upscaled_complete(
        self,
        shot_id: str,
        path: str,
    ):

        shot = self.state[
            "shots"
        ][shot_id]

        shot["status"] = (
            "upscaled_complete"
        )

        shot["upscaled_path"] = path

        shot["updated_at"] = (
            datetime.now()
            .isoformat()
        )

        self.save()

    def mark_complete(
        self,
        shot_id: str,
    ):

        shot = self.state[
            "shots"
        ][shot_id]

        shot["status"] = (
            "completed"
        )

        shot["updated_at"] = (
            datetime.now()
            .isoformat()
        )

        self.save()

    def mark_failed(
        self,
        shot_id: str,
        error: str,
    ):

        shot = self.state[
            "shots"
        ][shot_id]

        shot["status"] = "failed"

        shot["error"] = str(
            error
        )

        shot["updated_at"] = (
            datetime.now()
            .isoformat()
        )

        self.save()

    def reset_shot(
        self,
        shot_id: str,
    ):

        shot = self.state[
            "shots"
        ][shot_id]

        shot["status"] = "pending"
        shot["gpu_id"] = None
        shot["error"] = None

        self.save()

    def set_assembly_started(self):

        self.state["assembly"][
            "status"
        ] = "assembling"

        self.save()

    def set_assembly_complete(
        self,
        path: str,
    ):

        self.state["assembly"] = {
            "status": "completed",
            "path": path,
        }

        self.save()

    def set_assembly_failed(
        self,
        error: str,
    ):

        self.state["assembly"] = {
            "status": "failed",
            "path": None,
            "error": str(error),
        }

        self.save()
