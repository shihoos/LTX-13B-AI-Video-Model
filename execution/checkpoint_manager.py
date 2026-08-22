from __future__ import annotations

import json
import threading

from datetime import datetime

from pathlib import Path


class CheckpointManager:

    """
    Thread-safe persistent production state.

    Multiple GPU workers may update different shots
    concurrently, so all state mutations and saves are
    protected by one re-entrant lock.
    """

    def __init__(
        self,
        production_dir: Path,
    ):

        self.production_dir = (
            Path(
                production_dir
            )
        )

        self.shots_dir = (
            self.production_dir
            / "shots"
        )

        self.state_path = (
            self.production_dir
            / "production_state.json"
        )

        self._lock = (
            threading.RLock()
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

    def _load_state(
        self,
    ) -> dict:

        if not self.state_path.exists():

            now = (
                datetime.now()
                .isoformat()
            )

            return {
                "created_at": now,
                "updated_at": now,
                "status": "not_started",
                "shots": {},
                "assembly": {
                    "status": "not_started",
                    "path": None,
                    "error": None,
                },
            }

        with self.state_path.open(
            "r",
            encoding="utf-8",
        ) as file:

            state = json.load(
                file
            )

        if not isinstance(
            state,
            dict,
        ):

            raise RuntimeError(
                "production_state.json "
                "is not a valid object."
            )

        state.setdefault(
            "shots",
            {},
        )

        state.setdefault(
            "assembly",
            {
                "status": "not_started",
                "path": None,
                "error": None,
            },
        )

        state.setdefault(
            "status",
            "not_started",
        )

        return state

    def save(self) -> None:

        with self._lock:

            self.state[
                "updated_at"
            ] = (
                datetime.now()
                .isoformat()
            )

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

                file.flush()

            temp_path.replace(
                self.state_path
            )

    def initialize_shot(
        self,
        shot_id: str,
        scene_id: str,
    ) -> None:

        with self._lock:

            if shot_id not in self.state[
                "shots"
            ]:

                self.state[
                    "shots"
                ][
                    shot_id
                ] = {

                    "shot_id":
                        shot_id,

                    "scene_id":
                        scene_id,

                    "status":
                        "pending",

                    "gpu_id":
                        None,

                    "raw_path":
                        None,

                    "detailer_path":
                        None,

                    "upscaled_path":
                        None,

                    "error":
                        None,

                    "updated_at":
                        datetime.now()
                        .isoformat(),
                }

                self.save()

    def get_shot(
        self,
        shot_id: str,
    ) -> dict | None:

        with self._lock:

            shot = (
                self.state[
                    "shots"
                ].get(
                    shot_id
                )
            )

            if shot is None:
                return None

            return dict(
                shot
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
            shot.get(
                "status"
            )
            == "completed"
        )

    def mark_generating(
        self,
        shot_id: str,
        gpu_id: int,
    ) -> None:

        with self._lock:

            shot = self._require_shot(
                shot_id
            )

            shot[
                "status"
            ] = "generating"

            shot[
                "gpu_id"
            ] = gpu_id

            shot[
                "error"
            ] = None

            shot[
                "updated_at"
            ] = (
                datetime.now()
                .isoformat()
            )

            self.save()

    def mark_raw_complete(
        self,
        shot_id: str,
        path: str,
    ) -> None:

        with self._lock:

            shot = self._require_shot(
                shot_id
            )

            shot[
                "status"
            ] = "raw_complete"

            shot[
                "raw_path"
            ] = str(
                path
            )

            shot[
                "updated_at"
            ] = (
                datetime.now()
                .isoformat()
            )

            self.save()

    def mark_detailer_complete(
        self,
        shot_id: str,
        path: str,
    ) -> None:

        with self._lock:

            shot = self._require_shot(
                shot_id
            )

            shot[
                "status"
            ] = "detailer_complete"

            shot[
                "detailer_path"
            ] = str(
                path
            )

            shot[
                "updated_at"
            ] = (
                datetime.now()
                .isoformat()
            )

            self.save()

    def mark_upscaled_complete(
        self,
        shot_id: str,
        path: str,
    ) -> None:

        with self._lock:

            shot = self._require_shot(
                shot_id
            )

            shot[
                "status"
            ] = "upscaled_complete"

            shot[
                "upscaled_path"
            ] = str(
                path
            )

            shot[
                "updated_at"
            ] = (
                datetime.now()
                .isoformat()
            )

            self.save()

    def mark_complete(
        self,
        shot_id: str,
    ) -> None:

        with self._lock:

            shot = self._require_shot(
                shot_id
            )

            shot[
                "status"
            ] = "completed"

            shot[
                "updated_at"
            ] = (
                datetime.now()
                .isoformat()
            )

            self.save()

    def mark_failed(
        self,
        shot_id: str,
        error: str,
    ) -> None:

        with self._lock:

            shot = self._require_shot(
                shot_id
            )

            shot[
                "status"
            ] = "failed"

            shot[
                "error"
            ] = str(
                error
            )

            shot[
                "updated_at"
            ] = (
                datetime.now()
                .isoformat()
            )

            self.save()

    def reset_shot(
        self,
        shot_id: str,
    ) -> None:

        with self._lock:

            shot = self._require_shot(
                shot_id
            )

            shot[
                "status"
            ] = "pending"

            shot[
                "gpu_id"
            ] = None

            shot[
                "error"
            ] = None

            self.save()

    def set_assembly_started(
        self,
    ) -> None:

        with self._lock:

            self.state[
                "assembly"
            ] = {

                "status":
                    "assembling",

                "path":
                    None,

                "error":
                    None,
            }

            self.save()

    def set_assembly_complete(
        self,
        path: str,
    ) -> None:

        with self._lock:

            self.state[
                "assembly"
            ] = {

                "status":
                    "completed",

                "path":
                    str(
                        path
                    ),

                "error":
                    None,
            }

            self.save()

    def set_assembly_failed(
        self,
        error: str,
    ) -> None:

        with self._lock:

            self.state[
                "assembly"
            ] = {

                "status":
                    "failed",

                "path":
                    None,

                "error":
                    str(
                        error
                    ),
            }

            self.save()

    def get_assembly(
        self,
    ) -> dict:

        with self._lock:

            return dict(
                self.state.get(
                    "assembly",
                    {},
                )
            )

    def _require_shot(
        self,
        shot_id: str,
    ) -> dict:

        shot = (
            self.state[
                "shots"
            ].get(
                shot_id
            )
        )

        if shot is None:

            raise KeyError(
                "Unknown shot ID: "
                f"{shot_id}"
            )

        return shot
