from __future__ import annotations

import threading


class GPUScheduler:

    """
    Dispatch independent shots across available GPU workers.

    Each worker owns one GPU ID and consumes shots from a shared
    queue. Worker failures are collected and returned to the caller.
    """

    def __init__(
        self,
        gpu_ids=None,
    ):

        if gpu_ids is None:

            gpu_ids = [
                0,
                1,
            ]

        self.gpu_ids = [
            int(gpu_id)
            for gpu_id in gpu_ids
        ]

        if not self.gpu_ids:

            raise ValueError(
                "GPUScheduler requires "
                "at least one GPU ID."
            )

    def run(
        self,
        shots: list,
        worker_function,
    ) -> list[tuple[int, str, str]]:

        if not shots:

            return []

        if not callable(
            worker_function
        ):

            raise TypeError(
                "worker_function must be callable."
            )

        queue = list(
            shots
        )

        queue_lock = (
            threading.Lock()
        )

        failure_lock = (
            threading.Lock()
        )

        failures = []

        def worker(
            gpu_id: int,
        ):

            while True:

                with queue_lock:

                    if not queue:
                        return

                    shot = queue.pop(
                        0
                    )

                shot_id = getattr(
                    shot,
                    "shot_id",
                    "<unknown>",
                )

                try:

                    print(
                        f"[GPU {gpu_id}] "
                        f"Starting {shot_id}"
                    )

                    worker_function(
                        gpu_id,
                        shot,
                    )

                    print(
                        f"[GPU {gpu_id}] "
                        f"Completed {shot_id}"
                    )

                except Exception as error:

                    message = (
                        f"{type(error).__name__}: "
                        f"{error}"
                    )

                    print(
                        f"[GPU {gpu_id}] "
                        f"FAILED {shot_id}: "
                        f"{message}"
                    )

                    with failure_lock:

                        failures.append(
                            (
                                gpu_id,
                                str(
                                    shot_id
                                ),
                                message,
                            )
                        )

        threads = []

        for gpu_id in self.gpu_ids:

            thread = (
                threading.Thread(
                    target=worker,
                    args=(
                        gpu_id,
                    ),
                    daemon=False,
                    name=(
                        f"ltx-gpu-{gpu_id}"
                    ),
                )
            )

            threads.append(
                thread
            )

            thread.start()

        for thread in threads:

            thread.join()

        return failures
