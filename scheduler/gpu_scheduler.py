from __future__ import annotations

import threading


class GPUScheduler:

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
            int(gpu)
            for gpu in gpu_ids
        ]

        if not self.gpu_ids:
            raise ValueError(
                "At least one GPU is required."
            )

    def run(
        self,
        jobs: list,
        worker_function,
    ) -> list[tuple[int, str, str]]:

        if not jobs:
            return []

        queue = list(jobs)

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

                    job = queue.pop(0)

                job_id = (
                    getattr(
                        job,
                        "shot_id",
                        None,
                    )
                    or
                    getattr(
                        job,
                        "scene_id",
                        "<unknown>",
                    )
                )

                try:

                    print(
                        f"[H3 GPU {gpu_id}] "
                        f"START {job_id}"
                    )

                    worker_function(
                        gpu_id,
                        job,
                    )

                    print(
                        f"[H3 GPU {gpu_id}] "
                        f"DONE {job_id}"
                    )

                except Exception as error:

                    message = (
                        f"{type(error).__name__}: "
                        f"{error}"
                    )

                    with failure_lock:
                        failures.append(
                            (
                                gpu_id,
                                str(job_id),
                                message,
                            )
                        )

        threads = []

        for gpu_id in self.gpu_ids:

            thread = (
                threading.Thread(
                    target=worker,
                    args=(gpu_id,),
                    daemon=False,
                    name=(
                        f"h3-gpu-{gpu_id}"
                    ),
                )
            )

            thread.start()
            threads.append(
                thread
            )

        for thread in threads:
            thread.join()

        return failures
