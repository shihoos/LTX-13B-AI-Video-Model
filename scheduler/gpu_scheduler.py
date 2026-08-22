from __future__ import annotations

import threading


class GPUScheduler:

    def __init__(
        self,
        gpu_ids=None,
    ):

        if gpu_ids is None:
            gpu_ids = [0, 1]

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
        jobs,
        worker_function,
    ):

        if not jobs:
            return []

        queue = list(jobs)

        queue_lock = threading.Lock()
        result_lock = threading.Lock()

        results = []
        failures = []

        def worker(gpu_id):

            while True:

                with queue_lock:

                    if not queue:
                        return

                    job = queue.pop(0)

                job_id = (
                    getattr(
                        job,
                        "scene_id",
                        None,
                    )
                    or
                    getattr(
                        job,
                        "shot_id",
                        "<unknown>",
                    )
                )

                try:

                    print(
                        f"[GPU {gpu_id}] "
                        f"START {job_id}"
                    )

                    result = worker_function(
                        gpu_id,
                        job,
                    )

                    with result_lock:
                        results.append(
                            (
                                gpu_id,
                                str(job_id),
                                result,
                            )
                        )

                    print(
                        f"[GPU {gpu_id}] "
                        f"DONE {job_id}"
                    )

                except Exception as error:

                    message = (
                        f"{type(error).__name__}: "
                        f"{error}"
                    )

                    with result_lock:
                        failures.append(
                            (
                                gpu_id,
                                str(job_id),
                                message,
                            )
                        )

                    print(
                        f"[GPU {gpu_id}] "
                        f"FAILED {job_id}: "
                        f"{message}"
                    )

        threads = []

        for gpu_id in self.gpu_ids:

            thread = threading.Thread(
                target=worker,
                args=(gpu_id,),
                name=f"h3-gpu-{gpu_id}",
                daemon=False,
            )

            thread.start()
            threads.append(
                thread
            )

        for thread in threads:
            thread.join()

        if failures:

            details = "\n".join(
                f"GPU {gpu}: {job}: {error}"
                for gpu, job, error
                in failures
            )

            raise RuntimeError(
                "One or more GPU jobs failed:\n"
                + details
            )

        return results
