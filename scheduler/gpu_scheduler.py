import threading


class GPUScheduler:

    """
    Dispatches independent shots to worker functions.

    The worker function is responsible for targeting the
    correct ComfyUI/GPU worker.
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

        self.gpu_ids = list(
            gpu_ids
        )

    def run(
        self,
        shots: list,
        worker_function,
    ):

        if not shots:

            return

        queue = list(
            shots
        )

        lock = threading.Lock()

        def worker(
            gpu_id: int,
        ):

            while True:

                with lock:

                    if not queue:
                        return

                    shot = queue.pop(
                        0
                    )

                try:

                    worker_function(
                        gpu_id,
                        shot,
                    )

                except Exception as error:

                    print(
                        f"GPU {gpu_id} "
                        f"failed for "
                        f"{shot.shot_id}: "
                        f"{error}"
                    )

        threads = []

        for gpu_id in (
            self.gpu_ids
        ):

            thread = (
                threading.Thread(
                    target=worker,
                    args=(gpu_id,),
                    daemon=False,
                )
            )

            threads.append(
                thread
            )

            thread.start()

        for thread in threads:

            thread.join()
