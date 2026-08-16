import os
import threading

from scheduler.shot_queue import (
    ShotQueue,
)


class GPUScheduler:
    """
    Distributes independent video shots across
    available GPUs.

    Example:

    GPU 0 -> Shot 001 -> Shot 003 -> Shot 005

    GPU 1 -> Shot 002 -> Shot 004 -> Shot 006
    """

    def __init__(
        self,
        gpu_ids=None,
    ):

        if gpu_ids is None:

            gpu_ids = [0, 1]

        self.gpu_ids = gpu_ids

    def run(
        self,
        shots: list,
        worker_function,
    ):

        queue = ShotQueue(shots)

        threads = []

        for gpu_id in self.gpu_ids:

            thread = threading.Thread(
                target=self._worker_loop,
                args=(
                    gpu_id,
                    queue,
                    worker_function,
                ),
                daemon=False,
            )

            threads.append(thread)

            thread.start()

        for thread in threads:

            thread.join()

    def _worker_loop(
        self,
        gpu_id,
        queue,
        worker_function,
    ):

        print(
            f"GPU {gpu_id} worker started."
        )

        while not queue.empty():

            shot = queue.get()

            if shot is None:
                break

            try:

                worker_function(
                    gpu_id,
                    shot,
                )

            except Exception as error:

                print(
                    f"GPU {gpu_id} failed "
                    f"for shot "
                    f"{shot.shot_id}: "
                    f"{error}"
                )

            finally:

                queue.task_done()

        print(
            f"GPU {gpu_id} worker finished."
        )
