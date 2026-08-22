from queue import Queue


class ShotQueue:

    def __init__(
        self,
        shots: list,
    ):

        self.queue = Queue()

        for shot in shots:

            self.queue.put(
                shot
            )

    def get(self):

        try:

            return self.queue.get_nowait()

        except Exception:

            return None

    def task_done(self):

        self.queue.task_done()
