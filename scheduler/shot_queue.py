from queue import Queue


class ShotQueue:

    def __init__(
        self,
        shots: list,
    ):

        self.queue = Queue()

        for shot in shots:

            self.queue.put(shot)

    def get(self):

        if self.queue.empty():

            return None

        return self.queue.get()

    def task_done(self):

        self.queue.task_done()

    def empty(self):

        return self.queue.empty()
