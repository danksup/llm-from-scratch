import engine.backend as nx

class WarmupStableDecay:
    def __init__(self, start, end) -> None:
        self.start = start
        self.end = end

    def __str__(self) -> str:
        return "WarmupStableDecay"

    @staticmethod
    def str_name():
        return "WarmupStableDecay"

    def __call__(self, progress):
        pass
