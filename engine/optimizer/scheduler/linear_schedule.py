import engine.backend as nx

class LinearSchedule:
    def __init__(self, start, end) -> None:
        self.start = start
        self.end = end

    def __repr__(self) -> str:
        return "LinearSchedule"

    def __call__(self, progress):
        return self.start + (self.end - self.start) * progress