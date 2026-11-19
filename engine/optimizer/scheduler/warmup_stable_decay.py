import engine.backend as nx
from . import cosine_decay
from . import linear_schedule
from typing import Literal

SCHEDULER = {
    "cosine_decay": cosine_decay.CosineDecay,
    "linear_schedule":linear_schedule.LinearSchedule
}

class WarmupStableDecay:
    def __init__(self, start, end, stable_ratio=0.8, decay_type:Literal["cosine_decay", "linear_schedule", "1-sqrt"]="1-sqrt") -> None:
        self.start = start
        self.end = end
        self.stable_ratio = stable_ratio
        self.decay_type = decay_type

        if decay_type == "1-sqrt":
            self.a = lambda decay_progress : self.end + (self.start - self.end) * (1.0 - nx.sqrt(decay_progress))
        else:
            self.a = SCHEDULER[decay_type](start=start, end=end)

    def __str__(self) -> str:
        return "WarmupStableDecay"

    @staticmethod
    def str_name():
        return "WarmupStableDecay"

    def __call__(self, progress):
        if self.stable_ratio >= 1.0:
            return self.start

        decay_progress = (progress - self.stable_ratio) / (1.0 - self.stable_ratio)
        decay_progress = nx.clip(decay_progress, 0.0, 1.0)

        return self.a(decay_progress)
