import engine.backend as nx
from typing import Any

class CosineDecay:
    def __init__(self, start, end) -> None:
        self.start = start
        self.end = end

    def __str__(self) -> str:
        return "CosineDecay"

    def __call__(self, progress) -> Any:
        return self.end + 0.5 * (self.start - self.end) * (1 + nx.cos(nx.pi * progress))

