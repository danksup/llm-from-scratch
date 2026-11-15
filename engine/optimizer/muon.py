# Portions of this file (zeropower_via_newtonschulz5) are adapted from Muon:
# Copyright (c) 2024 Keller Jordan
# See the root LICENSE file for details.

from .sgd import SGD
from .adamw import AdamW
from .rmsprop import RMSProp

import engine.backend as nx
OPTIMIZERS = {
    "sgd": SGD,
    "adamw": AdamW,
    "rmsprop": RMSProp
}

from .scheduler import LinearSchedule, CosineDecay
SCHEDULER = {
    "cosine_decay": CosineDecay,
    "linear_schedule": LinearSchedule
}

class Muon:
    def __init__(self, lr=1e-3, momentum=0.95, nesterov=True, ns_step=5, ns_dtype=nx.float32, weight_decay=0.01, use_master=True, scheduler=None, min_lr:float|None=None, t_warmup:None|int=None, foreach=False, eight_bit=False, fallback_optimizer="adamw", fallback_configs: dict | None = None) -> None:
        self.lr = nx.float_32(lr)
        self.init_lr = nx.float_32(lr)
        self.momentum = nx.float_32(momentum)
        self.nesterov = nesterov
        self.ns_step = ns_step
        self.ns_dtype = ns_dtype
        self.weight_decay = nx.float_32(weight_decay)
        self.use_master = use_master
        self.min_lr = min_lr
        self.scheduler = scheduler
        if scheduler:
            self.schedule = SCHEDULER[scheduler](self.init_lr, min_lr)
            if min_lr is not None:
                if min_lr > lr:
                    raise ValueError("min lr cant be bigger than init lr")
                if isinstance(min_lr, float):
                    self.min_lr = nx.float_32(min_lr)
        self.foreach = foreach
        self.eight_bit = eight_bit
        self.t_warmup = t_warmup
        self.state = {}
        self.state["t"] = nx.array(1, dtype=nx.int32)

        self.fallback_optimizer_name = fallback_optimizer
        self.fallback_optimizer = OPTIMIZERS[fallback_optimizer](**(self.match_aux_config() | (fallback_configs or {})))

    def match_aux_config(self):
        a =  {
            "scheduler": self.scheduler,
            "lr":self.lr.item(),
            "min_lr": self.min_lr.item() if hasattr(self.min_lr, "item") else self.min_lr, #type:ignore
            "use_master":self.use_master,
            "foreach":self.foreach,
            "t_warmup": self.t_warmup
        }

        if self.fallback_optimizer_name == "adamw":
            a["eight_bit"] = self.eight_bit

        return a

    @staticmethod
    def zeropower_via_newtonschulz5(G, steps, dtype=nx.float32):
        """
        adapted from the Muon optimizer by Keller Jordan et al. (2024)
        https://kellerjordan.github.io/posts/muon/
        https://github.com/KellerJordan/Muon
        """
        assert G.ndim >= 2
        X = G.astype(nx.float32)

        idx = list(range(G.ndim))
        idx[-1], idx[-2] = idx[-2], idx[-1]

        if G.shape[-2] > G.shape[-1]:
            X = X.transpose(*idx)
        X = X / (nx.norm(X, axis=(-2, -1), keepdims=True) + 1e-7)
        X = X.astype(dtype)

        a, b, c = (3.4445, -4.7750,  2.0315)
        for i in range(steps):
            A = X @ X.transpose(*idx)
            B = b*A + c * (A@A)
            X = a * X + B @ X

        if G.shape[-2] > G.shape[-1]:
            X = X.transpose(*idx)

        return X.astype(nx.float32)

    @staticmethod
    def _step():
        pass
