# Portions of this file (zeropower_via_newtonschulz5) are adapted from Muon:
# Copyright (c) 2024 Keller Jordan
# See the root LICENSE file for details.

from .sgd import SGD
from .adamw import AdamW
from .rmsprop import RMSProp
import math
from typing import Any
import copy
import ast

import engine.backend as nx
OPTIMIZERS = {
    "sgd": SGD,
    "adamw": AdamW,
    "rmsprop": RMSProp
}

from .scheduler import LinearSchedule, CosineDecay, WarmupStableDecay

SCHEDULER = {
    "none": None,
    "cosine_decay": CosineDecay,
    "linear_schedule": LinearSchedule,
    "wsd": WarmupStableDecay
}

class Muon:
    def __init__(self, lr=1e-3, momentum=0.95, nesterov=True, ns_step=5, ns_dtype="float32", weight_decay=0.01, use_master=True, scheduler=None, min_lr:float|None=None, t_warmup:None|int=None, foreach=False, eight_bit=False, fallback_optimizer="adamw", fallback_configs: dict | None = None) -> None:
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
            if min_lr is not None:
                if min_lr > lr:
                    raise ValueError("min lr cant be bigger than init lr")
                self.min_lr = nx.float_32(min_lr)
            else:
                raise ValueError("?")

            self.schedule = SCHEDULER[scheduler](self.init_lr, self.min_lr)


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

    def step_many(self, name_param_gradient_decay:list[Any]) -> dict[Any,Any]:
        group = {}
        no_muon = []
        m_dtype = nx.float32 if not self.eight_bit else nx.int8
        for x in name_param_gradient_decay:
            name,param,gradient,decay_bool, use_muon, scale, bias = x
            if use_muon:
                shape = gradient.shape
                grouping = (shape,decay_bool)
                if grouping not in group:
                    group[grouping] = []

                group[grouping].append((name,param,gradient,decay_bool, use_muon, scale, bias))
            else:
                no_muon.append(x)

        name_param_gradient_decay.clear()

        optimized = {}
        for group_tuple, thing in group.items():
            _, should_decay = group_tuple
            names = [i[0] for i in thing]
            gradients = nx.stack([i[2] for i in thing])

            if group_tuple not in self.state:
                params = []
                for i in thing:
                    param = self.dequantize_incoming_param(i[1], i[5], i[6])
                    params.append(param)
                params = nx.stack(params)
                self.state[group_tuple] = {
                    "names": names.copy() ,
                    "m": nx.zeros_like(params, m_dtype),
                    "m_scales": None,
                }
            else:
                if self.use_master:
                    params = self.state[group_tuple]["master"]
                else:
                    params = []
                    for i in thing:
                        param = self.dequantize_incoming_param(i[1], i[5], i[6])
                        params.append(param)
                    params = nx.stack(params)

            state_shape = self.state[group_tuple]
            thing.clear()

            m = (state_shape["m"], state_shape.get("m_scales", None))

            weight_decay = self.weight_decay if should_decay else nx.float_32(0.0)

            new_params, m, m_scales = self.__step(m, params, gradients, self.lr, self.momentum, weight_decay, self.nesterov, self.ns_step, self.ns_dtype,self.eight_bit)
            del params, gradients

            nx.eval(new_params, m, m_scales)

            self.state[group_tuple] = {"m":m, "m_scales":m_scales}

            if self.use_master:
                self.state[group_tuple]["master"] = new_params

            name_list = []
            for idx, name in enumerate(names):
                optimized[name] = new_params[idx]
                name_list.append(name)
            self.state[group_tuple]["names"] = name_list

            del new_params

        fallback_optimized = self.fallback_optimizer.step_many(no_muon)
        return optimized | fallback_optimized

    def step(self, tup):
        name, w, dw, decay, use_muon, scale, bias = tup
        if use_muon:
            m_dtype = nx.float32 if not self.eight_bit else nx.int8

            if name not in self.state:
                w = self.dequantize_incoming_param(w, scale, bias)

                self.state[name] = {
                    "m": nx.zeros_like(w, m_dtype),
                    "m_scales": None,
                }
            else:
                if self.use_master:
                    w = self.state[name]["master"]
                else:
                    w = self.dequantize_incoming_param(w, scale, bias)



            state_shape = self.state[name]
            m = (state_shape["m"], state_shape.get("m_scales", None))
            weight_decay = self.weight_decay if decay else nx.float_32(0.0)

            new_param, m, m_scales = self.__step(m, w, dw , self.lr, self.momentum, weight_decay, self.nesterov, self.ns_step, self.ns_dtype,self.eight_bit)
            nx.eval(new_param, m, m_scales)
            self.state[name] = {"m":m, "m_scales":m_scales}
            if self.use_master:
                self.state[name]["master"] = new_param
        else:
            new_param = self.fallback_optimizer.step(tup)

        return new_param

    def calculate_lr(self, max_step, total_epoch):
        t_warmup = self.t_warmup
        if self.scheduler:
            current_step = self.state["t"]
            if t_warmup is not None and t_warmup>0 :
                if current_step > t_warmup:
                    total_step = max_step * total_epoch
                    progress = min(1, (current_step - t_warmup) / max(1, total_step - t_warmup)) #type:ignore
                    self.lr = self.schedule(progress)
                else:
                    self.lr = self.init_lr * min(1, current_step / t_warmup)
            else:
                total_step = max_step * total_epoch
                progress = min(1, current_step / total_step) #type:ignore
                self.lr = self.schedule(progress)

        self.fallback_optimizer.calculate_lr(max_step, total_epoch)

    def increment_t(self):
        self.state["t"] += 1
        self.fallback_optimizer.increment_t()

    @staticmethod
    def zeropower_via_newtonschulz5(G, steps, dtype="float32"):
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
        X = X.astype(nx.str_to_dtype[dtype])

        a, b, c = (3.4445, -4.7750,  2.0315)
        for i in range(steps):
            A = X @ X.transpose(*idx)
            B = b*A + c * (A@A)
            X = a * X + B @ X

        if G.shape[-2] > G.shape[-1]:
            X = X.transpose(*idx)

        return X.astype(nx.float32)

    @staticmethod
    @nx.compile
    def __step(m,param, gradient, lr, momentum, weight_decay, nesterov, ns_step, ns_type="float32", eight_bit=False):
        m, m_scales = m
        if m_scales is not None:
            m = nx.linear_dequantize(m, m_scales)
        else:
            m = m.astype(nx.float32)

        m = momentum * m + (1 - momentum) * gradient
        if nesterov:
            gradient = (1 - momentum) * gradient + momentum * m
        else:
            gradient = m

        X = Muon.zeropower_via_newtonschulz5(gradient, ns_step, ns_type)
        X = X * 0.2 * math.sqrt(max(gradient.shape[-1], gradient.shape[-2])) #moonlight

        param = (param * (1 - lr * weight_decay)) - (X * lr)

        if eight_bit:
            m,m_scales = nx.linear_quantize(m)

        return param, m, m_scales

    def to_dict(self) -> dict[Any, Any]:
        muon = {"state":{}}
        muon["state"]["muon.t"] = nx.copy(self.state["t"])

        state = copy.deepcopy(self.state)

        optimizer_state_name = {}
        for shape, a in state.items():
            if shape != "t":
                for b, c in a.items():
                    if b != "names":
                        if c is not None:
                            muon["state"][f"muon.{str(shape)}.{b}"] = c
                    else:
                        optimizer_state_name[f"muon.{str(shape)}.{b}"] = c

        muon_configs = {
            "type":"muon",
            "lr": self.lr.item() if hasattr(self.lr, "item") else self.lr,
            "init_lr": self.init_lr.item()if hasattr(self.init_lr, "item") else self.init_lr,
            "momentum": self.momentum.item()if hasattr(self.momentum, "item") else self.momentum,
            "weight_decay": self.weight_decay.item()if hasattr(self.weight_decay, "item") else self.weight_decay,
            "use_master":self.use_master,
            "scheduler":self.scheduler,
            "min_lr": self.min_lr.item() if hasattr(self.min_lr, "item") else self.min_lr, #type:ignore
            "ns_dtype":self.ns_dtype,
            "ns_step":self.ns_step,
            "nesterov":self.nesterov,
            "t_warmup": self.t_warmup,
            "foreach":self.foreach,
            "eight_bit":self.eight_bit,
            "names": optimizer_state_name,
            "fallback_optimizer": self.fallback_optimizer_name,
        }
        muon["configs"] = muon_configs

        fallback_things = self.fallback_optimizer.to_dict()
        muon["state"] |= fallback_things["state"]
        muon["configs"]["fallback_configs"] = fallback_things["configs"]

        return muon

    @classmethod
    def from_dict(cls,  configs, states) -> "Muon":
        cfg = copy.deepcopy(configs)

        if configs.get("type", None) is not None:
            cfg.pop("type")

        names = None
        if configs.get("names", None) is not None:
            names = cfg.pop("names")

        init_lr = 1e-3
        if configs.get("init_lr", None) is not None:
            init_lr = cfg.pop("init_lr")

        saved_lr = 1e-3
        if configs.get("lr", None) is not None:
            saved_lr = cfg.pop("lr")
            cfg["lr"] = init_lr

        fallback_configs = {}
        if configs.get("fallback_configs", None) is not None:
            fallback_configs = cfg.pop("fallback_configs")

        muon = cls(**cfg)
        muon.lr = nx.float_32(saved_lr)
        muon.fallback_optimizer = muon.fallback_optimizer.from_dict(fallback_configs, states)

        built = {}

        for name_shape, tensor in states.items():
            if name_shape.startswith("muon."):
                name_shape = name_shape[5:]
                if name_shape != "t":
                    a,b = name_shape.rsplit(".", 1)
                    try:
                        formatted = ast.literal_eval(a)
                    except:
                        formatted = a

                    if formatted not in built:
                        built[formatted] = {b:tensor}
                    else:
                        built[formatted][b] = tensor
                else:
                    built["t"] = tensor

        if names is not None:
            for name_shape, tensor in names.items():
                name_shape = name_shape[5:]
                a,b = name_shape.rsplit(".", 1)
                formatted = ast.literal_eval(a)
                if formatted not in built:
                    built[formatted] = {b:tensor}
                else:
                    built[formatted][b] = tensor

        muon.state = built

        return muon

    @staticmethod
    def dequantize_incoming_param(param, scale, bias):
        return nx.dequantize(param, scale, bias, nx.float32).astype(nx.float32)
