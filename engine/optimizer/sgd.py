import engine.backend as nx
from typing import Any, Callable
import copy

class SGD:
    def __init__(self, lr=1e-4, momentum:float=0.0,weight_decay:float=.0,dampening:float=.0, use_master:bool=True, scheduler:None|Callable=None, min_lr:None | float= None, t_warmup:int|None=None, foreach=False) -> None:
        assert lr > 0, "lr must be non-negative"
        assert momentum >= 0 and momentum < 1, "allowed momentum range: [0, 1)"
        assert weight_decay >= 0, "weight_decay must be non-negative"
        assert dampening >= 0, "dampening must be non-negative"

        self.lr = nx.float_32(lr)
        self.init_lr = nx.float_32(lr)
        self.min_lr = min_lr
        self.use_master = use_master
        self.momentum = nx.float_32(momentum)
        self.weight_decay = nx.float_32(weight_decay)
        self.dampening = nx.float_32(dampening)
        self.scheduler = scheduler
        self.t_warmup = t_warmup
        self.foreach= foreach

        if scheduler:
            self.schedule = scheduler(self.init_lr, min_lr)
            if min_lr is not None:
                if min_lr > lr:
                    raise ValueError("min lr cant be bigger than init lr")
                if isinstance(min_lr, float):
                    self.min_lr = nx.float_32(min_lr)

        self.state = {}
        self.state["t"] = nx.array(1, dtype=nx.int32)

    def step_many(self, name_param_gradient_decay:list[Any]):
        group = {}
        for x in name_param_gradient_decay:
            if len(x) == 3:
                x += True,
            name,param,gradient,decay_bool = x
            shape = param.shape
            grouping = (shape,decay_bool)
            if grouping not in group:
                group[grouping] = []

            group[grouping].append((name,param,gradient,decay_bool))
        name_param_gradient_decay.clear()

        optimized = {}
        for group_tuple, thing in group.items():
            _, should_decay = group_tuple
            names = [i[0] for i in thing]
            gradients = nx.stack([i[2] for i in thing])
            if group_tuple not in self.state:
                params = nx.stack([i[1] for i in thing])
                self.state[group_tuple] = {
                    "names": names.copy() ,
                }
                if self.momentum > 0:
                    self.state[group_tuple]["v"] =  nx.zeros_like(params, nx.float32)
            else:
                if self.use_master:
                    params = self.state[group_tuple]["master"]
                else:
                    params = nx.stack([i[1] for i in thing])

            thing.clear()

            v = 0
            if self.momentum > 0.0:
                v = self.state[group_tuple]["v"]

            weight_decay = self.weight_decay if should_decay else nx.float_32(0.0)
            new_params, new_v= self.__step(params, gradients, self.lr, self.momentum, v, weight_decay, self.dampening)

            if self.momentum > 0:
                self.state[group_tuple]["v"] = new_v
                nx.eval(new_v, new_params)
            else:
                nx.eval(new_params)

            del params, gradients

            if self.use_master:
                self.state[group_tuple]["master"] = new_params

            name_list = []
            for idx, name in enumerate(names):
                optimized[name] = new_params[idx]
                name_list.append(name)
            self.state[group_tuple]["names"] = name_list

            del new_params


        return optimized

    def step(self, tup):
        name, w, dw, decay = tup

        if name not in self.state:
            self.state[name] = {
                "v": nx.zeros_like(w, nx.float32) if self.momentum > 0.0 else 0,
            }
        else:
            if self.use_master:
                w = self.state[name]["master"]

        weight_decay = self.weight_decay if decay else nx.float_32(0.0)

        v = self.state[name]["v"]
        new_param, v = self.__step(w, dw, self.lr, self.momentum, v, weight_decay, self.dampening)
        if self.momentum > 0.0:
            self.state[name]["v"] = v
            nx.eval(v,new_param)
        else:
            nx.eval(new_param)

        if self.use_master:
            self.state[name]["master"] = new_param

        return new_param

    @staticmethod
    @nx.compile
    def __step(params:Any ,gradient:Any, lr, momentum:Any=0, v:Any=0, weight_decay:Any=0, dampening:Any=0) -> Any:
        v = v * momentum + (1 - dampening) * gradient
        params = params * (1 - lr * weight_decay)
        params = params - lr * v
        return params, v

    def calculate_lr(self, max_step, total_epoch):
        if self.scheduler:
            current_step = self.state["t"]
            t_warmup = self.t_warmup
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

    def increment_t(self):
        self.state["t"] += 1

    def to_dict(self) -> dict[str,Any]:
        sgd = {}
        sgd["state"] = copy.deepcopy(self.state)
        sgd["sgd_configs"] = {
            "lr": self.init_lr,
            "momentum":self.momentum.item(),
            "min_lr": self.min_lr,
            "weight_decay":self.weight_decay.item(),
            "dampening":self.dampening.item(),
            "scheduler":self.scheduler,
            "use_master":self.use_master,
            "t_warmup":self.t_warmup,
            "foreach":self.foreach

        }

        return sgd

    @classmethod
    def from_dict(cls, thing:dict[Any, Any]) -> "SGD":
        configs = thing["sgd_configs"]
        states = thing["state"]
        sgd = cls(**configs)
        sgd.state = states

        return sgd
