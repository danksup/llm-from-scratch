import engine.backend as nx
from typing import Any, Callable

class SGD:
    def __init__(self, lr=1e-4, momentum:float=0.0,weight_decay:float=.0,dampening:float=.0, use_master:bool=True, scheduler:None|Callable=None, min_lr:None | float= None) -> None:
        self.lr = nx.float_32(lr)
        self.init_lr = lr
        self.min_lr = min_lr
        self.t = nx.array(0, dtype=nx.int32)
        self.use_master = use_master
        self.momentum = nx.float_32(momentum)
        self.weight_decay = nx.float_32(weight_decay)
        self.dampening = nx.float_32(dampening)
        if scheduler:
            if  min_lr and min_lr > lr:
                raise ValueError("min lr cant be bigger than init lr")
        if use_master:
            self.masters = {}
        self.scheduler = scheduler

        if momentum > 0.0:
            self.state = {}
    
    def step_many(self, name_param_gradient:list[Any], train_contexts, batch_size, total_epoch):
        if self.scheduler:
            current_step = self.t
            total_step = ((len(train_contexts)) // batch_size) * total_epoch
            progress = min(1, current_step / total_step) 
            self.lr = self.scheduler(self.init_lr, self.min_lr, progress)

        self.t += 1

        group = {}
        for name,param,gradient in name_param_gradient:
            shape = param.shape
            if shape not in group:
                group[shape] = []
            group[shape].append((name,param,gradient))

        optimized = {}
        for shape, thing in group.items():
            names = [i[0] for i in thing]
            params = nx.stack([i[1] for i in thing])
            gradients = nx.stack([i[2] for i in thing])

            if self.use_master:
                if shape not in self.masters:
                    self.masters[shape] = {
                        "names": names.copy() ,
                        "master": nx.copy(params),
                    }
                else:
                    params = self.masters[shape]["master"]
                    assert self.masters[shape]["names"] == names

            if self.momentum > 0.0:
                if shape not in self.state:
                    self.state[shape] = {
                        "v": nx.zeros_like(params, nx.float32)
                    }

            v = 0
            if self.momentum > 0.0:
                v = self.state[shape]["v"]
            new_params, new_v= self.__step(params, gradients, self.lr, self.momentum, v, self.weight_decay, self.dampening)

            if self.momentum > 0:
                self.state[shape]["v"] = new_v

            if self.use_master:
                name_list = []
                for idx, name in enumerate(names):
                    optimized[name] = new_params[idx]
                    name_list.append(name)
                self.masters[shape]["names"] = name_list
            else:
                for idx, name in enumerate(names):
                    optimized[name] = new_params[idx]
                
        return optimized
    
    @staticmethod
    @nx.compile
    def __step(params:Any ,gradient:Any, lr, momentum:Any=0, v:Any=0, weight_decay:Any=0, dampening:Any=0) -> Any:
        v = v * momentum + (1 - dampening) * gradient
        params = params * (1 - lr * weight_decay)
        params = params - lr * v
        return params, v
    
    def to_dict(self) -> dict[str,Any]:
        sgd = {
            "t":self.t.item(),
            "lr": self.init_lr,
            "momentum":self.momentum,
            "min_lr": self.min_lr,
            "weight_decay":self.weight_decay,
            "dampening":self.dampening,
            "scheduler":self.scheduler,
            "use_master":self.use_master
        }
        if self.use_master:
            master_copy = {}
            for key, val in self.masters.items():
                master_copy[key] = {
                    "names": val["names"].copy(),
                    "master":val["master"].tolist()
                }
            sgd["masters"] = master_copy
        if self.momentum > 0.0:
            state_copy = {}
            for shape, val in self.state.items():
                state_copy[shape] = {
                    "v": val["v"].tolist()
                }
            sgd["state"] = state_copy

        return sgd
    
    @classmethod
    def from_dict(cls, thing) -> "SGD":
        sgd = cls(lr=thing["lr"], momentum=thing["momentum"], weight_decay=thing["weight_decay"], dampening=thing["dampening"], use_master=thing["use_master"], scheduler=thing["scheduler"], min_lr=thing["min_lr"])
        sgd.t = nx.array(thing["t"], nx.int32)

        if sgd.use_master:
            masters = {}
            for key, val in thing["masters"].items():
                masters[key] = {
                    "names": val["names"],
                    "master": nx.array(val["master"], nx.float32)
                } 
            sgd.masters = masters
        
        if sgd.momentum > 0:
            state = {}
            for shape, val in thing["state"].items():
                state[shape] = {
                    "v": nx.array(val["v"], nx.float32)
                }
            sgd.state = state

        return sgd