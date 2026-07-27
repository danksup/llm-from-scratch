import engine.backend as nx
from typing import Any

class SGD:
    def __init__(self, min_lr=1e-4, max_lr=1e-2, use_master:bool=True) -> None:
        self.lr = nx.float_32(min_lr)
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.t = nx.array(0, dtype=nx.int32)
        self.use_master = use_master
        if use_master:
            self.masters = {}
    
    def step_many(self, name_param_gradient:list[Any], train_contexts, batch_size, total_epoch):
        total_step = ((len(train_contexts)) // batch_size) * total_epoch
        progress = min(1, self.t / total_step) 
        self.lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + nx.cos(nx.pi * progress))
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

            new_params = self.__step(params, gradients, self.lr)

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
    def __step(params:Any ,gradient:Any, lr) -> nx.ArrayLike:
        params = params - lr * gradient
        return params
    
    def to_dict(self) -> dict[str,Any]:
        sgd = {
            "t":self.t.item(),
            "min_lr": self.min_lr,
            "max_lr": self.max_lr,
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

        return sgd
    
    @classmethod
    def from_dict(cls, thing) -> "SGD":
        sgd = cls(thing["min_lr"], thing["max_lr"], thing["use_master"])
        sgd.t = nx.array(thing["t"], nx.int32)

        if sgd.use_master:
            masters = {}
            for key, val in thing["masters"].items():
                masters[key] = {
                    "names": val["names"],
                    "master": nx.array(val["master"], nx.float32)
                } 
            sgd.masters = masters

        return sgd