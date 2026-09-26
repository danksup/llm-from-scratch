import engine.backend as nx
from typing import Any, Callable

class RMSProp:
    def __init__(self, lr=1e-3, momentum:float=0.0, beta1:float=0.9, beta2:float=0.999, epsilon:float=1e-8, weight_decay:float=0.01, use_master:bool=True, scheduler:None|Callable=None, min_lr:None | float= None, *, _all_not_decayed:bool=False) -> None:
        assert lr >= 0, "lr must be non-negative"
        assert beta1 >= 0 and beta1 < 1, "allowed beta1 range: [0,1)"
        assert beta2 >= 0 and beta2 < 1, "allowed beta2 range: [0,1)"
        assert weight_decay >= 0, "weight_decay must be non-negative"

        self.state = {}
        self.state["t"] = nx.array(0, dtype=nx.int32)
        self.init_lr = nx.float_32(lr)
        self.lr = nx.float_32(lr)
        self.min_lr = min_lr
        self.scheduler = scheduler
        self.momentum = nx.float_32(momentum)
        if scheduler:
            self.schedule = scheduler(self.init_lr, min_lr)
            if min_lr is not None:
                if min_lr > lr:
                    raise ValueError("min lr cant be bigger than init lr")
                if isinstance(min_lr, float):
                    self.min_lr = nx.float_32(min_lr)
        self.beta1 = nx.float_32(beta1)
        self.beta2 = nx.float_32(beta2)
       
        self.epsilon = nx.float_32(epsilon)
        self.weight_decay = nx.float_32(weight_decay)
        self.use_master = use_master

        self.__all_not_decayed = _all_not_decayed
    
    def step_many(self, name_param_gradient_decay:list[Any], max_step:int, total_epoch:int) -> dict[Any,Any]:

        if self.scheduler:
            current_step = self.state["t"]
            total_step = max_step * total_epoch
            progress = min(1, current_step / total_step) 
            self.lr = self.schedule(progress)

        
        self.state["t"] = self.state.get("t", nx.array(0, dtype=nx.int32)) + 1

        group = {}
        for x in name_param_gradient_decay:
            if len(x) == 3:
                x += True,
            name,param,gradient,decay_bool = x
            shape = param.shape
            if self.__all_not_decayed:
                decay_bool = False
            grouping = (shape,decay_bool)
            if grouping not in group:
                group[grouping] = []

            group[grouping].append((name,param,gradient,decay_bool))
                
        optimized = {}
        for group_tuple, thing in group.items():
            _, should_decay = group_tuple
            names = [i[0] for i in thing]
            params = nx.stack([i[1] for i in thing])
            gradients = nx.stack([i[2] for i in thing])

            if group_tuple not in self.state:
                self.state[group_tuple] = {
                    "names": names.copy() ,
                    "v": nx.zeros_like(params,  nx.float32),
                }
                if self.momentum > 0:
                    self.state[group_tuple]["m"] =  nx.zeros_like(params, nx.float32),

                if self.use_master:
                    self.state[group_tuple]["master"] = nx.copy(params)
            else:
                if self.use_master:
                    params = self.state[group_tuple]["master"]
                    assert self.state[group_tuple]["names"] == names
            state_shape = self.state[group_tuple]    

            weight_decay = self.weight_decay if should_decay else nx.float_32(0.0)

            if self.momentum > 0:
                m_v = (state_shape["m"], state_shape["v"])
                new_params, m,v,_ = self.__step_with_momentum(m_v,params,gradients,self.lr,  self.epsilon, self.beta1, self.beta2, weight_decay)
            else:
                v = state_shape["v"]
                new_params, m,v,_ = self.__step_zero_momentum(v, params,gradients,self.lr,  self.epsilon, self.beta2, weight_decay)

            del params, gradients

            self.state[group_tuple] = {"v":v}
            if self.momentum > 0:
                self.state[group_tuple]["m"] = m

            if self.use_master:
                self.state[group_tuple]["master"] = new_params

            name_list = []
            for idx, name in enumerate(names):
                optimized[name] = new_params[idx]
                name_list.append(name)
            self.state[group_tuple]["names"] = name_list
            
            del new_params

        return optimized
    
    @staticmethod
    @nx.compile
    def __step_zero_momentum(v, params:Any, grads:Any, lr:Any, epsilon:float, beta2:float, weight_decay:float) -> tuple[Any,...]: 
        norm = nx.sqrt(nx.sum(grads**2, axis=tuple(range(1, grads.ndim)), keepdims=True, dtype=nx.float32), dtype=nx.float32)
        grads = nx.where(norm > 1.0, grads * (1.0 / (norm + epsilon)), grads)

        print(type(v), type(beta2))
        v = beta2 * v + (1.0 - beta2) * (grads**2)
        step = (lr / (nx.sqrt(v) + epsilon)) * grads
        
        params = params - (lr * weight_decay * params)
        params = params - step
        return params, v

    @staticmethod
    @nx.compile
    def __step_with_momentum(m_v, params:Any, grads:Any, lr:Any, epsilon:float, beta1:float, beta2:float, weight_decay:float) -> tuple[Any,...]: 
        m, v = m_v
        norm = nx.sqrt(nx.sum(grads**2, axis=tuple(range(1, grads.ndim)), keepdims=True, dtype=nx.float32), dtype=nx.float32)
        grads = nx.where(norm > 1.0, grads * (1.0 / (norm + epsilon)), grads)
        
        v = beta2 * v + (1.0 - beta2) * (grads**2)
        
        m = beta1 * m + (grads / (nx.sqrt(v) + epsilon))
        step = lr * m
        
        params = params - (lr * weight_decay * params)
        params = params - step
        return params, m, v
    
    def to_dict(self, config_only:bool=True) -> dict[Any, Any]:
        adamw = {}
        
        if not config_only:
            adamw["t"] = self.state["t"].item()
            for key, value in self.state.items():
                shape_copy = {}
                if key != "t":
                    shape_copy["names"] = value["names"]
                    if self.use_master:
                        shape_copy["master"] = value["master"].tolist()

                    if "m" in value:
                        shape_copy["m"] = value["m"].tolist()
                    shape_copy["v"] = value["v"].tolist()
                    adamw[key] = shape_copy

        adamw_configs = {
            "config_only": config_only,
            "lr": self.init_lr.item(),
            "beta1": self.beta1.item(),
            "beta2": self.beta2.item(),
            "epsilon": self.epsilon.item(),
            "weight_decay":self.weight_decay.item(),
            "use_master":self.use_master,
            "scheduler":self.scheduler,
            "min_lr": self.min_lr
        }
        if self.min_lr is not None and hasattr(self.min_lr, "dtype"):
            adamw_configs["min_lr"] = self.min_lr.item() #type:ignore

        adamw["adamw_configs"] = adamw_configs
        return adamw

    @classmethod
    def from_dict(cls, thing:dict[Any, Any]) -> "RMSProp":
        configs = thing["adamw_configs"]
        config_only = configs["config_only"]
        lr = configs["lr"]
        beta1 = configs["beta1"]
        beta2 = configs["beta2"]
        epsilon = configs["epsilon"]
        weight_decay = configs["weight_decay"]
        use_master = configs["use_master"]
        scheduler = configs["scheduler"]
        min_lr = configs["min_lr"]
        adamw = cls(lr=lr, beta1=beta1, beta2=beta2, epsilon=epsilon, weight_decay=weight_decay, use_master=use_master, scheduler=scheduler, min_lr=min_lr)
        if not config_only:
            adamw.state["t"] = nx.array(thing["t"], dtype=nx.int32)
            for key, value in thing.items():
                shape_copy = {}
                if key != "t" and key != "adamw_configs":
                    shape_copy["names"] = value["names"]
                    if adamw.use_master:
                        shape_copy["master"] = nx.array(value["master"], dtype=nx.float32)

                    if "m" in value:
                        shape_copy["m"] = nx.array(value["m"], dtype=nx.float32)
                    shape_copy["v"] = nx.array(value["v"], dtype=nx.float32)
                    adamw.state[key] = shape_copy
        else:
            adamw.state = {}
        return adamw