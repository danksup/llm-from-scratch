import engine.backend as nx
import engine.scheduler as scheduler
from typing import Any, Callable

class AdamW:
    def __init__(self, lr=1e-3, beta1:float=0.9, beta2:float=0.999, epsilon:float=1e-8, weight_decay:float=0.01, use_master:bool=True, scheduler:None|Callable=None, min_lr:None | float= None) -> None:
        assert lr > 0, "lr must be non-negative"
        assert beta1 >= 0 and beta1 < 1, "allowed beta1 range: [0,1)"
        assert beta2 >= 0 and beta2 < 1, "allowed beta2 range: [0,1)"
        assert weight_decay >= 0, "weight_decay must be non-negative"

        self.state = {}
        self.state["t"] = nx.array(0, dtype=nx.int32)
        self.init_lr = nx.float_32(lr)
        self.lr = nx.float_32(lr)
        self.min_lr = min_lr
        if scheduler:
            if  min_lr and min_lr > lr:
                raise ValueError("min lr cant be bigger than init lr")
        self.beta1 = nx.float_32(beta1)
        self.beta2 = nx.float_32(beta2)
       
        self.epsilon = nx.float_32(epsilon)
        self.weight_decay = nx.float_32(weight_decay)
        self.use_master = use_master
        self.schduler = scheduler
    
    def step_many(self, name_param_gradient:list[Any], train_contexts, batch_size, total_epoch) -> dict[Any,Any]:

        if self.schduler:
            current_step = self.state["t"]
            total_step = ((len(train_contexts)) // batch_size) * total_epoch
            progress = min(1, current_step / total_step) 
            self.lr = self.schduler(self.init_lr, self.min_lr, progress)

        self.state["t"] += 1

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
            if shape not in self.state:
                self.state[shape] = {
                    "names": names.copy() ,
                    "m": nx.zeros_like(params, nx.float32),
                    "v": nx.zeros_like(params,  nx.float32),
                }
                if self.use_master:
                    self.state[shape]["master"] = nx.copy(params),
            else:
                if self.use_master:
                    params = self.state[shape]["master"]
                    assert self.state[shape]["names"] == names
            state_shape = self.state[shape]    
            m_v_t = (state_shape["m"], state_shape["v"], self.state["t"])
            new_params, m,v,_ = self.__step(m_v_t,params,gradients,self.lr,  self.epsilon, self.beta1, self.beta2, self.weight_decay)
            self.state[shape] = {"m":m, "v":v}
            if self.use_master:
                self.state[shape]["master"] = new_params

            name_list = []
            for idx, name in enumerate(names):
                optimized[name] = new_params[idx]
                name_list.append(name)
            self.state[shape]["names"] = name_list
        return optimized
    
    @staticmethod
    @nx.compile
    def __step(m_v_t, params:Any, grads:Any, lr:Any, epsilon:float, beta1:float, beta2:float, weight_decay:float) -> tuple[Any,...]: 
        m,v,t = m_v_t       
        norm = nx.sqrt(nx.sum(grads**2, axis=tuple(range(1, grads.ndim)), keepdims=True, dtype=nx.float32), dtype=nx.float32)
        grads = nx.where(norm > 1.0, grads * (1.0 / (norm + epsilon)), grads)
        m = beta1 * m + (1.0 - beta1) * grads
        v = beta2 * v + (1.0 - beta2) * (grads**2)
        m_hat = m / (1.0 - beta1 ** t)
        v_hat = v / (1.0 - beta2 ** t)
        params = params * (1 - lr * weight_decay)
        params = params - lr * m_hat / (nx.sqrt(v_hat) + epsilon)
        return params, m, v, t
    
    def to_dict(self) -> dict[Any, Any]:
        adamw = {}
        adamw["t"] = self.state["t"].item()
        for key, value in self.state.items():
            shape_copy = {}
            if key != "t":
                shape_copy["names"] = value["names"]
                if self.use_master:
                    shape_copy["master"] = value["master"].tolist()
                shape_copy["m"] = value["m"].tolist()
                shape_copy["v"] = value["v"].tolist()
                adamw[key] = shape_copy

        adamw_configs = {
            "lr": self.init_lr,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "epsilon": self.epsilon,
            "weight_decay":self.weight_decay,
            "use_master":self.use_master,
            "scheduler":self.schduler,
            "min_lr": self.min_lr
        }
        adamw["adamw_configs"] = adamw_configs
        return adamw

    @classmethod
    def from_dict(cls, thing:dict[Any, Any]) -> "AdamW":
        configs = thing["adamw_configs"]
        lr = configs["lr"]
        beta1 = configs["beta1"]
        beta2 = configs["beta2"]
        epsilon = configs["epsilon"]
        weight_decay = configs["weight_decay"]
        use_master = configs["use_master"]
        scheduler = configs["scheduler"]
        min_lr = configs["min_lr"]
        adamw = cls(lr=lr, beta1=beta1, beta2=beta2, epsilon=epsilon, weight_decay=weight_decay, use_master=use_master, scheduler=scheduler, min_lr=min_lr)
        adamw.state["t"] = nx.array(thing["t"], dtype=nx.int32)
        for key, value in thing.items():
            shape_copy = {}
            if key != "t" and key != "adamw_configs":
                shape_copy["names"] = value["names"]
                if adamw.use_master:
                    shape_copy["master"] = nx.array(value["master"], dtype=nx.float32)
                shape_copy["m"] = nx.array(value["m"], dtype=nx.float32)
                shape_copy["v"] = nx.array(value["v"], dtype=nx.float32)
                adamw.state[key] = shape_copy
        return adamw