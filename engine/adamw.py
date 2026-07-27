import engine.backend as nx
from typing import Any

class AdamW:
    def __init__(self,min_lr=1e-4, max_lr=1e-3, beta:float=0.9, beta2:float=0.999, epsilon:float=1e-8, weight_decay=0.01) -> None:
        """
        m: first moment 
            m = β_{1} * m + (1 - β_{1}) * gradient    \n
        v: second moment 
            v = β_{2} * v + (1 - β_{2}) * gradient^2  \n
        bias correction:
            m̂ = m / (1 - β_{1}ᵗ)   \n
            v̂ = m / (1 - β_{2}ᵗ)   \n
        parameter update:
            w = w - lr * m̂ / (sqrt(v̂) + ε)
        """
        # self.memory = {}
        self.state = {}
        self.state["t"] = nx.array(0, dtype=nx.int32)
        self.lr = nx.float_32(min_lr)
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.beta1 = nx.float_32(beta)
        self.beta2 = nx.float_32(beta2)
        self.epsilon = nx.float_32(epsilon)
        self.weight_decay = nx.float_32(weight_decay)
    
    def step_many(self, name_param_gradient:list[Any], train_contexts, batch_size, total_epoch) -> dict[Any,Any]:
        #cosine decay: lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(pi * current_step / total_steps))
        current_step = self.state["t"]
        total_step = ((len(train_contexts)) // batch_size) * total_epoch
        progress = min(1, current_step / total_step) 
        self.lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + nx.cos(nx.pi * progress))

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
                    "master": nx.copy(params),
                    "m": nx.zeros_like(params, nx.float32),
                    "v": nx.zeros_like(params,  nx.float32),
                }
            else:
                params = self.state[shape]["master"]
                assert self.state[shape]["names"] == names
            state_shape = self.state[shape]    
            m_v_t = (state_shape["m"], state_shape["v"], self.state["t"])
            new_params, m,v,_ = self.__step(m_v_t,params,gradients,self.lr,  self.epsilon, self.beta1, self.beta2, self.weight_decay)
            self.state[shape] = {"master":new_params,"m":m, "v":v}
            name_list = []
            for idx, name in enumerate(names):
                optimized[name] = new_params[idx]
                name_list.append(name)
            self.state[shape]["names"] = name_list
        return optimized
    
    @nx.compile
    @staticmethod
    def __step(m_v_t, params:Any, grads:Any, lr:Any, epsilon:float, beta1:float, beta2:float, weight_decay:float) -> tuple[Any,...]: 
        m,v,t = m_v_t       
        norm = nx.sqrt(nx.sum(grads**2, axis=tuple(range(1, grads.ndim)), keepdims=True, dtype=nx.float32), dtype=nx.float32)
        grads = nx.where(norm > 1.0, grads * (1.0 / (norm + epsilon)), grads)
        m = beta1 * m + (1.0 - beta1) * grads
        v = beta2 * v + (1.0 - beta2) * (grads**2)
        m_hat = m / (1.0 - beta1 ** t)
        v_hat = v / (1.0 - beta2 ** t)
        params = params - lr * weight_decay * params
        params = params - lr * m_hat / (nx.sqrt(v_hat) + epsilon)
        return params, m, v, t
    
    def to_dict(self) -> dict[Any, Any]:
        state_copy = {}
        state_copy["t"] = self.state["t"].item()
        for key, value in self.state.items():
            shape_copy = {}
            if key != "t":
                shape_copy["names"] = value["names"]
                shape_copy["master"] = value["master"].tolist()
                shape_copy["m"] = value["m"].tolist()
                shape_copy["v"] = value["v"].tolist()
                state_copy[key] = shape_copy
        return state_copy

    @classmethod
    def from_dict(cls, thing:dict[Any, Any]) -> "AdamW":
        adam = cls()
        adam.state["t"] = nx.array(thing["t"], dtype=nx.int32)
        for key, value in thing.items():
            shape_copy = {}
            if key != "t":
                shape_copy["names"] = value["names"]
                shape_copy["master"] = nx.array(value["master"], dtype=nx.float32)
                shape_copy["m"] = nx.array(value["m"], dtype=nx.float32)
                shape_copy["v"] = nx.array(value["v"], dtype=nx.float32)
                adam.state[key] = shape_copy
        return adam

