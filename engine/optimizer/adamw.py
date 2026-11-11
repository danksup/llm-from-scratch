import engine.backend as nx
from typing import Any, Callable
import copy
import ast
from .scheduler import LinearSchedule, CosineDecay

SCHEDULER = {
    "cosine_decay": CosineDecay,
    "linear_schedule": LinearSchedule
}

class AdamW:
    def __init__(self, lr=1e-3, beta1:float=0.9, beta2:float=0.999, epsilon:float=1e-8, weight_decay:float=0.01, use_master:bool=True, scheduler:None|str=None, min_lr:None | float= None, t_warmup:None|int=None, *, _all_not_decayed:bool=False, eight_bit=False, foreach=False) -> None:
        assert lr >= 0, "lr must be non-negative"
        assert beta1 >= 0 and beta1 < 1, "allowed beta1 range: [0,1)"
        assert beta2 >= 0 and beta2 < 1, "allowed beta2 range: [0,1)"
        assert weight_decay >= 0, "weight_decay must be non-negative"

        self.state = {}
        self.state["t"] = nx.array(1, dtype=nx.int32)
        self.init_lr = nx.float_32(lr)
        self.lr = nx.float_32(lr)
        self.min_lr = min_lr
        self.scheduler = scheduler
        self.t_warmup = t_warmup
        self.eight_bit = eight_bit
        self.foreach= foreach

        if scheduler:
            self.schedule = SCHEDULER[scheduler](self.init_lr, min_lr)
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

    def step_many(self, name_param_gradient_decay:list[Any]) -> dict[Any,Any]:
        group = {}
        m_dtype = nx.float32 if not self.eight_bit else nx.int8
        v_dtype = nx.float32 if not self.eight_bit else nx.uint8
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
                    "m": nx.zeros_like(params, m_dtype),
                    "v": nx.zeros_like(params,  v_dtype),
                    "m_scales": None,
                    "v_scales": None,
                }
            else:
                if self.use_master:
                    params = self.state[group_tuple]["master"]
                else:
                    params = nx.stack([i[1] for i in thing])

            state_shape = self.state[group_tuple]
            thing.clear()

            m_v_t = (state_shape["m"], state_shape["v"], self.state["t"], state_shape["m_scales"], state_shape["v_scales"],)

            weight_decay = self.weight_decay if should_decay else nx.float_32(0.0)

            new_params, m,v, m_scales, v_scales = self.__step(m_v_t,params,gradients,self.lr,  self.epsilon, self.beta1, self.beta2, weight_decay, self.eight_bit)
            del params, gradients

            nx.eval(new_params, m, v, m_scales, v_scales)

            self.state[group_tuple] = {"m":m, "v":v, "m_scales":m_scales, "v_scales":v_scales}

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
        if self.__all_not_decayed:
            decay = False

        m_dtype = nx.float32 if not self.eight_bit else nx.int8
        v_dtype = nx.float32 if not self.eight_bit else nx.uint8
        if name not in self.state:
            self.state[name] = {
                "m": nx.zeros_like(w, m_dtype),
                "v": nx.zeros_like(w, v_dtype),
                "m_scales": None,
                "v_scales": None,
            }
        else:
            if self.use_master:
                w = self.state[name]["master"]

        state_shape = self.state[name]
        m_v_t = (state_shape["m"], state_shape["v"], self.state["t"], state_shape.get("m_scales", None), state_shape.get("v_scales", None),)
        weight_decay = self.weight_decay if decay else nx.float_32(0.0)

        new_param, m,v, m_scales, v_scales = self.__step(m_v_t, w, dw,self.lr,  self.epsilon, self.beta1, self.beta2, weight_decay, self.eight_bit)
        nx.eval(new_param, m, v, m_scales, v_scales)
        self.state[name] = {"m":m, "v":v, "m_scales":m_scales, "v_scales":v_scales}
        if self.use_master:
            self.state[name]["master"] = new_param

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

    def increment_t(self):
        self.state["t"] += 1

    @staticmethod
    @nx.compile
    def __step(m_v_t, params:Any, grads:Any, lr:Any, epsilon:float, beta1:float, beta2:float, weight_decay:float, eight_bit) -> tuple[Any,...]:
        m,v,t,m_scales,v_scales = m_v_t
        if m_scales is not None:
            m = nx.linear_dequantize(m, m_scales)
        else:
            m = m.astype(nx.float32)
        if v_scales is not None:
            v = nx.log_dequantize(v, v_scales, epsilon)
        else:
            v = v.astype(nx.float32)

        m = beta1 * m + (1.0 - beta1) * grads
        v = beta2 * v + (1.0 - beta2) * (grads**2)
        m_hat = m / (1.0 - beta1 ** t)
        v_hat = v / (1.0 - beta2 ** t)
        params = params * (1 - lr * weight_decay)
        params = params - lr * m_hat / (nx.sqrt(v_hat) + epsilon)

        if eight_bit:
            m,m_scales = nx.linear_quantize(m)
            v,v_scales = nx.log_quantize(v, epsilon)
        return params, m, v, m_scales, v_scales

    def to_dict(self) -> dict[Any, Any]:
        adamw = {"state":{}}
        adamw["state"]["optimizer.t"] = nx.copy(self.state["t"])

        state = copy.deepcopy(self.state)

        optimizer_state_name = {}
        for shape, a in state.items():
            if shape != "t":
                for b, c in a.items():
                    if b != "names":
                        if c is not None:
                            adamw["state"][f"optimizer.{str(shape)}.{b}"] = c
                    else:
                        optimizer_state_name[f"optimizer.{str(shape)}.{b}"] = c

        adamw_configs = {
            "type":"adamw",
            "lr": self.lr.item(),
            "init_lr": self.init_lr.item()if hasattr(self.init_lr, "item") else self.init_lr,
            "beta1": self.beta1.item()if hasattr(self.beta1, "item") else self.beta1,
            "beta2": self.beta2.item()if hasattr(self.beta2, "item") else self.beta2,
            "epsilon": self.epsilon.item() if hasattr(self.epsilon, "item") else self.epsilon,
            "weight_decay": self.weight_decay.item()if hasattr(self.weight_decay, "item") else self.weight_decay,
            "use_master":self.use_master,
            "scheduler":self.scheduler,
            "min_lr": self.min_lr.item() if hasattr(self.min_lr, "item") else self.min_lr, #type:ignore
            "t_warmup": self.t_warmup,
            "foreach":self.foreach,
            "eight_bit":self.eight_bit,
            "_all_not_decayed":self.__all_not_decayed,
            "names": optimizer_state_name
        }

        adamw["configs"] = adamw_configs
        return adamw

    @classmethod
    def from_dict(cls,  configs, states) -> "AdamW":
        if configs.get("type", None) is not None:
            configs.pop("type")

        names = None
        if configs.get("names", None) is not None:
            names = configs.pop("names")

        init_lr = 1e-3
        if configs.get("init_lr", None) is not None:
            init_lr = configs.pop("init_lr")

        adamw = cls(**configs)
        adamw.init_lr = nx.float_32(init_lr)

        built = {}
        # built["names"] = names

        for name_shape, tensor in states.items():
            if name_shape.startswith("optimizer"):
                name_shape = name_shape[10:]
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
                name_shape = name_shape[10:]
                a,b = name_shape.rsplit(".", 1)
                formatted = ast.literal_eval(a)
                if formatted not in built:
                    built[formatted] = {b:tensor}
                else:
                    built[formatted][b] = tensor

        adamw.state = built

        return adamw
