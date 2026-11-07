import engine.backend as nx
from typing import Any, Literal, Callable
from engine.optimizer.adamw import AdamW

class Adam:
    def __init__(self, lr=1e-3, beta1:float=0.9, beta2:float=0.999, epsilon:float=1e-8, use_master:bool=True,scheduler:None | Callable = None, min_lr:None | float= None, t_warmup:int|None=None, eight_bit=False, foreach=False) -> None:
        self.__adamw = AdamW(lr=lr, beta1=beta1, beta2=beta2, epsilon=epsilon, weight_decay=0.0, use_master=use_master, min_lr=min_lr, t_warmup=t_warmup, scheduler=scheduler, _all_not_decayed=True, eight_bit=eight_bit, foreach=foreach)
        self.state = self.__adamw.state
        self.lr = self.__adamw.lr
        self.init_lr = self.__adamw.init_lr
        self.epsilon = self.__adamw.epsilon
        self.use_master = self.__adamw.use_master
        self.scheduler = self.__adamw.scheduler
        self.min_lr = self.__adamw.min_lr
        self.t_warmup = self.__adamw.t_warmup
        self.eight_bit = self.__adamw.eight_bit
        self.foreach= foreach

    def step_many(self, name_param_gradient:list[Any]) -> dict[Any,Any]:
        optimized = self.__adamw.step_many(name_param_gradient) #type:ignore
        self.lr = self.__adamw.lr
        self.state = self.__adamw.state
        return optimized

    def step(self, tup):
        return self.__adamw.step(tup)

    def calculate_lr(self, max_step, total_epoch):
        self.__adamw.calculate_lr(max_step, total_epoch)
        self.lr = self.__adamw.lr

    def increment_t(self):
        self.__adamw.increment_t()
    
    # def to_dict(self, config_only:bool=True) -> dict[str, Any]:
    #     return self.__adamw.to_dict(config_only=config_only)
    
    # @classmethod
    # def from_dict(cls, thing) -> "Adam":
    #     adamw = AdamW.from_dict(thing)
    #     lr = adamw.lr
    #     beta1 = adamw.beta1
    #     beta2 = adamw.beta2
    #     epsilon = adamw.epsilon
    #     use_master = adamw.use_master
    #     scheduler = adamw.scheduler
    #     min_lr = adamw.min_lr
    #     t_warmup = adamw.t_warmup
    #     adam = cls(lr=lr, beta1=beta1, beta2=beta2, epsilon=epsilon, use_master=use_master, scheduler=scheduler, min_lr=min_lr, t_warmup=t_warmup)

    #     adam.__adamw = adamw

    #     if not thing["config_only"]:
    #         adam.state = adamw.state
    #     else:
    #         adam.state = {}
    #     return adam
