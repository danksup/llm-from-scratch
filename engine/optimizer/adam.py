import engine.backend as nx
from typing import Any, Literal, Callable
from engine.optimizer.adamw import AdamW

class Adam:
    def __init__(self, lr=1e-3, beta1:float=0.9, beta2:float=0.999, epsilon:float=1e-8, use_master:bool=True,scheduler:None | Callable = None, min_lr:None | float= None) -> None:
        self.__adamw = AdamW(lr=lr, beta1=beta1, beta2=beta2, epsilon=epsilon, weight_decay=0.0, use_master=use_master, min_lr=min_lr, scheduler=scheduler)
        self.state = self.__adamw.state
        self.lr = self.__adamw.lr
        self.epsilon = self.__adamw.epsilon
        self.use_master = self.__adamw.use_master
        self.scheduler = self.__adamw.scheduler
        self.min_lr = self.__adamw.min_lr

    def step_many(self, name_param_gradient:list[Any], train_contexts, batch_size, total_epoch) -> dict[Any,Any]:
        optimized = self.__adamw.step_many(name_param_gradient, train_contexts, batch_size, total_epoch)
        self.lr = self.__adamw.lr
        self.state = self.__adamw.state
        return optimized
    
    def to_dict(self, config_only:bool=True) -> dict[str, Any]:
        return self.__adamw.to_dict(config_only=config_only)
    
    @classmethod
    def from_dict(cls, thing) -> "Adam":
        adamw = AdamW.from_dict(thing)
        lr = adamw.lr
        beta1 = adamw.beta1
        beta2 = adamw.beta2
        epsilon = adamw.epsilon
        use_master = adamw.use_master
        scheduler = adamw.scheduler
        min_lr = adamw.min_lr
        adam = cls(lr=lr, beta1=beta1, beta2=beta2, epsilon=epsilon, use_master=use_master, scheduler=scheduler, min_lr=min_lr)

        adam.__adamw = adamw

        if not thing["config_only"]:
            adam.state = adamw.state
        else:
            adam.state = {}
        return adam
