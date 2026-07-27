import engine.backend as nx
from typing import Any
from engine.optimizer.adamw import AdamW

class Adam:
    def __init__(self,min_lr=1e-4, max_lr=1e-3, beta1:float=0.9, beta2:float=0.999, epsilon:float=1e-8, use_master:bool=True) -> None:
        self.__adamw = AdamW(min_lr, max_lr, beta1, beta2, epsilon, 0.0, use_master)
        self.state = self.__adamw.state
        self.lr = self.__adamw.lr
        self.min_lr = self.__adamw.min_lr
        self.max_lr = self.__adamw.max_lr
        self.epsilon = self.__adamw.epsilon
        self.use_master = self.__adamw.use_master
    
    def step_many(self, name_param_gradient:list[Any], train_contexts, batch_size, total_epoch) -> dict[Any,Any]:
        optimized = self.__adamw.step_many(name_param_gradient, train_contexts, batch_size, total_epoch)
        self.lr = self.__adamw.lr
        self.state = self.__adamw.state
        return optimized
    
    def to_dict(self) -> dict[str, Any]:
        return self.__adamw.to_dict()
    
    @classmethod
    def from_dict(cls, thing) -> "Adam":
        adamw = AdamW.from_dict(thing)
        min_lr = adamw.min_lr
        max_lr = adamw.max_lr
        beta1 = adamw.beta1
        beta2 = adamw.beta2
        epsilon = adamw.epsilon
        use_master = adamw.use_master
        adam = cls(min_lr, max_lr, beta1, beta2, epsilon, use_master)

        adam.__adamw = adamw
        adam.lr = adamw.lr
        adam.state = adamw.state
        return adam
