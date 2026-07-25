import engine.backend as nx
from typing import Any

class RMSNorm:
    def __init__(self, embed_dim:int, epsilon:float=1e-5) -> None:
        self.gamma = nx.ones((embed_dim,), dtype=nx.float32)
        self.d_gamma = None
        self.epsilon = nx.float_32(epsilon)

    @staticmethod
    def _forward(x:nx.ArrayLike, gamma:nx.ArrayLike, epsilon:Any) -> tuple[nx.ArrayLike, tuple[Any,...]]:
        '''
        rmsnorm = gamma * (x / sqrt(mean(x^2) + epsilon)) \n
        gamma = trainable weight. shape (D,)\n
        epsilon = very small value to prevent division by 0
        '''
        x32 = x.astype(nx.float32) #type:ignore
        rms = nx.sqrt(nx.mean(x32 * x32, axis=-1, keepdims=True) + epsilon)
        rms = rms
        normalized = x32 / rms
        rmsnorm = gamma * normalized
        caches = (normalized, rms, x32)
        return rmsnorm, caches
    
    @staticmethod
    def _backward(gradient:nx.ArrayLike, caches:tuple[Any,...], gamma:nx.ArrayLike) -> tuple[nx.ArrayLike,...]:
        normalized, rms, x32  = caches
        d_gamma = nx.sum(gradient * normalized, axis=(0, 1), dtype=nx.float32)
        dx_norm = gradient * gamma
        d = x32.shape[-1]
        sum_term = nx.sum(dx_norm * normalized, axis=-1, keepdims=True)
        dx = (dx_norm - (normalized / d) * sum_term) / rms

        return dx, d_gamma
    
    def to_dict(self)-> dict[str, Any]:
        return {
            "gamma":self.gamma.tolist(),
        }
    
    @classmethod
    def from_dict(cls, thing:dict[str, Any]) -> "RMSNorm": 
        rmsnorm = cls(len(thing["gamma"]))
        rmsnorm.gamma = nx.array(thing["gamma"],dtype=nx.float32)

        return rmsnorm