from typing import Any, Union

import engine.attention as attn
import engine.backend as nx
import engine.initializers as init
from engine.dropout import Dropout
from engine.moe import MoE
from engine.rmsnorm import RMSNorm

Attention = attn.AttentionFull | attn.AttentionSWA
import time

ATTN_TYPE = {
    "swa": attn.AttentionSWA,
    "full": attn.AttentionFull,
}

class TransformerBlock:
    def __init__(self ,attention:Attention, ff:MoE, rmsnorm1:RMSNorm, rmsnorm2:RMSNorm) -> None:
        self.causal_mask = None

        self.attention = attention
        self.attention_type = attention.self_type()
        self.ff = ff
        self.rmsnorm1 = rmsnorm1
        self.rmsnorm2 = rmsnorm2
        

    def __str__(self) -> str:
        param_count = self.count_param()
        this = {
            "param_count":param_count,
        }
        # this_str = f""
        return str(this)

    def count_param(self, *, quantized=False, use_symmetric=False) -> int:
        total = 0
        if quantized and nx.backend == "MLX" and not use_symmetric:
            total += self.ff.Wcombined.size * 4
            total += self.ff.Wout.size * 4
            total += self.attention.Wqkv.size * 4
            total += self.attention.Wo.size * 4
        else:
            total += self.ff.Wcombined.size
            total += self.ff.Wout.size
            total += self.attention.Wqkv.size
            total += self.attention.Wo.size

        total += self.ff.router.size
        total += self.rmsnorm1.gamma.size
        total += self.rmsnorm2.gamma.size
        return total

    @staticmethod
    @nx.compile
    def _forward(x, causal_mask:Any, attention:str, attn_configs:tuple[Any,...], attn_params:tuple[Any,...], ff_configs, ff_params, epsilon:float, gamma1:Any, gamma2:Any, p:float, is_training:bool, quantization:tuple[Any,...]|None=None, *, use_symmetric:bool=False) -> tuple[Any, Any, Any, Any, Any]:
        rmsnorm1_out, caches_rmsnorm1 = RMSNorm._forward(x, gamma1,epsilon)

        rmsnorm1_out = rmsnorm1_out.astype(x.dtype)

        attn_out, caches_attn = ATTN_TYPE[attention]._forward(rmsnorm1_out, causal_mask, attn_configs, attn_params, quantization[0], use_symmetric=use_symmetric) #type:ignore
        drop_attn_out, mask1 = Dropout._forward(attn_out, p,is_training)

        attn_out = drop_attn_out + x

        rmsnorm2_out, caches_rmsnorm2 = RMSNorm._forward(attn_out, gamma2,epsilon)

        rmsnorm2_out = rmsnorm2_out.astype(x.dtype)
        ff_out, caches_ff, aux_loss, normalized_histogram = MoE.forward(rmsnorm2_out, ff_configs, ff_params, quantization[1], use_symmetric=use_symmetric) #type:ignore
        drop_ff_out, mask2 =  Dropout._forward(ff_out, p,is_training)

        ff_out = drop_ff_out + attn_out

        masks = (mask1, mask2)
        caches = (caches_attn, caches_ff, caches_rmsnorm1, caches_rmsnorm2)
        return ff_out, masks, caches, aux_loss, normalized_histogram

    @staticmethod
    @nx.compile
    def _backward(gradient:Any, mask1:Any, mask2:Any, attention:str, p, caches_attn:tuple[Any,...], caches_ff:tuple[Any,...], caches_rmsnorm1:tuple[Any,...], caches_rmsnorm2:tuple[Any,...], attn_configs:tuple[Any,...], attn_params:tuple[Any,...], gamma1:Any, gamma2:Any, ff_params:tuple, moe_configs, quantization:tuple[Any,...]|None=None, *, use_symmetric:bool=False) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any]:
        d_ff_drop = Dropout._backward(gradient, mask2, p) #grad dtype
        dx_ff,  dWcombined, dWout, d_router = MoE.backward(d_ff_drop, caches_ff, moe_configs, ff_params, quantization[1], use_symmetric=use_symmetric) #out:fp32 #type:ignore

        d_rmsn2,d_gamma2 = RMSNorm._backward(dx_ff, caches_rmsnorm2 ,gamma2)

        d_attn_out = gradient + d_rmsn2

        d_attn_out = d_attn_out.astype(gradient.dtype)
        d_attn_drop = Dropout._backward(d_attn_out, mask1, p)
        d_attn, dWqkv, dWo = ATTN_TYPE[attention]._backward(d_attn_drop, caches_attn, attn_configs, attn_params, quantization[0], use_symmetric=use_symmetric) #type:ignore

        d_attn = d_attn.astype(nx.float32)
        d_rmsn1, d_gamma1 = RMSNorm._backward(d_attn,caches_rmsnorm1,gamma1)

        dx = d_rmsn1 + d_attn_out

        return dx, dWout, dWcombined, d_router, dWqkv,dWo, d_gamma1, d_gamma2

    #TODO:compiled, dtype consistency fix/check
    def inference_forward(self, x, max_cache_len, cached_k=None, cached_v=None,  position=0, * ,use_symmetric=False):
        # print("x", x.dtype)

        rmsnorm1_out, _ = RMSNorm._forward(x, self.rmsnorm1.gamma, self.rmsnorm1.epsilon)
        rmsnorm1_out = rmsnorm1_out.astype(x.dtype)

        attn_quant_params = self.attention.scales + self.attention.biases
        attn_out, cached_k, cached_v = self.attention.inference_forward(x=rmsnorm1_out, max_cache_len=max_cache_len,freqs= self.attention.freqs, quantization=attn_quant_params, cached_k=cached_k, cached_v= cached_v, position=position, use_symmetric=use_symmetric)
        attn_out = attn_out + x

        rmsnorm2_out, _ = RMSNorm._forward(attn_out, self.rmsnorm2.gamma, self.rmsnorm2.epsilon)
        rmsnorm2_out = rmsnorm2_out.astype(x.dtype)

        ff_quant_params = self.ff.scales + self.ff.biases
        ff_out,_,_,_ = MoE.forward(rmsnorm2_out, self.ff.configs, (self.ff.Wcombined, self.ff.Wout, self.ff.router), ff_quant_params, use_symmetric=use_symmetric)
        ff_out = ff_out + attn_out

        return ff_out, cached_k, cached_v

    def get_configs(self):
        return {
            "attn_type": self.attention.self_type(),
            "attention": list(self.attention.configs)[0:-1],
            "ff":self.ff.configs,
            "rmsnorm1": self.rmsnorm1.configs,
            "rmsnorm2": self.rmsnorm2.configs,
        }

    @classmethod
    def from_weights(cls, attn_type, attn_configs, attn_weights,attn_quants, ff_configs, ff_weights,ff_quants, rmsnorm1_configs,gamma1, rmsnorm2_configs,gamma2, dtype):
        attn = ATTN_TYPE[attn_type].from_weight(attn_configs, attn_weights, quants=attn_quants, dtype=dtype)
        ff = MoE.from_weight(configs=ff_configs, weights=ff_weights, quants=ff_quants, dtype=dtype)
        rmsnorm1 = RMSNorm.from_weight(rmsnorm1_configs, gamma1)
        rmsnorm2 = RMSNorm.from_weight(rmsnorm2_configs, gamma2)
        block = TransformerBlock(attn, ff, rmsnorm1, rmsnorm2)
        return block

    def copy(self):
        attn_copy = self.attention.copy()
        ff_copy = self.ff.copy()
        rms1_copy = self.rmsnorm1.copy()
        rms2_copy = self.rmsnorm2.copy()
        
        block_copy = TransformerBlock(attn_copy, ff_copy, rms1_copy, rms2_copy)
        return block_copy

