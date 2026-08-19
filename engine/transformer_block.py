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
    def __init__(self ,embed_dim, attention:Attention, ff_dim, n_experts=1, cf=1.25, top_k =2, dtype=nx.float16, attn_init=init.glorot_uniform, moe_init=init.glorot_uniform, quantized:bool=False) -> None:
        self.causal_mask = None
        self.embed_dim = embed_dim
        self.hidden_width = ff_dim
        self.n_experts = n_experts
        self.cf = cf
        self.dtype = dtype

        self.attention = attention
        self.attention_type = attention.self_type()
        self.ff = MoE(cf, top_k, n_experts, embed_dim, self.hidden_width, dtype=dtype, initializer=moe_init, quantized=quantized)
        self.rmsnorm1 = RMSNorm(embed_dim)
        self.rmsnorm2 = RMSNorm(embed_dim)
        self.quantized = quantized

    def __str__(self) -> str:
        param_count = self.count_param()
        this = {
            "param_count":param_count,
            "embed_dim":self.embed_dim,
            "hidden_width":self.hidden_width,
            "n_experts":self.n_experts,
        }
        # this_str = f""
        return str(this)

    def count_param(self) -> int:
        total = 0
        if self.quantized and nx.backend == "MLX":
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
    def _forward(x, causal_mask:Any, attention:str, attn_configs:tuple[Any,...], attn_params:tuple[Any,...], ff_configs, ff_params, epsilon:float, gamma1:Any, gamma2:Any, p:float, is_training:bool, quantization:tuple[Any,...]|None=None) -> tuple[Any, Any, Any, Any, Any]:
        rmsnorm1_out, caches_rmsnorm1 = RMSNorm._forward(x, gamma1,epsilon)

        rmsnorm1_out = rmsnorm1_out.astype(x.dtype)
        attn_out, caches_attn = ATTN_TYPE[attention]._forward(rmsnorm1_out, causal_mask, attn_configs, attn_params, quantization[0]) #type:ignore
        drop_attn_out, mask1 = Dropout._forward(attn_out, p,is_training)

        attn_out = drop_attn_out + x

        rmsnorm2_out, caches_rmsnorm2 = RMSNorm._forward(attn_out, gamma2,epsilon)

        rmsnorm2_out = rmsnorm2_out.astype(x.dtype)
        ff_out, caches_ff, router_loss, normalized_histogram = MoE.forward(rmsnorm2_out, ff_configs, ff_params, quantization[1]) #type:ignore
        drop_ff_out, mask2 =  Dropout._forward(ff_out, p,is_training)

        ff_out = drop_ff_out + attn_out

        masks = (mask1, mask2)
        caches = (caches_attn, caches_ff, caches_rmsnorm1, caches_rmsnorm2)
        return ff_out, masks, caches, router_loss, normalized_histogram

    @staticmethod
    @nx.compile
    def _backward(gradient:Any, mask1:Any, mask2:Any, attention:str, p, caches_attn:tuple[Any,...], caches_ff:tuple[Any,...], caches_rmsnorm1:tuple[Any,...], caches_rmsnorm2:tuple[Any,...], attn_configs:tuple[Any,...], attn_params:tuple[Any,...], gamma1:Any, gamma2:Any, ff_params:tuple, moe_configs, quantization:tuple[Any,...]|None=None) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any]:
        d_ff_drop = Dropout._backward(gradient, mask2, p) #grad dtype
        dx_ff,  dWcombined, dWout, d_router = MoE.backward(d_ff_drop, caches_ff, moe_configs, ff_params, quantization[1]) #out:fp32 #type:ignore

        d_rmsn2,d_gamma2 = RMSNorm._backward(dx_ff, caches_rmsnorm2 ,gamma2)

        d_attn_out = gradient + d_rmsn2

        d_attn_out = d_attn_out.astype(gradient.dtype)
        d_attn_drop = Dropout._backward(d_attn_out, mask1, p)
        d_attn, dWqkv, dWo = ATTN_TYPE[attention]._backward(d_attn_drop, caches_attn, attn_configs, attn_params, quantization[0]) #type:ignore

        d_attn = d_attn.astype(nx.float32)
        d_rmsn1, d_gamma1 = RMSNorm._backward(d_attn,caches_rmsnorm1,gamma1)

        dx = d_rmsn1 + d_attn_out

        return dx, dWout, dWcombined, d_router, dWqkv,dWo, d_gamma1, d_gamma2

    #TODO:compiled, dtype fix, quantization
    def inference_forward(self, x, max_cache_len, cached_k=None, cached_v=None,  position=0):
        # print("x", x.dtype)
        rmsnorm1_out, _ = RMSNorm._forward(x, self.rmsnorm1.gamma, self.rmsnorm1.epsilon)
        rmsnorm1_out = rmsnorm1_out.astype(x.dtype)

        attn_quant_params = self.attention.scales + self.attention.biases
        attn_out, cached_k, cached_v = self.attention.inference_forward(x=rmsnorm1_out, max_cache_len=max_cache_len,freqs= self.attention.freqs, quantization=attn_quant_params, cached_k=cached_k, cached_v= cached_v, position=position)
        attn_out = attn_out + x

        rmsnorm2_out, _ = RMSNorm._forward(attn_out, self.rmsnorm2.gamma, self.rmsnorm2.epsilon)
        rmsnorm2_out = rmsnorm2_out.astype(x.dtype)

        ff_quant_params = self.ff.scales + self.ff.biases
        ff_out,_,_,_ = MoE.forward(rmsnorm2_out, self.ff.configs, (self.ff.Wcombined, self.ff.Wout, self.ff.router), ff_quant_params)
        ff_out = ff_out + attn_out

        return ff_out, cached_k, cached_v

    def to_dict(self) -> dict:
        return {
            "block_configs": {
                "cf":self.cf,
                "n_experts" :self.n_experts,
                "hidden_width":self.hidden_width,
                "embed_dim":self.embed_dim,
                "dtype": nx.dtype_to_srt[self.dtype]
            },
            "attention":{"type":self.attention.self_type(), "param":self.attention.to_dict()},
            "ff":self.ff.to_dict(),
            "rmsnorm1":self.rmsnorm1.to_dict(),
            "rmsnorm2":self.rmsnorm2.to_dict(),
        }

    @classmethod
    def from_dict(cls,thing:dict) -> "TransformerBlock":
        configs = thing["block_configs"]
        attn_cls = ATTN_TYPE[thing["attention"]["type"]]
        attn_param = thing["attention"]["param"]
        attention = attn_cls.from_dict(attn_param)
        transformer_block = cls(configs["embed_dim"], attention, configs["hidden_width"], configs["n_experts"],configs["cf"], dtype = nx.str_to_dtype[configs["dtype"]])
        transformer_block.ff = MoE.from_dict(thing["ff"])
        transformer_block.rmsnorm1 = RMSNorm.from_dict(thing["rmsnorm1"])
        transformer_block.rmsnorm2 = RMSNorm.from_dict(thing["rmsnorm2"])
        return transformer_block
