from engine.moe import MoE
import engine.attention as attn
from engine.rmsnorm import RMSNorm
from engine.dropout import Dropout
import engine.initializers as init
import engine.backend as nx
from typing import Any, Union
Attention = Union[attn.AttentionFull, attn.AttentionSWA]
import time

ATTN_TYPE = {
    "swa": attn.AttentionSWA,
    "full": attn.AttentionFull,
}

class TransformerBlock:
    def __init__(self ,embed_dim, attention:Attention, ff_dim, n_experts=1, cf=1.25, top_k =2, dtype=nx.float16, attn_init=init.glorot_uniform, moe_init=init.glorot_uniform) -> None:
        self.causal_mask = None
        self.embed_dim = embed_dim
        self.hidden_width = ff_dim
        self.n_experts = n_experts
        self.cf = cf
        self.dtype = dtype

        self.attention = attention
        self.attention_type = attention.self_type()
        self.ff = MoE(cf, top_k, n_experts, embed_dim, self.hidden_width, dtype=dtype, initializer=moe_init)
        self.rmsnorm1 = RMSNorm(embed_dim)
        self.rmsnorm2 = RMSNorm(embed_dim)
    
    def __repr__(self) -> str:
        param_count = self.param_count()
        this = {
            "param_count":param_count,
            "embed_dim":self.embed_dim,
            "hidden_width":self.hidden_width,
            "n_experts":self.n_experts,
        }
        # this_str = f""
        return str(this)
    
    def param_count(self) -> int:
        total = 0
        total += self.ff.Wcombined.size
        total += self.ff.Wout.size
        total += self.ff.router.size
        total += self.attention.Wqkv.size
        total += self.attention.Wo.size
        total += self.rmsnorm1.gamma.size
        total += self.rmsnorm2.gamma.size
        return total

    @staticmethod
    @nx.compile
    def _forward(x, causal_mask:Any, attention:str, attn_configs:tuple[Any,...], attn_params:tuple[Any,...], n_experts, cf, top_k:int, Wcombined:Any,router, hidden_width:int, Wout:Any, epsilon:float, gamma1:Any, gamma2:Any, p:float, is_training:bool) -> tuple[Any, Any, Any, Any, Any]:
        '''
        flow:
            input = x shape(B,T,D) -> rmsnorm(x) = rmsnorm_out -> attention(rmsnorm_out) + residual = attn_out
            \n
            -> rmsnorm(attn_out) = rmsnorm_out -> swiglu(rmsnorm_out)  -> ff_out + resudial = ff_out shape(B,T,D)
        '''
        # print("x block",x.dtype)
        rmsnorm1_out, caches_rmsnorm1 = RMSNorm._forward(x, gamma1,epsilon)

        rmsnorm1_out = rmsnorm1_out.astype(x.dtype) 

        attn_out, caches_attn = ATTN_TYPE[attention]._forward(rmsnorm1_out, causal_mask, attn_configs, attn_params)
        # print("attn forward", attn_out.dtype)

        drop_attn_out, mask1 = Dropout._forward(attn_out, p,is_training)
        # print("drop attn out", drop_attn_out.dtype)

        attn_out = drop_attn_out + x
        # print("attn out", attn_out.dtype)

        rmsnorm2_out, caches_rmsnorm2 = RMSNorm._forward(attn_out, gamma2,epsilon)

        rmsnorm2_out = rmsnorm2_out.astype(x.dtype) 

        ff_out, caches_ff, router_loss, normalized_histogram = MoE.forward(rmsnorm2_out, cf, top_k, router,n_experts,hidden_width,Wcombined, Wout)
        # print("ff_out", ff_out.dtype)
        
        drop_ff_out, mask2 =  Dropout._forward(ff_out, p,is_training)
        # print("drop ff out", drop_ff_out.dtype)

        ff_out = drop_ff_out + attn_out
        # print("FINAL BLOCK OUTPUT", ff_out.dtype)
        
        masks = (mask1, mask2)
        caches = (caches_attn, caches_ff, caches_rmsnorm1, caches_rmsnorm2)
        return ff_out, masks, caches, router_loss, normalized_histogram

    @staticmethod
    @nx.compile
    def _backward(gradient:Any, mask1:Any, mask2:Any, attention:str, p, caches_attn:tuple[Any,...], caches_ff:tuple[Any,...], caches_rmsnorm1:tuple[Any,...], caches_rmsnorm2:tuple[Any,...], attn_configs:tuple[Any,...], attn_params:tuple[Any,...], gamma1:Any, gamma2:Any, ff_params:tuple, moe_configs) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any]:
        d_ff_drop = Dropout._backward(gradient, mask2, 0.1) #grad dtype
        dx_ff,  dWcombined, dWout, d_router = MoE.backward(d_ff_drop, caches_ff, moe_configs, ff_params) #out:fp32
        d_rmsn2,d_gamma2 = RMSNorm._backward(dx_ff, caches_rmsnorm2 ,gamma2)

        d_attn_out = gradient + d_rmsn2
        d_attn_out = d_attn_out.astype(gradient.dtype)
        d_attn_drop = Dropout._backward(d_attn_out, mask1, p)
        # print("d_attn_drop", d_attn_drop.dtype)
        d_attn, dWqkv, dWo = ATTN_TYPE[attention]._backward(d_attn_drop, caches_attn, attn_configs, attn_params)
        # print("d_attn", d_attn.dtype)

        d_attn = d_attn.astype(nx.float32) #type:ignore
        d_rmsn1, d_gamma1 = RMSNorm._backward(d_attn,caches_rmsnorm1,gamma1)

        dx = d_rmsn1 + d_attn_out

        return dx, dWout, dWcombined, d_router, dWqkv,dWo, d_gamma1, d_gamma2
    
    def inference_forward(self, x, max_cache_len, cached_k=None, cached_v=None,  position=0):
        rmsnorm1_out, _ = RMSNorm._forward(x, self.rmsnorm1.gamma, self.rmsnorm1.epsilon)
        x = x.astype(x.dtype)

        attn_out, cached_k, cached_v = self.attention.inference_forward(rmsnorm1_out,max_cache_len, self.attention.freqs, cached_k, cached_v, position)
        attn_out = attn_out + x

        rmsnorm2_out, _ = RMSNorm._forward(attn_out, self.rmsnorm2.gamma, self.rmsnorm2.epsilon)

        ff_out,_,_,_ = MoE.forward(rmsnorm2_out, self.ff.cf, self.ff.top_k,self.ff.router, self.ff.n_experts, self.ff.hidden_width, self.ff.Wcombined, self.ff.Wout)
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

  