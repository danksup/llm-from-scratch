from engine.moe import MoE
from engine.attention import AttentionLayer
from engine.rmsnorm import RMSNorm
from engine.dropout import Dropout
from engine.rope import precompute_freqs
import engine.backend as nx
from typing import Any
import time

class TransformerBlock:
    def __init__(self,embed_dim,ff_dim, n_heads, n_kv_heads, n_experts=1, cf=1.25, top_k =2, W=8, dtype=nx.float16) -> None:
        self.causal_mask = None
        self.embed_dim = embed_dim
        self.hidden_width = ff_dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = self.n_heads // self.n_kv_heads 
        self.W = W
        assert embed_dim % n_heads == 0
        self.head_dim = embed_dim // n_heads
        self.n_experts = n_experts
        self.cf = cf
        self.dtype = dtype

        assert self.head_dim % 2 == 0, "head dim !% 2"
        self.freqs = precompute_freqs(self.head_dim, 16384)

        self.attention = AttentionLayer(embed_dim, n_heads, n_kv_heads, dtype=dtype)
        self.ff = MoE(cf, top_k, n_experts, embed_dim, self.hidden_width, dtype=dtype)
        self.rmsnorm1 = RMSNorm(embed_dim)
        self.rmsnorm2 = RMSNorm(embed_dim)

    @nx.compile
    @staticmethod
    def _forward(x, causal_mask:Any, embed_dim:int, n_heads:int, n_kv_heads, n_rep, W, head_dim:int, n_experts, cf, top_k:int,freqs:Any, Wqkv:Any, Wo:Any, Wcombined:Any,router, hidden_width:int, Wout:Any, epsilon:float, gamma1:Any, gamma2:Any, p:float, is_training:bool) -> tuple[Any, Any, Any, Any, Any]:
        '''
        flow:
            input = x shape(B,T,D) -> rmsnorm(x) = rmsnorm_out -> attention(rmsnorm_out) + residual = attn_out
            \n
            -> rmsnorm(attn_out) = rmsnorm_out -> swiglu(rmsnorm_out)  -> ff_out + resudial = ff_out shape(B,T,D)
        '''
        # print("x block",x.dtype)
        rmsnorm1_out, caches_rmsnorm1 = RMSNorm._forward(x, gamma1,epsilon)

        rmsnorm1_out = rmsnorm1_out.astype(x.dtype) 

        attn_out, caches_attn = AttentionLayer._forward(rmsnorm1_out,causal_mask, embed_dim, n_kv_heads,n_heads, n_rep, head_dim, W, freqs, Wqkv, Wo,)
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

    @nx.compile
    @staticmethod
    def _backward(gradient:Any, mask1:Any, mask2:Any, p, caches_attn:tuple[Any,...], caches_ff:tuple[Any,...], caches_rmsnorm1:tuple[Any,...], caches_rmsnorm2:tuple[Any,...], attn_params:tuple[Any,...], gamma1:Any, gamma2:Any, ff_params:tuple, moe_configs) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any]:
        d_ff_drop = Dropout._backward(gradient, mask2, 0.1) #grad dtype
        dx_ff,  dWcombined, dWout, d_router = MoE.backward(d_ff_drop, caches_ff, moe_configs, ff_params) #out:fp32
        d_rmsn2,d_gamma2 = RMSNorm._backward(dx_ff, caches_rmsnorm2 ,gamma2)

        d_attn_out = gradient + d_rmsn2
        d_attn_out = d_attn_out.astype(gradient.dtype)
        d_attn_drop = Dropout._backward(d_attn_out, mask1, p)
        # print("d_attn_drop", d_attn_drop.dtype)
        d_attn, dWqkv, dWo = AttentionLayer._backward(d_attn_drop, caches_attn, attn_params)
        # print("d_attn", d_attn.dtype)

        d_attn = d_attn.astype(nx.float32)
        d_rmsn1, d_gamma1 = RMSNorm._backward(d_attn,caches_rmsnorm1,gamma1)

        dx = d_rmsn1 + d_attn_out

        return dx, dWout, dWcombined, d_router, dWqkv,dWo, d_gamma1, d_gamma2
    
    def inference_forward(self, x, max_cache_len, cached_k=None, cached_v=None,  position=0):
        rmsnorm1_out, _ = RMSNorm._forward(x, self.rmsnorm1.gamma, self.rmsnorm1.epsilon)
        x = x.astype(x.dtype)

        attn_out, cached_k, cached_v = self.attention.inference_forward(rmsnorm1_out,max_cache_len, self.freqs, cached_k, cached_v, position)
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
                "n_heads": self.n_heads,
                "hidden_width":self.hidden_width,
                "embed_dim":self.embed_dim,
                "n_kv_heads": self.n_kv_heads,
                "dtype": nx.dtype_to_srt[self.dtype]
            },
            "attention":self.attention.to_dict(),
            "ff":self.ff.to_dict(),
            "rmsnorm1":self.rmsnorm1.to_dict(),
            "rmsnorm2":self.rmsnorm2.to_dict(),
        } 
    
    @classmethod
    def from_dict(cls,thing:dict) -> "TransformerBlock":
        configs = thing["block_configs"]
        transformer_block = cls(configs["embed_dim"], configs["hidden_width"], configs["n_heads"], configs["n_kv_heads"], configs["n_experts"],configs["cf"], dtype = nx.str_to_dtype[configs["dtype"]])
        transformer_block.attention = AttentionLayer.from_dict(thing["attention"])
        transformer_block.ff = MoE.from_dict(thing["ff"])
        transformer_block.rmsnorm1 = RMSNorm.from_dict(thing["rmsnorm1"])
        transformer_block.rmsnorm2 = RMSNorm.from_dict(thing["rmsnorm2"])
        return transformer_block

  