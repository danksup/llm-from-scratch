from typing import Any

import engine.attention as attn
import engine.backend as nx
from engine.dropout import Dropout
from engine.ffn import FFN
from engine.rmsnorm import RMSNorm
Attention = attn.AttentionFull | attn.AttentionChunked

ATTN_TYPE = {
    "chunked": attn.AttentionChunked,
    "full": attn.AttentionFull,
}

class TransformerBlock:
    def __init__(self ,attention:Attention, ff:FFN, rmsnorm1:RMSNorm, rmsnorm2:RMSNorm) -> None:
        self.causal_mask = None

        self.attention = attention
        self.attention_type = attention.self_type()
        self.ff = ff
        self.rmsnorm1 = rmsnorm1
        self.rmsnorm2 = rmsnorm2

    def zeroes_gradient(self):
        self.attention.zeroes_gradient()
        self.ff.zeroes_gradient()
        self.rmsnorm1.zeroes_gradient()
        self.rmsnorm2.zeroes_gradient()

    def __str__(self) -> str:
        param_count = self.count_params()
        this = {
            "param_count":param_count,
        }
        # this_str = f""
        return str(this)

    def count_params(self, dense=False, *, quantized=False) -> int:
        total = 0
        if quantized:
            total += self.attention.Wqkv.size * 4
            total += self.attention.Wo.size * 4
        else:
            total += self.attention.Wqkv.size
            total += self.attention.Wo.size

        ff_param_sparse, ff_param_dense = self.ff.count_params()
        total += ff_param_dense if dense else ff_param_sparse
        total += self.rmsnorm1.gamma.size
        total += self.rmsnorm2.gamma.size
        total += self.attention.Q_norm.gamma.size
        total += self.attention.K_norm.gamma.size
        return total

    @staticmethod
    @nx.compile
    def _forward(x, causal_mask:Any,doc_id, attention:str, attn_configs:tuple[Any,...], attn_params:tuple[Any,...], ff_configs, ff_params, epsilon:float, gamma1:Any, gamma2:Any, p:float, is_training:bool, quantization:tuple[Any,...]|None=None, *, recompute_activation:bool=True) -> tuple[Any, Any, Any, Any, Any]:
        rmsnorm1_out, caches_rmsnorm1 = RMSNorm._forward(x, gamma1,epsilon)

        rmsnorm1_out = rmsnorm1_out.astype(x.dtype)

        attn_out, caches_attn = ATTN_TYPE[attention]._forward(rmsnorm1_out, causal_mask,doc_id, attn_configs, attn_params, quantization[0], recompute_activation=recompute_activation) #type:ignore
        drop_attn_out, mask1 = Dropout._forward(attn_out, p,is_training)

        attn_out = drop_attn_out + x

        rmsnorm2_out, caches_rmsnorm2 = RMSNorm._forward(attn_out, gamma2,epsilon)

        rmsnorm2_out = rmsnorm2_out.astype(x.dtype)
        ff_out, caches_ff, z_loss, normalized_histogram = FFN.forward(rmsnorm2_out, ff_configs, ff_params, quantization[1], recompute_activation=recompute_activation, is_training=is_training) #type:ignore
        drop_ff_out, mask2 =  Dropout._forward(ff_out, p,is_training)

        ff_out = drop_ff_out + attn_out

        masks = (mask1, mask2)
        caches = (caches_attn, caches_ff, caches_rmsnorm1, caches_rmsnorm2)
        return ff_out, masks, caches, z_loss, normalized_histogram

    @staticmethod
    @nx.compile
    def _backward(gradient:Any, mask1:Any, mask2:Any, attention:str, p, caches_attn:tuple[Any,...], caches_ff:tuple[Any,...], caches_rmsnorm1:tuple[Any,...], caches_rmsnorm2:tuple[Any,...], attn_configs:tuple[Any,...], attn_params:tuple[Any,...], gamma1:Any, gamma2:Any, ff_params:tuple, ff_configs, gradient_scale, quantization:tuple[Any,...]|None=None, *, recompute_activation:bool=True) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any,Any, Any, Any, Any]:
        d_ff_drop = Dropout._backward(gradient, mask2, p) #grad dtype

        dx_ff, dWcombined, dWout, d_router, dW_dense_combined, dW_dense_out = FFN.backward(d_ff_drop, caches_ff, ff_configs, ff_params,gradient_scale, quantization[1], recompute_activation=recompute_activation) #out:fp32 #type:ignore

        d_rmsn2,d_gamma2 = RMSNorm._backward(dx_ff, caches_rmsnorm2 ,gamma2)

        d_attn_out = gradient + d_rmsn2

        d_attn_out = d_attn_out.astype(gradient.dtype)
        d_attn_drop = Dropout._backward(d_attn_out, mask1, p)

        d_attn, dWqkv, dWo, Q_norm_d_gamma, K_norm_d_gamma = ATTN_TYPE[attention]._backward(d_attn_drop, caches_attn, attn_configs, attn_params, quantization[0], recompute_activation=recompute_activation) #type:ignore

        d_attn = d_attn.astype(nx.float32)
        d_rmsn1, d_gamma1 = RMSNorm._backward(d_attn,caches_rmsnorm1,gamma1)

        dx = d_rmsn1 + d_attn_out

        return dx, dWout, dWcombined, d_router,dW_dense_combined, dW_dense_out, dWqkv,dWo, d_gamma1, d_gamma2, Q_norm_d_gamma, K_norm_d_gamma

    #TODO:compiled, dtype consistency fix/check
    def inference_forward(self, x, max_cache_len, cached_k=None, cached_v=None,  position=0):
        rmsnorm1_out, _ = RMSNorm._forward(x, self.rmsnorm1.gamma, self.rmsnorm1.epsilon)
        rmsnorm1_out = rmsnorm1_out.astype(x.dtype)

        attn_quant_params = self.attention.scales + self.attention.biases
        attn_out, cached_k, cached_v = self.attention.inference_forward(x=rmsnorm1_out, max_cache_len=max_cache_len,freqs= self.attention.freqs, quantization=attn_quant_params, cached_k=cached_k, cached_v= cached_v, position=position)
        attn_out = attn_out + x

        rmsnorm2_out, _ = RMSNorm._forward(attn_out, self.rmsnorm2.gamma, self.rmsnorm2.epsilon)
        rmsnorm2_out = rmsnorm2_out.astype(x.dtype)

        ff_quant_params = self.ff.scales + self.ff.biases
        ff_out= FFN.inference_forward(rmsnorm2_out, self.ff.configs, (self.ff.Wcombined, self.ff.Wout, self.ff.router, self.ff.W_dense_combined, self.ff.W_dense_out, self.ff.router_bias ), ff_quant_params)
        ff_out = ff_out + attn_out

        return ff_out, cached_k, cached_v

    def get_configs(self):
        return {
            "attn_type": self.attention.self_type(),
            "attention": list(self.attention.configs)[0:-1],
            "attention_Q_norm": self.attention.Q_norm.configs,
            "attention_K_norm": self.attention.K_norm.configs,
            "ff":self.ff.configs,
            "rmsnorm1": self.rmsnorm1.configs,
            "rmsnorm2": self.rmsnorm2.configs,
        }
    
    def copy(self):
        attn_copy = self.attention.copy()
        ff_copy = self.ff.copy()
        rms1_copy = self.rmsnorm1.copy()
        rms2_copy = self.rmsnorm2.copy()
        
        block_copy = TransformerBlock(attn_copy, ff_copy, rms1_copy, rms2_copy)
        return block_copy

    def quantize(self):
        self.attention.quantize()
        self.ff.quantize()

    def set_gradient(self,**kwargs):#*, dWout, dWcombined, d_router,dW_dense_combined, dW_dense_out, dWqkv,dWo, d_gamma1, d_gamma2, Q_norm_d_gamma, K_norm_d_gamma):
        self.attention.set_gradient(kwargs["dwqkv"], kwargs["dwo"], kwargs["q_d_gamma"], kwargs["k_d_gamma"])
        self.rmsnorm1.set_gradient(kwargs["tblock_d_gamma1"])
        self.rmsnorm2.set_gradient(kwargs["tblock_d_gamma2"])
        self.ff.set_gradient(dWcombined=kwargs["dwcombined"], dWout=kwargs["dwout"], d_router=kwargs["d_router"], dW_dense_combined=kwargs["dw_dense_combined"],dW_dense_out=kwargs["dw_dense_out"])

