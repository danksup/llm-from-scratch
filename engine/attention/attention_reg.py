import engine.backend as nx
from engine.activations import softmax_derivative
from engine.rope import rope_forward, rope_inverse
from typing import Any, Callable
import engine.initializers as initializer
from engine.rope import precompute_freqs

class AttentionFull:
    def __init__(self,embed_dim:int, n_heads:int, n_kv_heads:int=-1,  dtype:Any=nx.float16,  initializer:Callable=initializer.glorot_uniform, quantized:bool=False, *, use_symmetric=False, init=True) -> None:
        self.n_kv_heads = n_kv_heads

        if n_kv_heads < 0:
            n_kv_heads = n_heads

        self.n_kv_heads = n_kv_heads
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        assert embed_dim % n_heads == 0
        assert n_heads % n_kv_heads == 0, "cant have more kv heads than query heads."
        self.head_dim = embed_dim // n_heads
        assert self.head_dim % 2 == 0,  f"rope needs headdim to be multiple of 2, get headdim of {self.head_dim} instead. math: embed_dim // n_heads -> {embed_dim} // {n_heads} = {embed_dim//n_heads}"
        self.dtype = dtype

        self.n_rep = self.n_heads // self.n_kv_heads

        self.freqs = precompute_freqs(self.head_dim, 16384)

        self.configs = self.embed_dim, self.n_kv_heads, self.n_heads, self.n_rep, self.head_dim, self.freqs

        self.quantized = quantized
        self.scales = (None, None)
        self.biases = (None,None)

        if init:
            wqkv_shape = embed_dim + 2 * n_kv_heads * self.head_dim, embed_dim
            self.Wqkv = initializer(wqkv_shape, dtype=dtype)
            assert nx.isfinite(self.Wqkv).all(), f"non-finite detected when initializing attentipn.Wqkv."

            wo_shape = embed_dim,embed_dim
            self.Wo = initializer(wo_shape, dtype=dtype)
            assert nx.isfinite(self.Wo).all(), f"non-finite detected when initializing attentipn.Wo."

            self.use_symmetric = use_symmetric
            if quantized:
                self.Wqkv, wqkv_scale, wqkv_bias = nx.quantize(self.Wqkv, regular=use_symmetric)
                self.Wo, wo_scale, wo_bias = nx.quantize(self.Wo, regular=use_symmetric)
                self.scales = (wqkv_scale, wo_scale)
                self.biases = (wqkv_bias, wo_bias)

        self.dWqkv = None
        self.dWo = None

    @staticmethod
    def self_type() -> str:
        return "full"

    @classmethod
    def multihead(cls, embed_dim, n_heads, dtype, initializer, quantized, *, use_symmetric=False):
        if quantized:
            pass
        mha = cls(embed_dim, n_heads=n_heads, n_kv_heads=n_heads, dtype=dtype, initializer=initializer, quantized=quantized, use_symmetric=use_symmetric)
        return mha

    @classmethod
    def multiquery(cls, embed_dim, n_heads, dtype, initializer, quantized, *, use_symmetric=False):
        mqa = cls(embed_dim, n_heads=n_heads, n_kv_heads=1, dtype=dtype, initializer=initializer, quantized=quantized, use_symmetric=use_symmetric)
        return mqa

    @staticmethod
    def _forward(x:nx.ArrayLike, causal_mask:nx.ArrayLike,  attn_configs:tuple[Any,...], attn_params: tuple[Any,...], quantization,  *, use_symmetric:bool=False) -> tuple[nx.ArrayLike, tuple[nx.ArrayLike,...]]:
        #fp_16_x shape = (B,T,D)
        #Wqkv.T (D, D + 2 * n_kv_heads * H)
        #combined (B, T, D + 2 * n_kv_heads * H)
        embed_dim, n_kv_heads, n_heads, n_rep, head_dim, freqs = attn_configs
        Wqkv, Wo = attn_params

        wqkv_scale, wo_scale, wqkv_bias, wo_bias = quantization

        if wqkv_scale is not None:
            combined = nx.quantized_matmul(x, Wqkv, wqkv_scale,wqkv_bias, transpose=True, regular=use_symmetric)
        else:
            combined =  x @ Wqkv.T  # dtype

        Q = combined[..., :embed_dim] #(B, T, D)
        K = combined[..., embed_dim: embed_dim + (n_kv_heads * head_dim)]  #(B, T, n_kv_heads * H)
        V = combined[..., embed_dim + (n_kv_heads * head_dim):] #(B, T, n_kv_heads * H)

        B, T, _ = x.shape
        Q = Q.reshape(B, T, n_heads, head_dim).transpose(0,2,1,3)
        K = K.reshape(B, T, n_kv_heads, head_dim).transpose(0,2,1,3)
        V = V.reshape(B, T, n_kv_heads, head_dim).transpose(0,2,1,3)
        Q = rope_forward(Q, freqs)
        K = rope_forward(K, freqs)

        Q = Q.reshape(B,n_kv_heads, n_rep, T, head_dim)
        scores = nx.einsum("bkrQh,bkKh->bkrQK",Q, K) #(B, n_kv_heads, n_rep, Tq, Tk)

        scores = scores.astype(nx.float32) / nx.sqrt(head_dim, dtype=nx.float32) #type:ignore
        scores = nx.where(causal_mask, -1e9, scores)

        weights = nx.softmax(scores)
        weights = weights.astype(x.dtype)

        output = nx.einsum("bkrQK,bkKh->bkrQh",weights, V) #(B, n_kv_heads, n_rep, Tq, Dh)

        output = output.reshape(B, -1, T, head_dim)
        output_concat = output.transpose(0, 2, 1, 3).reshape(B, T, embed_dim)
        output_projected = nx.quantized_matmul(output_concat, Wo, wo_scale, wo_bias, regular=use_symmetric) #BTD

        cache =  (x, Q, K, V, weights, output_concat)
        return output_projected, cache

    @staticmethod
    def _backward(gradient:nx.ArrayLike, caches:tuple[Any,...], attn_configs:tuple[Any,...], attn_params: tuple[Any,...], quantization:tuple[Any,...]|None=None,  *, use_symmetric:bool=False) -> tuple[nx.ArrayLike,...]:
        x, Q, K, V, weights, output_concat = caches
        embed_dim, n_kv_heads, n_heads, n_rep, head_dim, freqs = attn_configs
        Wqkv, Wo = attn_params

        wqkv_scale, wo_scale, wqkv_bias, wo_bias = quantization #type:ignore

        B, T, _ = x.shape

        if wo_scale is not None:
            d_output_concat = nx.quantized_matmul(gradient, Wo, wo_scale,wo_bias, transpose=True, regular=use_symmetric) #B,T,D
        else:
            d_output_concat = gradient @ Wo.T

        d_output = d_output_concat.reshape(B, T, n_heads, head_dim).transpose(0, 2, 1, 3)
        d_output = d_output.reshape(B, n_kv_heads, n_rep, T, head_dim)

        dweights = nx.einsum("bkrQh,bkKh->bkrQK", d_output, V) #(B, n_kv_heads, n_rep, Tq, Tk)

        dV = nx.einsum("bkrQK,bkrQh->bkKh", weights, d_output)

        dscores = softmax_derivative(weights.astype(nx.float32), dweights.astype(nx.float32)) / nx.sqrt(head_dim, dtype=nx.float32) #type:ignore
        dscores = dscores.astype(gradient.dtype) #(B, n_kv_heads, n_rep, Tq, Tk)

        dQ = nx.einsum("bkrQK,bkKh->bkrQh", dscores, K) #(B, nkv, n_rep, Tq, Dh)

        dK = nx.einsum("bkrQK,bkrQh->bkKh",dscores, Q) #(B, nkv, Tk, Dh)

        del Q, dscores, dweights, d_output, d_output_concat

        dQ = dQ.reshape(B, -1, T, head_dim)
        dQ = rope_inverse(dQ, freqs)
        dK = rope_inverse(dK, freqs)

        dQ = dQ.transpose(0, 2, 1, 3).reshape(B, T, embed_dim)
        dK = dK.transpose(0, 2, 1, 3).reshape(B, T, n_kv_heads * head_dim)
        dV = dV.transpose(0, 2, 1, 3).reshape(B, T,  n_kv_heads * head_dim)

        dQKV = nx.concatenate([dQ, dK,dV], axis=-1)
        DQKV = dQKV.reshape(-1, embed_dim + 2 * (n_kv_heads * head_dim))

        del dQ, dK, dV

        X = x.reshape(-1, embed_dim)
        dWqkv = DQKV.T @ X

        H = output_concat.reshape(-1, embed_dim)
        G = gradient.reshape(-1, embed_dim)

        dWo = H.T @ G
        dx = nx.quantized_matmul(dQKV, Wqkv, wqkv_scale, wqkv_bias, regular=use_symmetric)

        # print("dx", dx.dtype)
        del x, output_concat, freqs, Wqkv, Wo
        return dx,dWqkv,dWo

    #TODO:compiled, dtype fix, quantization
    def inference_forward(self, x, max_cache_len, freqs, quantization, cached_k=None, cached_v=None, position = 0,  *, use_symmetric:bool=False):
        wqkv_scale, wo_scale, wqkv_bias, wo_bias = quantization #type:ignore
        if wqkv_scale is not None:
            combined = nx.quantized_matmul(x, self.Wqkv, wqkv_scale,wqkv_bias, transpose=True, regular=use_symmetric)
        else:
            combined =  x @ self.Wqkv.T  # dtype
        B, T, _ = x.shape

        K = combined[..., self.embed_dim: self.embed_dim + (self.n_kv_heads * self.head_dim)]

        K = K.reshape(B, T, self.n_kv_heads, self.head_dim).transpose(0,2,1,3)
        K = rope_forward(K, freqs, position)

        if cached_k is not None :
            cached_k = nx.concatenate([cached_k, K], axis = 2)
        else:
            cached_k = K

        V = combined[..., self.embed_dim + (self.n_kv_heads * self.head_dim):]
        V = V.reshape(B, T, self.n_kv_heads, self.head_dim).transpose(0,2,1,3)

        if cached_v is not None:
            cached_v = nx.concatenate([cached_v, V], axis = 2)
        else:
            cached_v = V

        if cached_k.shape[2] > max_cache_len:
            cached_k = cached_k[:, :, -max_cache_len:, :]

            cached_v = cached_v[:, :, -max_cache_len:, :]


        Q = combined[..., :self.embed_dim]
        Q = Q.reshape(B, T, self.n_heads, self.head_dim).transpose(0,2,1,3)
        Q = rope_forward(Q, freqs, position)

        repeats_cached_k = nx.repeat(cached_k, self.n_rep, axis=1 )
        repeats_cached_v = nx.repeat(cached_v, self.n_rep, axis=1 )

        scores = (Q @ repeats_cached_k.transpose(0,1,3,2)).astype(nx.float32) / nx.float_32(nx.sqrt(self.head_dim))
        weights = nx.softmax(scores)
        weights = weights.astype(x.dtype)
        output = weights @ repeats_cached_v
        output_concat = output.transpose(0, 2, 1, 3).reshape(B, T, self.embed_dim)
        output_projected = nx.quantized_matmul(output_concat, self.Wo, wo_scale, wo_bias, regular=use_symmetric) #BTD

        return output_projected, cached_k, cached_v

    @staticmethod
    def compute_mask(T):
        return nx.triu(nx.ones((T, T), dtype=nx.bool_), k=1)

    @classmethod
    def from_weight(cls, configs, weights,quants, dtype) -> "AttentionFull":
        embed_dim, n_kv_heads, n_heads, _, _, = configs
        wqkv, wo = weights

        attn = cls(embed_dim, n_heads, n_kv_heads,dtype, init=False)
        attn.Wqkv = wqkv
        attn.Wo = wo

        if quants is not None:
            scales, biases = quants
            attn.scales = scales
            attn.biases = biases

        return attn

    def copy(self):
        attn_copy = AttentionFull(self.embed_dim, self.n_heads, self.n_kv_heads, self.dtype, quantized=self.quantized, init=False)
        attn_copy.Wqkv = nx.copy(self.Wqkv)
        attn_copy.Wo = nx.copy(self.Wo)
        if self.quantized:
            attn_copy.scales = (nx.copy(self.scales[0]), nx.copy(self.scales[1]))
            attn_copy.biases = (nx.copy(self.biases[0]), nx.copy(self.biases[1]))

        return attn_copy