import engine.backend as nx
from engine.activations import softmax_derivative
from engine.rope import rope_forward, rope_inverse
from typing import Any, Callable
import engine.initializers as initializer
from engine.rope import precompute_freqs
from engine.rmsnorm import RMSNorm
import time

#TODO: do the thing
class AttentionChunked:
    def __init__(self,embed_dim:int, n_heads:int, Q_norm:RMSNorm, K_norm:RMSNorm, n_kv_heads:int=-1, chunk_size=8, dtype:Any=nx.float16, initializer:Callable=initializer.glorot_uniform, quantized:bool=False, *,use_symmetric:bool=False, init=True) -> None:
        self.n_kv_heads = n_kv_heads

        if n_kv_heads < 0:
            n_kv_heads = n_heads

        self.n_kv_heads = n_kv_heads
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        assert embed_dim % n_heads == 0
        assert n_heads % n_kv_heads == 0, "cant have more kv heads than query heads."
        head_dim = embed_dim // n_heads
        self.head_dim = head_dim
        assert self.head_dim % 2 == 0,  f"rope needs headdim to be multiple of 2, get headdim of {self.head_dim} instead. math: embed_dim // n_heads -> {embed_dim} // {n_heads} = {embed_dim//n_heads}"

        self.chunk_size = chunk_size
        self.dtype = dtype

        self.n_rep = self.n_heads // self.n_kv_heads

        self.freqs = precompute_freqs(self.head_dim, 16384)

        self.configs = self.embed_dim, self.n_kv_heads, self.n_heads, self.n_rep, head_dim, self.chunk_size, self.freqs

        self.quantized = quantized

        self.scales = (None, None)
        self.biases = (None,None)

        wqkv_shape = embed_dim + 2 * n_kv_heads * self.head_dim, embed_dim
        wo_shape = embed_dim,embed_dim

        self.Q_norm = Q_norm
        self.K_norm = K_norm


        if init:
            self.Wqkv = initializer(wqkv_shape, dtype=dtype)
            assert nx.isfinite(self.Wqkv).all(), f"non-finite detected when initializing attentipn.Wqkv."

            self.Wo = initializer(wo_shape, dtype=dtype)
            assert nx.isfinite(self.Wo).all(), f"non-finite detected when initializing attentipn.Wo."

            self.use_symmetric = use_symmetric

            if quantized:
                self.Wqkv, wqkv_scale, wqkv_bias = nx.quantize(self.Wqkv, regular=use_symmetric)
                self.Wo, wo_scale, wo_bias = nx.quantize(self.Wo, regular=use_symmetric)
                self.scales = (wqkv_scale, wo_scale)
                self.biases = (wqkv_bias, wo_bias)


        self.dWqkv = nx.zeros(wqkv_shape)
        self.dWo = nx.zeros(wo_shape)

    def zeroes_gradient(self):
        self.dWqkv = nx.zeros_like(self.dWqkv)
        self.dWo = nx.zeros_like(self.dWo)
        # self.Q_norm.d_gamma = nx.zeros_like(self.Q_norm.d_gamma)
        # self.K_norm.d_gamma = nx.zeros_like(self.K_norm.d_gamma)

    @staticmethod
    def self_type() -> str:
        return "chunked"

    @classmethod
    def multihead(cls,embed_dim, n_heads, chunk_size, dtype, initializer, Q_norm:RMSNorm, K_norm:RMSNorm,*, use_symmetric=False):
        mha = cls(embed_dim, n_heads=n_heads, n_kv_heads=n_heads, chunk_size=chunk_size, dtype=dtype, initializer=initializer, Q_norm=Q_norm, K_norm=K_norm, use_symmetric=use_symmetric)
        return mha
    @classmethod
    def multiquery(cls,embed_dim, n_heads, chunk_size, dtype, initializer, Q_norm:RMSNorm, K_norm:RMSNorm, *, use_symmetric=False):
        mqa = cls(embed_dim, n_heads=n_heads, n_kv_heads=1, chunk_size=chunk_size, dtype=dtype, initializer=initializer, Q_norm=Q_norm, K_norm=K_norm, use_symmetric=use_symmetric)
        return mqa

    @staticmethod
    def compute_weights_softmax(causal_mask, head_dim, Q, K_chunked, n_repeats):
        K_chunked_repeat = nx.repeat(K_chunked, n_repeats, axis=1)  #(B, n_heads, n_chunk, 2W, head_dim)
        scores = Q @ K_chunked_repeat.transpose(0,1,2,4,3) #(B, n_heads, n_chunk, WQ, 2WK)
        scores = scores.astype(nx.float32) /  nx.sqrt(head_dim, dtype=nx.float32)
        scores = nx.where(causal_mask == 0, -1e9, scores)
        weights_softmax = nx.softmax(scores)
        return weights_softmax

    @staticmethod
    def _forward(x:nx.ArrayLike, causal_mask:nx.ArrayLike, configs:tuple[Any,...], params:tuple[Any,...], quantization, *, use_symmetric=False, recompute_activation=True):
        embed_dim, n_kv_heads, n_heads, n_rep, head_dim, chunk_size, freqs = configs
        Wqkv, Wo, Q_norm_gamma, K_norm_gamma = params

        wqkv_scale, wo_scale, wqkv_bias, wo_bias = quantization

        if wqkv_scale is not None:
            combined = nx.quantized_matmul(x, Wqkv, wqkv_scale,wqkv_bias, transpose=True, regular=use_symmetric)
        else:
            combined = x @ Wqkv.T

        Q = combined[..., :embed_dim] #shape: (B, T, D)
        K = combined[..., embed_dim: embed_dim + (n_kv_heads * head_dim)]  #shape: (B, T, n_kv_heads * H)
        V = combined[..., embed_dim + (n_kv_heads * head_dim):] #shape: (B, T, n_kv_heads * H)

        B, T, _ = x.shape
        chunk_size = min(chunk_size, T-1)
        Q = Q.reshape(B, T, n_heads, head_dim).transpose(0,2,1,3) #(B,n_heads,T, Dh)
        K = K.reshape(B, T, n_kv_heads, head_dim).transpose(0,2,1,3) #(B, n_kv_heads, T, Dh)
        V = V.reshape(B, T, n_kv_heads, head_dim).transpose(0,2,1,3) #(B, n_kv_heads, T, Dh)

        Q, Q_norm_caches = RMSNorm._forward(Q, Q_norm_gamma, 1e-5)
        K, K_norm_caches = RMSNorm._forward(K, K_norm_gamma, 1e-5)
        # print("Q prerope", Q.dtype)
        # print("K prerope", K.dtype)
        Q = rope_forward(Q, freqs)
        K = rope_forward(K, freqs)
        # print("Q postrope", Q.dtype)
        # print("K postrope", K.dtype)

        remainder = (chunk_size - (T % chunk_size)) % chunk_size
        pad = [(0,0), (0,0), (0, remainder), (0,0)]
        Q = nx.pad(Q, pad) # (B, n_heads, T + remainder, Dh)
        K = nx.pad(K, pad) # (B, n_kv_heads, T + remainder, Dh)
        V = nx.pad(V, pad) # (B, n_kv_heads, T + remainder, Dh)

        n_chunk = Q.shape[2] // chunk_size
        Q = Q.reshape(B, n_heads, n_chunk, chunk_size, head_dim)
        K = K.reshape(B, n_kv_heads, n_chunk, chunk_size, head_dim)
        V = V.reshape(B, n_kv_heads, n_chunk, chunk_size, head_dim)

        front = nx.zeros((B, n_kv_heads, 1, chunk_size, head_dim), Q.dtype)

        preprendix = nx.concatenate([front,K[:,:,:-1,:,:]], axis=2)
        K_chunked = nx.concatenate([preprendix, K], axis=3) #(B, n_kv_heads, n_chunk, 2W, head_dim)
        # print("kchunked", K_chunked.dtype)

        preprendix = nx.concatenate([front,V[:,:,:-1,:,:]], axis=2)
        V_chunked = nx.concatenate([preprendix, V], axis=3)  #(B, n_kv_heads, n_chunk, 2W, head_dim)
        # print("vchunked", V_chunked.dtype)

        Q = Q.astype(x.dtype)
        K_chunked = K_chunked.astype(x.dtype)
        V_chunked = V_chunked.astype(x.dtype)

        V_chunked_repeat = nx.repeat(V_chunked, n_rep, axis=1)  #(B, n_heads, n_chunk, 2W, head_dim)

        weights_softmax = AttentionChunked.compute_weights_softmax(causal_mask, head_dim, Q, K_chunked, n_rep)
        weights = weights_softmax.astype(x.dtype) #(B, n_heads, n_chunk, WQ, 2WK)

        output = weights @ V_chunked_repeat #(B, n_heads, n_chunk, chunk_size, Dh)
        output_unchunked = output.reshape(B,n_heads,-1,head_dim) #(B, n_heads, n_chunk * 2WQ, Dh)
        output_unchunked = output_unchunked[:,:,:T,:] #(B, n_heads, T, Dh)
        output_unchunked = output_unchunked.transpose(0,2,1,3).reshape(B,T,-1)
        # print("output", output.dtype)
        # print("weights", weights.dtype)
        # print("vchunkedrepeat", V_chunked_repeat.dtype)

        if wo_scale is not None:
            output_projected = nx.quantized_matmul(output_unchunked, Wo, wo_scale, wo_bias, regular=use_symmetric) #B,T,D #dtype
        else:
            output_projected = output_unchunked @ Wo

        if not recompute_activation:
            cache = (x, Q, K_chunked, V_chunked, Q_norm_caches,K_norm_caches, weights_softmax, output_unchunked)
        else:
            cache = (x, Q, K_chunked, V_chunked, Q_norm_caches,K_norm_caches, output_unchunked, causal_mask)

        # print("projected chunked", output_projected.dtype)
        # print("output unchunked chunked", output_unchunked.dtype)
        # print("Wo chunked", Wo.dtype)
        return output_projected, cache

    @staticmethod
    def _backward(gradient:nx.ArrayLike, caches:tuple[Any,...], attn_configs:tuple[Any,...], attn_params: tuple[Any,...], quantization, * , use_symmetric=False, recompute_activation=True) :#-> tuple[nx.ArrayLike,...]:
        embed_dim, n_kv_heads, n_heads, n_rep, head_dim, chunk_size, freqs = attn_configs

        if not recompute_activation:
            x, Q, K_chunked, V_chunked, Q_norm_caches,K_norm_caches, weights_softmax, output_unchunked = caches
        else:
            x, Q, K_chunked, V_chunked, Q_norm_caches,K_norm_caches, output_unchunked, causal_mask = caches
            weights_softmax = AttentionChunked.compute_weights_softmax(causal_mask, head_dim, Q, K_chunked, n_rep)
        # print("chunked x", x.dtype)
        # print("chunked gradient",gradient.dtype)
        Wqkv, Wo, Q_norm_gamma, K_norm_gamma = attn_params

        wqkv_scale, wo_scale, wqkv_bias, wo_bias = quantization

        if wqkv_scale is not None :
            Wqkv = nx.dequantize(Wqkv, wqkv_scale,wqkv_bias, x.dtype, regular=use_symmetric)

        B, T, D = x.shape
        chunk_size = min(chunk_size, T-1)
        remainder = (chunk_size - (T % chunk_size)) % chunk_size

        if wo_scale is not None:
            d_output_unchunked = nx.quantized_matmul(gradient, Wo, wo_scale, wo_bias, True, regular=use_symmetric)
        else:
            d_output_unchunked = gradient @ Wo.T #(B,T,D)

        d_output_unchunked = d_output_unchunked.reshape(B,T, n_heads, head_dim).transpose(0,2,1,3)

        pad = [(0,0),(0,0), (0, remainder), (0,0)]
        d_output_padded = nx.pad(d_output_unchunked, pad)#(B,n_heads, T+remainder,D)

        n_chunk = d_output_padded.shape[2] // chunk_size
        d_output_chunked = d_output_padded.reshape(B, n_heads, n_chunk, chunk_size, head_dim)

        d_output_chunked = d_output_chunked.reshape(B, n_kv_heads, n_rep,n_chunk, chunk_size, head_dim)

        d_weights = nx.einsum("bkrcwd,bkcxd->bkrcwx", d_output_chunked, V_chunked)

        # a = time.perf_counter()
        # d_chunked_V = nx.einsum("bkrcwx,bkrcwd->bkcxd", weights_softmax.astype(gradient.dtype).reshape(B,n_kv_heads,n_rep, n_chunk, chunk_size, 2*chunk_size), d_output_chunked)

        weights_softmax_6d = weights_softmax.astype(gradient.dtype).reshape(B, n_kv_heads,n_rep, n_chunk, chunk_size, 2*chunk_size)
        d_chunked_V = weights_softmax_6d.transpose(0,1,2,3,5,4) @ d_output_chunked
        d_chunked_V = nx.sum(d_chunked_V, 2)
        # nx.eval(d_chunked_V)
        # b = time.perf_counter()
        # print(f"{b-a:.5f}")

        d_scores = softmax_derivative(weights_softmax, d_weights.reshape(B, -1, n_chunk, chunk_size, 2*chunk_size).astype(nx.float32))  / nx.sqrt(head_dim, dtype=nx.float32) #(B, n_heads, n_chunk, chunk_size, 2W) #type:ignore
        d_scores = d_scores.astype(gradient.dtype)
        d_scores = d_scores.reshape(B,n_kv_heads,n_rep,n_chunk,chunk_size,2*chunk_size)

        dQ = nx.einsum("bkrcwx,bkcxd->bkrcwd", d_scores, K_chunked).reshape(B, n_heads, -1, head_dim) #B, n_heads, T + remainder, head_dim

        # a = time.perf_counter()
        # d_chunked_K = nx.einsum("bkrcwx,bkrcwd->bkcxd", d_scores, Q.reshape(B, n_kv_heads, n_rep, n_chunk, chunk_size, head_dim))

        Q = Q.reshape(B, n_kv_heads, n_rep, n_chunk, chunk_size, head_dim)
        d_chunked_K = d_scores.transpose(0,1,2,3,5,4) @ Q
        # print("d_chunked",d_chunked_K.dtype)
        d_chunked_K = nx.sum(d_chunked_K, 2)

        # nx.eval(d_chunked_K)
        # b = time.perf_counter()
        # print(f"{b-a:.5f}")

        dQ = dQ[:,:,:T,:] #B, n_heads, T, head_dim
        dQ = rope_inverse(dQ, freqs)
        dQ,Q_norm_d_gamma = RMSNorm._backward(dQ, Q_norm_caches, Q_norm_gamma)
        dQ = dQ.transpose(0, 2, 1, 3).reshape(B, T, embed_dim)

        dK_l = d_chunked_K[:, :, :, :chunk_size, :]
        dK_r = d_chunked_K[:, :, :, chunk_size:, :]
        dK_r[:,:,:-1,:,:] += dK_l[:,:,1:,:,:] #B, n_heads, n_chunk, chunk_size, Dh
        dK = dK_r.reshape(B, n_kv_heads, -1, head_dim)[:, :, :T, :] #B, n_kv_heads, T, Dh
        dK = rope_inverse(dK, freqs)
        dK,K_norm_d_gamma = RMSNorm._backward(dK, K_norm_caches, K_norm_gamma)
        dK = dK.transpose(0,2,1,3).reshape(B,T, head_dim*n_kv_heads) #B, T, D

        dV_l = d_chunked_V[:, :, :, :chunk_size, :]
        dV_r = d_chunked_V[:, :, :, chunk_size:, :]
        dV_r[:,:,-1:, :,:] += dV_l[:,:,:1,:,:] #B, n_heads, n_chunk, chunk_size, Dh
        dV = dV_r.reshape(B, n_kv_heads, -1, head_dim)[:, :, :T, :] #B, n_kv_heads, T, Dh
        dV = dV.transpose(0,2,1,3).reshape(B,T, head_dim*n_kv_heads)#B, T, D

        dQKV = nx.concatenate([dQ, dK,dV], axis=-1) #(B,T, D + 2 * (n_kv_heads * Dh))
        DQKV = dQKV.reshape(-1, embed_dim + 2 * (n_kv_heads * head_dim))

        X = x.reshape(-1, embed_dim)
        dWqkv = DQKV.T @ X
        # print("dwqkv",dWqkv.dtype)
        # print("X", X.dtype)

        H = output_unchunked.reshape(-1, embed_dim)
        G = gradient.reshape(-1, embed_dim)

        dWo = H.T @ G
        dx = dQKV @ Wqkv

        return dx,dWqkv,dWo, Q_norm_d_gamma, K_norm_d_gamma

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
        K, _ = RMSNorm._forward(K, self.K_norm.gamma, self.K_norm.epsilon)
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
        Q, _ = RMSNorm._forward(Q, self.Q_norm.gamma, self.Q_norm.epsilon)
        Q = rope_forward(Q, freqs, position)

        repeats_cached_k = nx.repeat(cached_k, self.n_rep, axis=1 )
        repeats_cached_v = nx.repeat(cached_v, self.n_rep, axis=1 )

        scores = (Q @ repeats_cached_k.transpose(0,1,3,2)).astype(nx.float32) / nx.float_32(nx.sqrt(self.head_dim))
        weights = nx.softmax(scores)
        weights = weights.astype(x.dtype)
        output = weights @ repeats_cached_v
        output_concat = output.transpose(0, 2, 1, 3).reshape(B, T, self.embed_dim)

        if wo_scale is not None:
            output_projected = nx.quantized_matmul(output_concat, self.Wo, wo_scale, wo_bias, regular=use_symmetric) #BTD
        else:
            output_projected = output_concat @ self.Wo

        return output_projected, cached_k, cached_v

    def compute_mask(self):
        chunk_size = self.chunk_size
        row_idx = nx.ones((chunk_size,chunk_size))
        column_idx = nx.ones((chunk_size,chunk_size))
        trilled = nx.tril(column_idx)
        return nx.concatenate([row_idx, trilled], axis=1)
        


    @classmethod
    def from_weight(cls, configs, weights, quants,attn_QK_gamma, dtype) -> "AttentionChunked":
        embed_dim, n_kv_heads, n_heads, _, _,chunk_size = configs
        wqkv, wo = weights

        Q_norm_gamma,Q_norm_configs, K_norm_gamma, K_norm_configs = attn_QK_gamma
        Q_norm = RMSNorm.from_weight(Q_norm_configs, Q_norm_gamma)
        K_norm = RMSNorm.from_weight(K_norm_configs, K_norm_gamma)


        attn = cls(embed_dim, n_heads, Q_norm, K_norm, n_kv_heads,chunk_size,dtype, init=False)
        attn.Wqkv = wqkv
        attn.Wo = wo

        if quants is not None:
            scales, biases = quants
            attn.scales = scales
            attn.biases = biases

        return attn

    def copy(self):
        Q_norm_copy = self.Q_norm.copy()
        K_norm_copy = self.K_norm.copy()
        attn_copy = AttentionChunked(self.embed_dim, self.n_heads, Q_norm_copy, K_norm_copy, self.n_kv_heads, self.chunk_size, self.dtype, quantized=self.quantized, init=False)
        attn_copy.Wqkv = nx.copy(self.Wqkv)
        attn_copy.Wo = nx.copy(self.Wo)
        if self.quantized:
            attn_copy.scales = (nx.copy(self.scales[0]), nx.copy(self.scales[1]))
            attn_copy.biases = (nx.copy(self.biases[0]), nx.copy(self.biases[1]))

        return attn_copy