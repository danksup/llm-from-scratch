import engine.backend as nx
from engine.activations import softmax, softmax_derivative
from engine.rope import rope_forward, rope_inverse
from typing import Any, Callable
import engine.initializers as initializer
from engine.rope import precompute_freqs

class AttentionSWA:
    def __init__(self,embed_dim:int, n_heads:int, n_kv_heads:int=-1, W=8, dtype:Any=nx.float16, initializer:Callable=initializer.glorot_uniform, quantized:bool=False) -> None:
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

        self.W = W
        self.dtype = dtype

        self.n_rep = self.n_heads // self.n_kv_heads

        self.freqs = precompute_freqs(self.head_dim, 16384)

        self.configs = self.embed_dim, self.n_kv_heads, self.n_heads, self.n_rep, head_dim, self.W, self.freqs

        wqkv_shape = embed_dim + 2 * n_kv_heads * self.head_dim, embed_dim
        self.Wqkv = initializer(wqkv_shape, dtype=dtype)
        assert nx.isfinite(self.Wqkv).all(), f"non-finite detected when initializing attentipn.Wqkv."

        wo_shape = embed_dim,embed_dim
        self.Wo = initializer(wo_shape, dtype=dtype)
        assert nx.isfinite(self.Wo).all(), f"non-finite detected when initializing attentipn.Wo."

        self.scales = (None, None)
        self.biases = (None,None)
        self.quantized = quantized
        if quantized:
            self.Wqkv, wqkv_scale, wqkv_bias = nx.quantize(self.Wqkv)
            self.Wo, wo_scale, wo_bias = nx.quantize(self.Wo)
            self.scales = (wqkv_scale, wo_scale)
            self.biases = (wqkv_bias, wo_bias)


        self.dWqkv = None
        self.dWo = None

    @staticmethod
    def self_type() -> str:
        return "swa"

    @classmethod
    def multihead(cls,embed_dim, n_heads, W, dtype, initializer):
        mha = cls(embed_dim, n_heads=n_heads, n_kv_heads=n_heads, W=W, dtype=dtype, initializer=initializer)
        return mha
    @classmethod
    def multiquery(cls,embed_dim, n_heads, W, dtype, initializer):
        mqa = cls(embed_dim, n_heads=n_heads, n_kv_heads=1, W=W, dtype=dtype, initializer=initializer)
        return mqa

    @staticmethod
    def _forward(x:nx.ArrayLike, causal_mask:nx.ArrayLike, configs:tuple[Any,...], params:tuple[Any,...], quantization):
        embed_dim, n_kv_heads, n_heads, n_rep, head_dim, W, freqs = configs
        Wqkv, Wo = params

        wqkv_scale, wo_scale, wqkv_bias, wo_bias = quantization

        if wqkv_scale is not None:
            combined = nx.quantized_matmul(x, Wqkv, wqkv_scale,wqkv_bias, transpose=True)
        else:
            combined = x @ Wqkv.T

        Q = combined[..., :embed_dim] #shape: (B, T, D)
        K = combined[..., embed_dim: embed_dim + (n_kv_heads * head_dim)]  #shape: (B, T, n_kv_heads * H)
        V = combined[..., embed_dim + (n_kv_heads * head_dim):] #shape: (B, T, n_kv_heads * H)

        B, T, _ = x.shape
        W = min(W, T-1)
        Q = Q.reshape(B, T, n_heads, head_dim).transpose(0,2,1,3) #(B,n_heads,T, Dh)
        K = K.reshape(B, T, n_kv_heads, head_dim).transpose(0,2,1,3) #(B, n_kv_heads, T, Dh)
        V = V.reshape(B, T, n_kv_heads, head_dim).transpose(0,2,1,3) #(B, n_kv_heads, T, Dh)

        Q = rope_forward(Q, freqs)
        K = rope_forward(K, freqs)

        pad = [(0,0), (0,0),(W,0), (0,0)]
        P = T + W
        stride = (head_dim * n_kv_heads * P, P * head_dim, head_dim, head_dim, 1)
        padded_K = nx.pad(K, pad, constant_value=0)  #(B, n_kv_head, T+W, Dh)
        padded_V = nx.pad(V, pad, constant_value=0) #(B,n_kv_head,T+W, Dh)
        shape = (padded_K.shape[0], padded_K.shape[1], T, W + 1, head_dim) #(B, n_kv_head, T, W+1, Dh)

        windows_K = nx.as_strided(padded_K, shape=shape,strides=stride) #shape=shape dtype
        windows_V = nx.as_strided(padded_V, shape=shape,strides=stride) #shape=shape dtype

        Q = Q.reshape(B, n_kv_heads, n_rep, T, head_dim)

        Q_6d = Q[:,:,:,:,None,:] #(B, n_kv_heads, n_rep, T,1, Dh)
        windows_K_6d = windows_K[:,:,None,:,:,:] #(B, n_kv_head, 1, T, W+1, Dh)
        scores = Q_6d @ windows_K_6d.transpose(0,1,2,3,5,4) #B, n_kv_heads, n_rep, T, 1, W+1 #dtype

        scores = scores[:,:,:,:,0,:].reshape(B, -1, T, W+1)
        scores = scores.astype(nx.float32) /  nx.sqrt(head_dim, dtype=nx.float32)
        scores = nx.where(causal_mask, -nx.inf, scores)
        weights_softmax = softmax(scores) #(B, n_heads, T, W+1) #fp32

        weights = weights_softmax.astype(x.dtype)
        weights = weights.reshape(B, n_kv_heads, n_rep, T, W+1)

        weights_6d = weights[:,:,:,:,None,:]     #(B, n_kv_head, n_rep, T, 1, W+1)
        windows_V_6d = windows_V[:,:,None,:,:,:] #(B, n_kv_head, 1, T, W+1, Dh)

        output = weights_6d @ windows_V_6d #(B,K,R,T,1,D)

        output = output[:,:,:,:,0,:]
        output_concat = output.transpose(0, 3, 1, 2, 4).reshape(B, T, embed_dim)
        output_projected = nx.quantized_matmul(output_concat, Wo, wo_scale, wo_bias) #B,T,D #dtype

        cache = (x, Q, windows_K, windows_V, weights_softmax, output_concat)
        return output_projected, cache

    @staticmethod
    def _backward(gradient:nx.ArrayLike, caches:tuple[Any,...], attn_configs:tuple[Any,...], attn_params: tuple[Any,...], quantization) :#-> tuple[nx.ArrayLike,...]:
        x, Q, windows_K, windows_V, weights_softmax, output_concat = caches
        embed_dim, n_kv_heads, n_heads, n_rep, head_dim, W, freqs = attn_configs
        Wqkv, Wo = attn_params

        wqkv_scale, wo_scale, wqkv_bias, wo_bias = quantization

        Wqkv = nx.dequantize(Wqkv, wqkv_scale,wqkv_bias, x.dtype)
        Wo = nx.dequantize(Wo, wo_scale,wo_bias, x.dtype)

        B, T, D = x.shape
        W = min(W, T-1)

        weights_softmax = weights_softmax.reshape(B, n_kv_heads, n_rep, T, W+1)
        weights = weights_softmax.astype(x.dtype)
        d_output_concat = nx.einsum("btd,fd->btf",gradient, Wo) #(B,T,D)

        d_output = d_output_concat.reshape(B, T, n_heads, head_dim).transpose(0, 2, 1, 3) #(B, n_heads, T,  Dh)
        d_output_split = d_output.reshape(B, n_kv_heads,n_rep,T, head_dim)

        d_output_split_6d = d_output_split[:,:,:,:,None,:] #B, n_kv_heads, n_rep, T, 1, Dh
        windows_V_6d = windows_V[:,:,None,:,:,:] #(B, n_kv_head, 1, T, W+1, Dh)
        d_weights = d_output_split_6d @ windows_V_6d.transpose(0,1,2,3,5,4) #B, n_kv_heads, n_rep,T, 1, W+1

        d_windows_V = nx.einsum("bkrtw,bkrtd->bktwd", weights, d_output_split) #(B, n_kv_head, T , W+1, Dh)

        d_weights = d_weights[:,:,:,:,0,:].astype(nx.float32)
        d_scores = softmax_derivative(weights_softmax, d_weights) / nx.sqrt(head_dim, dtype=nx.float32) #(B, n_kv_heads, n_rep, T, W+1)
        d_scores = d_scores.astype(x.dtype)

        del d_output_split, d_output_split_6d, d_weights, windows_V_6d

        d_scores_6d = d_scores[:,:,:,:,None,:] #(B, n_kv_heads, n_rep, T, 1,W+1)
        windows_K_6d = windows_K[:,:,None,:,:,:] #(B, n_kv_head, 1, T, W+1, Dh)
        dQ = d_scores_6d @ windows_K_6d

        dQ = dQ.reshape(B, -1, T, head_dim)

        d_windows_K = nx.einsum("bkrtw,bkrtd->bktwd", d_scores, Q) #(B,n_kv_heads,T, W+1, Dh)

        del Q, d_scores, d_scores_6d, windows_K_6d,  windows_K

        d_padded_K = nx.zeros((B, n_kv_heads, T+W, head_dim), dtype=d_windows_K.dtype)
        d_padded_V = nx.zeros((B, n_kv_heads, T+W, head_dim), dtype=d_windows_V.dtype)
        for slot in range(W + 1):
            d_padded_K[:, :, slot:slot + T, :] += d_windows_K[:, :, :, slot, :]
            d_padded_V[:, :, slot:slot + T, :] += d_windows_V[:, :, :, slot, :]

        dK = d_padded_K[:, :, W:, :]
        dV = d_padded_V[:, :, W:, :]

        del d_windows_K, d_windows_V, d_padded_K, d_padded_V

        dQ = rope_inverse(dQ, freqs) #grad dtype
        dK = rope_inverse(dK, freqs) #grad dtype

        dQ = dQ.transpose(0, 2, 1, 3).reshape(B, T, embed_dim)
        dK = dK.transpose(0, 2, 1, 3).reshape(B, T, n_kv_heads * head_dim)
        dV = dV.transpose(0, 2, 1, 3).reshape(B, T,  n_kv_heads * head_dim)

        dQKV = nx.concatenate([dQ, dK,dV], axis=-1) #(B,T, D + 2 * (n_kv_heads * Dh))
        DQKV = dQKV.reshape(-1, embed_dim + 2 * (n_kv_heads * head_dim))
        del dQ, dK, dV

        X = x.reshape(-1, embed_dim)
        dWqkv = DQKV.T @ X

        H = output_concat.reshape(-1, embed_dim)
        G = gradient.reshape(-1, embed_dim)

        dWo = H.T @ G
        dx = dQKV @ Wqkv

        del x, output_concat, freqs, Wqkv, Wo
        return dx,dWqkv,dWo

    #TODO:compiled, dtype fix, quantization
    def inference_forward(self, x, max_cache_len, freqs, quantization, cached_k=None, cached_v=None, position = 0):
        scale = nx.float_32(nx.sqrt(self.head_dim))

        wqkv_scale, wo_scale, wqkv_bias, wo_bias = quantization #type:ignore

        if wqkv_scale is not None:
            combined = nx.quantized_matmul(x, self.Wqkv, wqkv_scale, wqkv_bias, transpose=True)
        else:
            combined =  x @ self.Wqkv.T  # dtype
        B, T, _ = x.shape

        K = combined[..., self.embed_dim: self.embed_dim + (self.n_kv_heads * self.head_dim)]

        K = K.reshape(B, T, self.n_kv_heads, self.head_dim).transpose(0,2,1,3)
        K = rope_forward(K, freqs, position)
        K = K.astype(nx.float32)
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

        Q = Q.astype(nx.float32)

        repeats_cached_k = nx.repeat(cached_k, self.n_rep, axis=1 )
        repeats_cached_v = nx.repeat(cached_v, self.n_rep, axis=1 )

        scores = (Q @ repeats_cached_k.transpose(0,1,3,2)) / scale
        weights = softmax(scores)
        output = weights @ repeats_cached_v
        output_concat = output.transpose(0, 2, 1, 3).reshape(B, T, self.embed_dim)
        output_projected = nx.quantized_matmul(output_concat, self.Wo, wo_scale, wo_bias) #BTD

        return output_projected, cached_k, cached_v

    @staticmethod
    def compute_mask(W, T):
        window_idx = nx.arange(W + 1).reshape((1, W + 1))
        time_idx = nx.arange(T).reshape((T, 1))
        padded_position = time_idx + window_idx
        return padded_position < W

    def to_dict(self) -> dict:
        '''serialize into dict with weights turned into list'''
        attn_dict = {
            "embed_dim":self.embed_dim,
            "n_heads":self.n_heads,
            "n_kv_heads":self.n_kv_heads,
            "dtype": nx.dtype_to_srt[self.dtype],
            "W":self.W,
            "Wqkv":self.Wqkv.tolist(),
            "Wo":self.Wo.tolist(),
            "quantized":self.quantized
        }

        if self.quantized:
             attn_dict["scales"] = (self.scales[0].tolist(), self.scales[1].tolist()) #type:ignore
             attn_dict["biases"] = (self.biases[0].tolist(), self.biases[1].tolist()) #type:ignore
        else:
             attn_dict["scales"] = self.scales
             attn_dict["biases"] = self.biases
        return attn_dict



    @classmethod
    def from_dict(cls,thing) -> "AttentionSWA":
        """deserialize"""
        embed_dim = thing["embed_dim"]
        n_kv_heads = thing["n_kv_heads"]
        n_heads = thing["n_heads"]
        W = thing["W"]
        Wqkv = thing["Wqkv"]
        Wo = thing["Wo"]
        dtype = nx.str_to_dtype[thing["dtype"]]
        is_quantized = thing["quantized"]
        scales = thing["scales"]
        biases = thing["biases"]

        attention = cls(embed_dim=embed_dim, n_heads=n_heads, n_kv_heads=n_kv_heads, W=W, dtype=dtype, quantized=is_quantized)

        if is_quantized:
            if nx.backend == "MLX":
                attention.Wqkv = nx.array(Wqkv, dtype=nx.uint32)
                attention.Wo = nx.array(Wo, dtype=nx.uint32)
            else:
                attention.Wqkv = nx.array(Wqkv, dtype=nx.int8)
                attention.Wo = nx.array(Wo, dtype=nx.int8)

            attention.scales = (nx.array(scales[0], dtype=dtype), nx.array(scales[1], dtype=dtype))
            attention.biases = (nx.array(biases[0], dtype=dtype), nx.array(biases[1], dtype=dtype))
        else:
            attention.Wqkv = nx.array(Wqkv, dtype=dtype)
            attention.Wo = nx.array(Wo, dtype=dtype)

        return attention
