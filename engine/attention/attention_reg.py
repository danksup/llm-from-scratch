import engine.backend as nx
from engine.activations import softmax, softmax_derivative
from engine.rope import rope_forward, rope_inverse
from typing import Any, Callable
import engine.initializers as initializer
from engine.rope import precompute_freqs

class AttentionFull:
    def __init__(self,embed_dim:int, n_heads:int, n_kv_heads:int=-1,  dtype:Any=nx.float16,  initializer:Callable=initializer.glorot_uniform) -> None:
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

        wqkv_shape = embed_dim + 2 * n_kv_heads * self.head_dim, embed_dim
        self.Wqkv = initializer(wqkv_shape, dtype=dtype)
        assert nx.isfinite(self.Wqkv).all(), f"non-finite detected when initializing attentipn.Wqkv."

        wo_shape = embed_dim,embed_dim
        self.Wo = initializer(wo_shape, dtype=dtype)
        assert nx.isfinite(self.Wo).all(), f"non-finite detected when initializing attentipn.Wo."

        self.dWqkv = None
        self.dWo = None


    @staticmethod
    def self_type() -> str:
        return "full"

    @classmethod
    def multihead(cls, embed_dim, n_heads, dtype, initializer):
        mha = cls(embed_dim, n_heads=n_heads, n_kv_heads=n_heads, dtype=dtype, initializer=initializer)
        return mha

    @classmethod
    def multiquery(cls, embed_dim, n_heads, dtype, initializer):
        mqa = cls(embed_dim, n_heads=n_heads, n_kv_heads=1, dtype=dtype, initializer=initializer)
        return mqa

    @staticmethod
    def _forward(x:nx.ArrayLike, causal_mask:nx.ArrayLike,  attn_configs:tuple[Any,...], attn_params: tuple[Any,...]) -> tuple[nx.ArrayLike, tuple[nx.ArrayLike,...]]:
        #fp_16_x shape = (B,T,D)
        #Wqkv.T (D, D + 2 * n_kv_heads * H)
        #combined (B, T, D + 2 * n_kv_heads * H)
        embed_dim, n_kv_heads, n_heads, n_rep, head_dim, freqs = attn_configs
        Wqkv, Wo = attn_params

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
        # print("qrope", Q.dtype)

        Q = Q.reshape(B,n_kv_heads, n_rep, T, head_dim)
        scores = nx.einsum("bkrQh,bkKh->bkrQK",Q, K) #(B, n_kv_heads, n_rep, Tq, Tk)
        # print("scores pre", scores.dtype)
        scores = scores.astype(nx.float32) / nx.sqrt(head_dim, dtype=nx.float32) #type:ignore
        scores = nx.where(causal_mask, -1e9, scores)
        # print("scores post", scores.dtype)
        weights = softmax(scores)
        weights = weights.astype(x.dtype)

        output = nx.einsum("bkrQK,bkKh->bkrQh",weights, V) #(B, n_kv_heads, n_rep, Tq, Dh)
        
        output = output.reshape(B, -1, T, head_dim)
        output_concat = output.transpose(0, 2, 1, 3).reshape(B, T, embed_dim)
        output_projected = output_concat @ Wo #BTD
        # print("proj", output_projected.dtype)
        cache =  (x, Q, K, V, weights, output_concat)
        return output_projected, cache
    
    @staticmethod
    def _backward(gradient:nx.ArrayLike, caches:tuple[Any,...], attn_configs:tuple[Any,...], attn_params: tuple[Any,...]) -> tuple[nx.ArrayLike,...]:
        x, Q, K, V, weights, output_concat = caches
        embed_dim, n_kv_heads, n_heads, n_rep, head_dim, freqs = attn_configs
        Wqkv, Wo = attn_params

        B, T, _ = x.shape
        d_output_concat = gradient @ Wo.T #B,T,D
        d_output = d_output_concat.reshape(B, T, n_heads, head_dim).transpose(0, 2, 1, 3)
        d_output = d_output.reshape(B, n_kv_heads, n_rep, T, head_dim)

        dweights = nx.einsum("bkrQh,bkKh->bkrQK", d_output, V) #(B, n_kv_heads, n_rep, Tq, Tk)

        dV = nx.einsum("bkrQK,bkrQh->bkKh", weights, d_output)

        dscores = softmax_derivative(weights.astype(nx.float32), dweights.astype(nx.float32)) / nx.sqrt(head_dim, dtype=nx.float32) #type:ignore
        dscores = dscores.astype(gradient.dtype) #(B, n_kv_heads, n_rep, Tq, Tk)

        dQ = nx.einsum("bkrQK,bkKh->bkrQh", dscores, K) #(B, nkv, n_rep, Tq, Dh)

        dK = nx.einsum("bkrQK,bkrQh->bkKh",dscores, Q) #(B, nkv, Tk, Dh)
        
        dQ = dQ.reshape(B, -1, T, head_dim)
        dQ = rope_inverse(dQ, freqs)
        dK = rope_inverse(dK, freqs)

        dQ = dQ.transpose(0, 2, 1, 3).reshape(B, T, embed_dim)
        dK = dK.transpose(0, 2, 1, 3).reshape(B, T, n_kv_heads * head_dim)
        dV = dV.transpose(0, 2, 1, 3).reshape(B, T,  n_kv_heads * head_dim)

        dQKV = nx.concatenate([dQ, dK,dV], axis=-1)
        DQKV = dQKV.reshape(-1, embed_dim + 2 * (n_kv_heads * head_dim))

        X = x.reshape(-1, embed_dim)
        dWqkv = DQKV.T @ X

        H = output_concat.reshape(-1, embed_dim)
        G = gradient.reshape(-1, embed_dim)

        dWo = H.T @ G
        dx = dQKV @ Wqkv

        # print("dx", dx.dtype)

        return dx,dWqkv,dWo
        
    def inference_forward(self, x, max_cache_len, freqs, cached_k=None, cached_v=None, position = 0):
        scale = nx.float_32(nx.sqrt(self.head_dim))
        combined = x @ self.Wqkv.T
        B, T, _ = x.shape    

        K = combined[..., self.embed_dim: self.embed_dim + (self.n_kv_heads * self.head_dim)] 

        K = K.reshape(B, T, self.n_kv_heads, self.head_dim).transpose(0,2,1,3)
        K = rope_forward(K, freqs, position) 
        K = K.astype(nx.float32)
        if cached_k is not None :
            cached_k = nx.concatenate([cached_k, K], axis = 2)
        else:
            cached_k = K

        # print(max(abs(K - cached_k[:, :, -1:, :])))
       
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
        output_projected = output_concat @ self.Wo
        
        return output_projected, cached_k, cached_v
    
    @staticmethod
    def compute_mask(T):        
        return nx.triu(nx.ones((T, T), dtype=nx.bool_), k=1)
    
    def to_dict(self) -> dict:
        return {
            "embed_dim":self.embed_dim,
            "n_heads":self.n_heads,
            "n_kv_heads":self.n_kv_heads,
            "Wqkv":self.Wqkv.tolist(),
            "Wo":self.Wo.tolist(),
            "dtype":nx.dtype_to_srt[self.dtype]
        }
    
    @classmethod
    def from_dict(cls,thing) -> "AttentionFull":
        embed_dim = thing["embed_dim"]
        n_kv_heads = thing["n_kv_heads"]
        n_heads = thing["n_heads"]
        Wqkv = thing["Wqkv"]
        Wo = thing["Wo"]
        dtype = nx.str_to_dtype[thing["dtype"]]

        attention = cls(embed_dim,n_heads, n_kv_heads)
        attention.Wqkv = nx.array(Wqkv, dtype=dtype)
        attention.Wo = nx.array(Wo, dtype=dtype)

        return attention
    
        