import engine.backend as nx
from engine.activations import softmax, softmax_derivative
from engine.rope import rope_forward, rope_inverse
from typing import Any, Callable
import engine.initializers as initializer
import time

class AttentionLayerSWA:
    def __init__(self,embed_dim:int, n_heads:int, n_kv_heads:int=-1, W=8, dtype:Any=nx.float16, initializer:Callable=initializer.glorot_uniform) -> None:
        self.n_kv_heads = n_kv_heads

        if n_kv_heads < 0:
            n_kv_heads = n_heads
        
        self.n_kv_heads = n_kv_heads
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        assert embed_dim % n_heads == 0
        assert n_heads % n_kv_heads == 0, "cant have more kv heads than query heads."
        self.head_dim = embed_dim // n_heads
        self.W = W
        self.dtype = dtype

        self.n_rep = self.n_heads // self.n_kv_heads 

        wqkv_shape = embed_dim + 2 * n_kv_heads * self.head_dim, embed_dim
        self.Wqkv = initializer(wqkv_shape, dtype=dtype)

        wo_shape = embed_dim,embed_dim
        self.Wo = initializer(wo_shape, dtype=dtype)
        
        self.dWqkv = None
        self.dWo = None

    @staticmethod
    def _forward(x:nx.ArrayLike, causal_mask:nx.ArrayLike, embed_dim:int, n_kv_heads:int, n_heads:int, n_rep:int, head_dim:int, W, freqs:Any, Wqkv:nx.ArrayLike, Wo:nx.ArrayLike):
        combined = x @ Wqkv.T 

        Q = combined[..., :embed_dim] #shape: (B, T, D)
        K = combined[..., embed_dim: embed_dim + (n_kv_heads * head_dim)]  #shape: (B, T, n_kv_heads * H)
        V = combined[..., embed_dim + (n_kv_heads * head_dim):] #shape: (B, T, n_kv_heads * H)
        
        B, T, _ = x.shape
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

        # start = time.perf_counter()
        Q_6d = Q[:,:,:,:,None,:] #(B, n_kv_heads, n_rep, T,1, Dh)
        windows_K_6d = windows_K[:,:,None,:,:,:] #(B, n_kv_head, 1, T, W+1, Dh)
        scores = Q_6d @ windows_K_6d.transpose(0,1,2,3,5,4) #B, n_kv_heads, n_rep, T, 1, W+1 #dtype

        # nx.eval(scores)  
        # end = time.perf_counter()
        # print(f"scores {end-start:.5f}")

        # start = time.perf_counter()
        # print("swa scores unscaled",scores.dtype)
        scores = scores[:,:,:,:,0,:].reshape(B, -1, T, W+1)
        scores = scores.astype(nx.float32) /  nx.sqrt(head_dim, dtype=nx.float32)
        scores = nx.where(causal_mask, -nx.inf, scores)
        weights_softmax = softmax(scores) #(B, n_heads, T, W+1) #fp32
        # print("weights softmax", weights.dtype)

        weights = weights_softmax.astype(x.dtype)
        weights = weights.reshape(B, n_kv_heads, n_rep, T, W+1)
        # print("weights after cast", weights.dtype)

        # nx.eval(weights)  
        # end = time.perf_counter()
        # print(f"scores {end-start:.5f}")

        # start = time.perf_counter()
        weights_6d = weights[:,:,:,:,None,:]     #(B, n_kv_head, n_rep, T, 1, W+1)
        windows_V_6d = windows_V[:,:,None,:,:,:] #(B, n_kv_head, 1, T, W+1, Dh)
        # output = nx.einsum("bkrtw,bktwd->bkrtd", weights, windows_V) #(B, n_kv_heads, n_rep, T, Dh)
        output = weights_6d @ windows_V_6d #(B,K,R,T,1,D)
        # nx.eval(output)  
        # end = time.perf_counter()
        # print(f"output {end-start:.5f}")
        
        output = output[:,:,:,:,0,:]
        output_concat = output.transpose(0, 3, 1, 2, 4).reshape(B, T, embed_dim)
        output_projected = output_concat @ Wo #B,T,D #dtype
        # print("output projected" , output_projected.dtype)
        # print("x forward", x.dtype)
        # print("wink forward", windows_K.dtype)
        # print("winv forward", windows_V.dtype)
        # print("weights forward", weights.dtype)
        cache = (x, Q, windows_K, windows_V, weights_softmax, output_concat)
        return output_projected, cache
    
    @staticmethod
    def _backward(gradient:nx.ArrayLike, caches:tuple[Any,...], attn_params: tuple[Any,...]) :#-> tuple[nx.ArrayLike,...]:
        x, Q, windows_K, windows_V, weights_softmax, output_concat = caches
        n_heads, head_dim, embed_dim, n_kv_heads, n_rep, W, Wo, freqs, Wqkv = attn_params
        B, T, D = x.shape
        weights_softmax = weights_softmax.reshape(B, n_kv_heads, n_rep, T, W+1)
        weights = weights_softmax.astype(x.dtype)
        d_output_concat = nx.einsum("btd,fd->btf",gradient, Wo) #(B,T,D)
        # print("doutput", d_output_concat.dtype)
        # print("wo", Wo.dtype)
        d_output = d_output_concat.reshape(B, T, n_heads, head_dim).transpose(0, 2, 1, 3) #(B, n_heads, T,  Dh)
        d_output_split = d_output.reshape(B, n_kv_heads,n_rep,T, head_dim)
        # print("d_output_split",d_output_split.dtype)
        # start = time.perf_counter()
        d_output_split_6d = d_output_split[:,:,:,:,None,:] #B, n_kv_heads, n_rep, T, 1, Dh
        windows_V_6d = windows_V[:,:,None,:,:,:] #(B, n_kv_head, 1, T, W+1, Dh)
        d_weights = d_output_split_6d @ windows_V_6d.transpose(0,1,2,3,5,4) #B, n_kv_heads, n_rep,T, 1, W+1
        
        # print("dweights",d_weights.dtype)
        # print("winv",windows_V.dtype)


        # nx.eval(d_weights)  
        # end = time.perf_counter()
        # print(f"d_weights {end-start:.5f}")

        # start = time.perf_counter()
        d_windows_V = nx.einsum("bkrtw,bkrtd->bktwd", weights, d_output_split) #(B, n_kv_head, T , W+1, Dh)
        # nx.eval(d_windows_V)  
        # end = time.perf_counter()
        # print(f"d_windows_V {end-start:.5f}")

        d_weights = d_weights[:,:,:,:,0,:].astype(nx.float32)
        d_scores = softmax_derivative(weights_softmax, d_weights) / nx.sqrt(head_dim, dtype=nx.float32) #(B, n_kv_heads, n_rep, T, W+1)
        d_scores = d_scores.astype(x.dtype)

        # start = time.perf_counter()
        d_scores_6d = d_scores[:,:,:,:,None,:] #(B, n_kv_heads, n_rep, T, 1,W+1)
        windows_K_6d = windows_K[:,:,None,:,:,:] #(B, n_kv_head, 1, T, W+1, Dh)
        dQ = d_scores_6d @ windows_K_6d
        # nx.eval(dQ)  
        # end = time.perf_counter()
        # print(f"dQ {end-start:.5f}")

        dQ = dQ.reshape(B, -1, T, head_dim)

        # start = time.perf_counter()
        d_windows_K = nx.einsum("bkrtw,bkrtd->bktwd", d_scores, Q) #(B,n_kv_heads,T, W+1, Dh)
        # nx.eval(d_windows_K)  
        # end = time.perf_counter()
        # print(f"d_windows_K {end-start:.5f}")

        # start = time.perf_counter()
        d_padded_K = nx.zeros((B, n_kv_heads, T+W, head_dim), dtype=d_windows_K.dtype)
        d_padded_V = nx.zeros((B, n_kv_heads, T+W, head_dim), dtype=d_windows_V.dtype)
        for slot in range(W + 1):
            d_padded_K[:, :, slot:slot + T, :] += d_windows_K[:, :, :, slot, :]
            d_padded_V[:, :, slot:slot + T, :] += d_windows_V[:, :, :, slot, :]
        # nx.eval(d_padded_K, d_padded_V)  
        # end = time.perf_counter()
        # print(f"padded loop {end-start:.5f}")

        dK = d_padded_K[:, :, W:, :]
        dV = d_padded_V[:, :, W:, :]

        dQ = rope_inverse(dQ, freqs) #grad dtype
        dK = rope_inverse(dK, freqs) #grad dtype

        dQ = dQ.transpose(0, 2, 1, 3).reshape(B, T, embed_dim)
        dK = dK.transpose(0, 2, 1, 3).reshape(B, T, n_kv_heads * head_dim)
        dV = dV.transpose(0, 2, 1, 3).reshape(B, T,  n_kv_heads * head_dim)

        # print("dq after rope", dQ.dtype)
        # print("dv", dV.dtype)
        # print("dk", dK.dtype)


        dQKV = nx.concatenate([dQ, dK,dV], axis=-1) #(B,T, D + 2 * (n_kv_heads * Dh))
        DQKV = dQKV.reshape(-1, embed_dim + 2 * (n_kv_heads * head_dim))

        X = x.reshape(-1, embed_dim)
        dWqkv = DQKV.T @ X
        # print("dwqkv",dWqkv.dtype)
        # print("x", X.dtype)
        # dWqkv = dWqkv.astype(nx.float32)

        H = output_concat.reshape(-1, embed_dim)
        G = gradient.reshape(-1, embed_dim)

        dWo = H.T @ G
        dx = dQKV @ Wqkv
        # print("dwo", dWo.dtype)
        # print("dx", dx.dtype)
        return dx,dWqkv,dWo
    
    def inference_forward(self, x, max_cache_len, freqs, cached_k=None, cached_v=None, position = 0):
        scale = nx.float_32(nx.sqrt(self.head_dim))
        combined = x @ self.Wqkv.T
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
    def compute_mask(W, T):
        window_idx = nx.arange(W + 1).reshape((1, W + 1))
        time_idx = nx.arange(T).reshape((T, 1))
        padded_position = time_idx + window_idx
        return padded_position < W
    
    def to_dict(self) -> dict:
        '''serialize into dict with weights turned into list'''
        return {
            "embed_dim":self.embed_dim,
            "n_heads":self.n_heads,
            "n_kv_heads":self.n_kv_heads,
            "dtype": nx.dtype_to_srt[self.dtype],
            "W":self.W,
            "Wqkv":self.Wqkv.tolist(),
            "Wo":self.Wo.tolist(),
        }
    
    @classmethod
    def from_dict(cls,thing) -> "AttentionLayerSWA":
        """deserialize"""
        embed_dim = thing["embed_dim"]
        n_kv_heads = thing["n_kv_heads"]
        n_heads = thing["n_heads"]
        W = thing["W"]
        Wqkv = thing["Wqkv"]
        Wo = thing["Wo"]
        dtype = nx.str_to_dtype[thing["dtype"]]

        attention = cls(embed_dim,n_heads, n_kv_heads, W, dtype)
        attention.Wqkv = nx.array(Wqkv, dtype=dtype)
        attention.Wo = nx.array(Wo, dtype=dtype)

        return attention