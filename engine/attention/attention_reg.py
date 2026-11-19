import engine.backend as nx
from engine.rope import rope_forward, rope_inverse
from typing import Any, Callable
import engine.initializers as initializer
from engine.rope import precompute_freqs
from engine.rmsnorm import RMSNorm

class AttentionFull:
    def __init__(self,embed_dim:int, n_heads:int,Q_norm:RMSNorm, K_norm:RMSNorm,  n_kv_heads:int=-1,theta=10000,s=1.0, ori_T=None, alpha=1.0, beta=32.0, dtype:Any=nx.float16,  initializer:Callable=initializer.glorot_uniform, quantized:bool=False, *, init=True) -> None:
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

        self.theta = theta
        self.s = s
        self.ori_T = ori_T
        self.alpha = alpha
        self.beta = beta
        self.freqs = precompute_freqs(self.head_dim, arg1=32768, freq_multiplier=theta, s=s, ori_T=ori_T, alpha=alpha, beta=beta)

        self.configs = self.embed_dim, self.n_kv_heads, self.n_heads, self.n_rep, self.head_dim, self.theta, self.s,self.ori_T, self.alpha,self.beta, self.freqs

        self.quantized = quantized
        self.scales = (None, None)
        self.biases = (None,None)

        wqkv_shape = embed_dim + 2 * n_kv_heads * self.head_dim, embed_dim
        wo_shape = embed_dim,embed_dim

        self.quant_map = {
            "Wqkv": 0,
            "Wo": 1,
        }

        if init:
            self.Wqkv = initializer(wqkv_shape, dtype=dtype)
            assert nx.isfinite(self.Wqkv).all(), f"non-finite detected when initializing attentipn.Wqkv."

            self.Wo = initializer(wo_shape, dtype=dtype)
            assert nx.isfinite(self.Wo).all(), f"non-finite detected when initializing attentipn.Wo."

            if quantized:
                self.Wqkv, wqkv_scale, wqkv_bias = nx.quantize(self.Wqkv)
                self.Wo, wo_scale, wo_bias = nx.quantize(self.Wo)
                self.scales = (wqkv_scale, wo_scale)
                self.biases = (wqkv_bias, wo_bias)
        else:
            self.Wqkv = nx.empty(wqkv_shape)
            self.Wo = nx.empty(wo_shape)

        self.dWqkv = nx.zeros(wqkv_shape)
        self.dWo = nx.zeros(wo_shape)

        self.Q_norm = Q_norm
        self.K_norm = K_norm

    def zeroes_gradient(self):
        self.dWqkv = nx.zeros_like(self.dWqkv)
        self.dWo = nx.zeros_like(self.dWo)
        self.Q_norm.d_gamma = nx.zeros_like(self.Q_norm.d_gamma)
        self.K_norm.d_gamma = nx.zeros_like(self.K_norm.d_gamma)

    def set_gradient(self, dwqkv, dwo, Q_d_gamma, K_d_gamma):
        self.dWqkv += dwqkv
        self.dWo += dwo
        self.Q_norm.set_gradient(Q_d_gamma)
        self.K_norm.set_gradient(K_d_gamma)

    def get_quants(self, name):
        return self.scales[self.quant_map[name]], self.biases[self.quant_map[name]]

    def set_quants(self, name, scale, bias):
        scale_list = list(self.scales)
        scale_list[self.quant_map[name]] = scale
        self.scales = tuple(scale_list)

        bias_list = list(self.biases)
        bias_list[self.quant_map[name]] = bias
        self.biases = tuple(bias_list)

    @staticmethod
    def self_type() -> str:
        return "full"

    @staticmethod
    def _forward(x:nx.ArrayLike, causal_mask:nx.ArrayLike,doc_id,  attn_configs:tuple[Any,...], attn_params: tuple[Any,...], quantization,  *, recompute_activation=True) -> tuple[nx.ArrayLike, tuple[nx.ArrayLike,...]]:
        #fp_16_x shape = (B,T,D)
        #Wqkv.T (D, D + 2 * n_kv_heads * H)
        #combined (B, T, D + 2 * n_kv_heads * H)
        embed_dim, n_kv_heads, n_heads, n_rep, head_dim, theta, s,ori_T, alpha, beta, freqs = attn_configs

        Wqkv, Wo, Q_norm_gamma, K_norm_gamma = attn_params

        wqkv_scale, wo_scale, wqkv_bias, wo_bias = quantization

        if wqkv_scale is not None:
            combined = nx.quantized_matmul(x, Wqkv, wqkv_scale,wqkv_bias, transpose=True)
        else:
            combined =  x @ Wqkv.T  # dtype

        Q = combined[..., :embed_dim] #(B, T, D)
        K = combined[..., embed_dim: embed_dim + (n_kv_heads * head_dim)]  #(B, T, n_kv_heads * H)
        V = combined[..., embed_dim + (n_kv_heads * head_dim):] #(B, T, n_kv_heads * H)

        B, T, _ = x.shape
        Q = Q.reshape(B, T, n_heads, head_dim).transpose(0,2,1,3)
        K = K.reshape(B, T, n_kv_heads, head_dim).transpose(0,2,1,3)
        V = V.reshape(B, T, n_kv_heads, head_dim).transpose(0,2,1,3)

        Q, Q_norm_caches = RMSNorm._forward(Q, Q_norm_gamma, 1e-5)
        K, K_norm_caches = RMSNorm._forward(K, K_norm_gamma, 1e-5)
        Q = rope_forward(Q, freqs, freq_multiplier=theta, s=s, ori_T=ori_T, alpha=alpha, beta=beta)
        K = rope_forward(K, freqs, freq_multiplier=theta, s=s, ori_T=ori_T, alpha=alpha, beta=beta)

        Q = Q.reshape(B,n_kv_heads, n_rep, T, head_dim).astype(x.dtype)
        K = K.astype(x.dtype)

        if doc_id is not None:
            combined_mask = causal_mask | (doc_id[:,None,None,:] != doc_id[:,None,:,None])
            combined_mask = combined_mask[:,:,None,:,:]
        else:
            combined_mask = causal_mask

        output, lse = AttentionFull.online_forward(Q, K, V, combined_mask, 512, 512)

        output = output.reshape(B, -1, T, head_dim)
        output_concat = output.transpose(0, 2, 1, 3).reshape(B, T, embed_dim)

        if wo_scale is not None:
            output_projected = nx.quantized_matmul(output_concat, Wo, wo_scale, wo_bias) #BTD
        else:
            output_projected = output_concat @ Wo


        cache =  (x, Q, Q_norm_caches, K, K_norm_caches, V, output_concat, combined_mask, lse)

        return output_projected, cache

    @staticmethod
    def online_forward(Q, K, V, combined_mask, Bq=1024, Bk=1024):
        B, n_kv_heads, n_rep, T, head_dim = Q.shape
        outputs = []
        lse_list = []

        for i in range(0, T, Bq):
            q_len = min(Bq, T - i)
            mi = nx.full((B, n_kv_heads, n_rep, q_len, 1),-1e9, nx.float32)
            li = nx.zeros((B, n_kv_heads, n_rep, q_len, 1), nx.float32)
            oi = nx.zeros((B, n_kv_heads, n_rep, q_len, head_dim), nx.float32)
            Qi = Q[..., i : i + q_len, :]

            for j in range(0, T, Bk):
                if j >= i + q_len:
                    break

                k_len = min(Bk, T - j)
                Kj = K[..., j : j + k_len, :]
                s_ij = nx.einsum("bkrQh, bkKh->bkrQK", Qi, Kj)
                s_ij = s_ij.astype(nx.float32) / nx.sqrt(head_dim, dtype=nx.float32)

                mask_ij = combined_mask[..., i : i + q_len, j : j + k_len]
                s_ij = nx.where(mask_ij, -1e9, s_ij)

                m_new = nx.maximum(nx.max(s_ij, axis=-1, keepdims=True), mi)
                alpha = nx.exp(mi - m_new)
                p_ij = nx.where(mask_ij, 0.0, nx.exp(s_ij - m_new))
                li = li * alpha + nx.sum(p_ij, axis=-1, keepdims=True)

                vj = V[..., j : j + k_len, :]
                vr = nx.einsum("bkrQK, bkKh->bkrQh", p_ij, vj.astype(nx.float32))
                oi = vr.astype(nx.float32) + oi*alpha
                mi = m_new

            oi = oi / nx.maximum(li, 1e-9)
            oi = oi.astype(Q.dtype)
            outputs.append(oi)
            logsumexp = mi + nx.log(nx.maximum(li, 1e-9))
            # nx.eval(*[oi, logsumexp])
            lse_list.append(logsumexp)

        outputs = nx.concatenate(outputs, axis=3)
        lse_list = nx.concatenate(lse_list, axis=3)
        return outputs, lse_list

    @staticmethod
    def _backward(gradient:nx.ArrayLike, caches:tuple[Any,...], attn_configs:tuple[Any,...], attn_params: tuple[Any,...], quantization:tuple[Any,...]|None=None,  *, recompute_activation=True) -> tuple[nx.ArrayLike,...]:
        embed_dim, n_kv_heads, n_heads, n_rep, head_dim,theta, s,ori_T, alpha, beta, freqs = attn_configs

        Wqkv, Wo, Q_norm_gamma, K_norm_gamma = attn_params

        wqkv_scale, wo_scale, wqkv_bias, wo_bias = quantization #type:ignore

        x, Q, Q_norm_caches, K, K_norm_caches, V, output_concat, combined_mask, lse = caches

        B, T, _ = x.shape

        if wo_scale is not None:
            d_output_concat = nx.quantized_matmul(gradient, Wo, wo_scale,wo_bias, transpose=True) #B,T,D
        else:
            d_output_concat = gradient @ Wo.T

        d_output = d_output_concat.reshape(B, T, n_heads, head_dim).transpose(0, 2, 1, 3)
        d_output = d_output.reshape(B, n_kv_heads, n_rep, T, head_dim)

        output = output_concat.reshape(B, T, n_kv_heads, n_rep, head_dim).transpose(0,2,3,1,4)
        dQ, dK, dV = AttentionFull.online_backward(d_output, output, lse, Q, K, V, combined_mask)

        del Q, K, V, d_output, d_output_concat

        dQ = dQ.reshape(B, -1, T, head_dim)
        dQ = rope_inverse(dQ, freqs, freq_multiplier=theta, s=s, ori_T=ori_T, alpha=alpha, beta=beta)
        dK = rope_inverse(dK, freqs, freq_multiplier=theta, s=s, ori_T=ori_T, alpha=alpha, beta=beta)
        dQ, Q_norm_d_gamma = RMSNorm._backward(dQ, Q_norm_caches, Q_norm_gamma)
        dK, K_norm_d_gamma = RMSNorm._backward(dK, K_norm_caches, K_norm_gamma)

        dQ = dQ.transpose(0, 2, 1, 3).reshape(B, T, embed_dim)
        dK = dK.transpose(0, 2, 1, 3).reshape(B, T, n_kv_heads * head_dim)
        dV = dV.transpose(0, 2, 1, 3).reshape(B, T,  n_kv_heads * head_dim)

        dQKV = nx.concatenate([dQ, dK,dV], axis=-1).astype(x.dtype)
        DQKV = dQKV.reshape(-1, embed_dim + 2 * (n_kv_heads * head_dim))

        del dQ, dK, dV

        X = x.reshape(-1, embed_dim)
        dWqkv = DQKV.T @ X

        H = output_concat.reshape(-1, embed_dim)
        G = gradient.reshape(-1, embed_dim)

        dWo = H.T @ G

        if wqkv_scale is not None:
            dx = nx.quantized_matmul(dQKV, Wqkv, wqkv_scale, wqkv_bias)
        else:
            dx = dQKV @ Wqkv

        del x, output_concat, freqs, Wqkv, Wo
        return dx,dWqkv,dWo,Q_norm_d_gamma,K_norm_d_gamma

    @staticmethod
    def online_backward(d_output, output, lse, Q, K, V, combined_mask, Bq=1024, Bk=1024):
        B, n_kv_heads, n_rep, T, head_dim = output.shape
        D = nx.sum(d_output.astype(nx.float32)*output.astype(nx.float32), axis=-1, keepdims=True, dtype=nx.float32)

        dQ = nx.zeros((B,n_kv_heads, n_rep, T, head_dim), nx.float32)
        dK = nx.zeros((B,n_kv_heads, T, head_dim), nx.float32)
        dV = nx.zeros((B,n_kv_heads, T, head_dim), nx.float32)

        for i in range(0, T, Bq):
            q_len = min(Bq, T - i)
            Qi = Q[...,i:i+q_len, :]
            dOi = d_output[..., i:i+q_len, :]
            dOi = dOi.astype(nx.float32)
            Li = lse[..., i:i+q_len, :]
            Di = D[..., i : i + q_len, :]

            for j in range(0, T, Bk):
                if j >= i + q_len:
                    break

                k_len = min(Bk, T - j)
                Kj = K[..., j : j + k_len, :]
                s_ij = nx.einsum("bkrQh, bkKh->bkrQK", Qi, Kj)
                s_ij = s_ij.astype(nx.float32) / nx.sqrt(head_dim, dtype=nx.float32)

                mask_ij = combined_mask[..., i : i + q_len, j : j + k_len]
                s_ij = nx.where(mask_ij, -1e9, s_ij)

                p_ij = nx.where(mask_ij, 0.0, nx.exp(s_ij - Li))

                dVj = nx.einsum("bkrQK,bkrQh->bkKh", p_ij, dOi)
                dV[..., j:j+k_len, :] += dVj

                Vj = V[..., j:j+k_len, :].astype(nx.float32)
                d_p_ij = nx.einsum("bkrQh,bkKh->bkrQK", dOi, Vj)
                d_s_ij = p_ij * (d_p_ij - Di) / nx.sqrt(head_dim, dtype=nx.float32)

                dQ[..., i : i + q_len, :] += nx.einsum("bkrQK, bkKh->bkrQh", d_s_ij, Kj.astype(nx.float32))
                dK[..., j : j + k_len, :] += nx.einsum("bkrQK,bkrQh->bkKh", d_s_ij, Qi.astype(nx.float32))

            # nx.eval(*[dQ, dK, dV])
        dQ = dQ.astype(d_output.dtype)
        dK = dK.astype(d_output.dtype)
        dV = dV.astype(d_output.dtype)

        return dQ, dK, dV

    def inference_forward(self, x, max_cache_len, freqs, quantization, cached_k, cached_v, position = 0):
        wqkv_scale, wo_scale, wqkv_bias, wo_bias = quantization #type:ignore
        if wqkv_scale is not None:
            combined = nx.quantized_matmul(x, self.Wqkv, wqkv_scale,wqkv_bias, transpose=True)
        else:
            combined =  x @ self.Wqkv.T  # dtype
        B, T, _ = x.shape

        K = combined[..., self.embed_dim: self.embed_dim + (self.n_kv_heads * self.head_dim)]

        K = K.reshape(B, T, self.n_kv_heads, self.head_dim).transpose(0,2,1,3)
        K, _ = RMSNorm._forward(K, self.K_norm.gamma, self.K_norm.epsilon)
        K = rope_forward(K, freqs, position, freq_multiplier=self.theta, s=self.s, ori_T=self.ori_T, alpha=self.alpha, beta=self.beta)
        K = K.astype(self.dtype)

        end = position + T

        V = combined[..., self.embed_dim + (self.n_kv_heads * self.head_dim):]
        V = V.reshape(B, T, self.n_kv_heads, self.head_dim).transpose(0,2,1,3)

        if end <=  max_cache_len:
            cached_k[:,:,position:end,:] = K
            active_K = cached_k[:,:,:end,:]
            cached_v[:,:,position:end,:] = V
            active_V = cached_v[:,:,:end,:]
        else:
            cached_k[:,:,:-T, :] = cached_k[:, :, T:, :]
            cached_v[:,:,:-T, :] = cached_v[:, :, T:, :]
            cached_k[:, :, -T:, :] = K
            cached_v[:, :, -T:, :] = V
            active_K = cached_k
            active_V = cached_v

        Q = combined[..., :self.embed_dim]
        Q = Q.reshape(B, T, self.n_heads, self.head_dim).transpose(0,2,1,3)
        Q, _ = RMSNorm._forward(Q, self.Q_norm.gamma, self.Q_norm.epsilon)
        Q = rope_forward(Q, freqs, position, freq_multiplier=self.theta, s=self.s, ori_T=self.ori_T, alpha=self.alpha, beta=self.beta)
        Q = Q.astype(self.dtype)

        repeats_cached_k = nx.repeat(active_K, self.n_rep, axis=1 )
        repeats_cached_v = nx.repeat(active_V, self.n_rep, axis=1 )

        scores = (Q @ repeats_cached_k.transpose(0,1,3,2)).astype(nx.float32) / nx.float_32(nx.sqrt(self.head_dim))
        if T > 1:
            if position > 0:
                scores[..., -T:] = nx.where(self.compute_mask(T), -1e9, scores[..., -T:])
            else:
                scores = nx.where(self.compute_mask(T), -1e9, scores)
        weights = nx.softmax(scores)
        weights = weights.astype(x.dtype)
        output = weights @ repeats_cached_v
        output_concat = output.transpose(0, 2, 1, 3).reshape(B, T, self.embed_dim)

        if wo_scale is not None:
            output_projected = nx.quantized_matmul(output_concat, self.Wo, wo_scale, wo_bias) #BTD
        else:
            output_projected = output_concat @ self.Wo #type:ignore

        return output_projected, cached_k, cached_v

    @staticmethod
    def compute_mask(T):
        return nx.triu(nx.ones((T, T), dtype=nx.bool_), k=1)

    def set_rope_scale(self, s):
        self.s = s
        self.freqs = precompute_freqs(self.head_dim, 32768, self.theta, s, self.ori_T, self.alpha, self.beta)
        self.configs = self.embed_dim, self.n_kv_heads, self.n_heads, self.n_rep, self.head_dim, self.theta,self.s, self.ori_T, self.alpha,self.beta, self.freqs

    def copy(self):
        Q_norm_copy = self.Q_norm.copy()
        K_norm_copy = self.K_norm.copy()
        attn_copy = AttentionFull(embed_dim=self.embed_dim, n_heads=self.n_heads, Q_norm=Q_norm_copy, K_norm=K_norm_copy, n_kv_heads=self.n_kv_heads, theta=self.theta,ori_T=self.ori_T, s=self.s, alpha=self.alpha, beta=self.beta,dtype=self.dtype, quantized=self.quantized, init=False)
        attn_copy.Wqkv = nx.copy(self.Wqkv)
        attn_copy.Wo = nx.copy(self.Wo)
        if self.quantized:
            attn_copy.scales = (nx.copy(self.scales[0]), nx.copy(self.scales[1]))
            attn_copy.biases = (nx.copy(self.biases[0]), nx.copy(self.biases[1]))

        return attn_copy

    def quantize(self):
        if self.quantized:
            return

        self.quantized = True
        self.Wqkv, scale_wcombined, bias_wcombined = nx.quantize(self.Wqkv)
        self.Wo, scale_wout, bias_out = nx.quantize(self.Wo)

        self.scales = (scale_wcombined, scale_wout)
        self.biases = (bias_wcombined, bias_out)
