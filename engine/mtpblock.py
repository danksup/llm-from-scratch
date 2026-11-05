import engine.backend as nx
from engine.rmsnorm import RMSNorm
from engine.transformer_block import TransformerBlock
import engine.initializers as initializer
class MTPBlock:
    def __init__(self,embed_dim, transformer_block:TransformerBlock, rmsnorm1:RMSNorm, rmsnorm2:RMSNorm, dtype, initializer=initializer.glorot_uniform, quantized=False, *, init=True) -> None:
        self.embed_dim = embed_dim
        self.dtype = dtype

        self.rmsnorm1 = rmsnorm1
        self.rmsnorm2 = rmsnorm2
        self.transformer_block = transformer_block

        self.quantized = quantized
        self.scale = self.bias = None

        projection_shape = 2*embed_dim, embed_dim
        if init:
            self.Wproj = initializer((projection_shape), dtype=dtype)

            if quantized:
                self.Wproj, proj_scale, proj_bias = nx.quantize(self.Wproj)
                self.scale = proj_scale
                self.bias = proj_bias
        else:
            self.Wproj = nx.empty(projection_shape)

        self.dWproj = nx.zeros(projection_shape, dtype=dtype)

    def zeroes_gradient(self):
        self.dWproj = nx.zeros_like(self.dWproj)
        self.transformer_block.zeroes_gradient()
        self.rmsnorm2.zeroes_gradient()
        self.rmsnorm1.zeroes_gradient()

    def count_params(self, dense=False):
        total = 0
        total += self.Wproj.size if not self.quantized else self.Wproj.size * 4
        total += self.transformer_block.count_params(dense=dense, quantized=self.quantized)
        total += self.rmsnorm1.gamma.size
        total += self.rmsnorm2.gamma.size

        return total

    def get_quants(self, name):
        return self.scale, self.bias

    def set_quants(self, scale, bias, name=None):
        self.scale = scale
        self.bias = bias

    @staticmethod
    def forward(h, next_token_embed, params, configs, attention, causal_mask, doc_id, epsilon,quantization, *, is_training=True, recompute_activation=True):
        wproj, gamma1, gamma2, attn_params,ffn_params, block_gamma1, block_gamma2 = params
        wproj_scale, wproj_bias, tblock_quants = quantization
        attn_configs, ff_configs = configs
        z = nx.concatenate([h, next_token_embed], axis=-1) #B,T,2D

        if wproj_scale is not None:
            projected = nx.quantized_matmul(z, wproj, wproj_scale, wproj_bias)
        else:
            projected = z @ wproj #B,T,D

        rmsnorm1_out, rmsnorm1_cache = RMSNorm._forward(projected, gamma1, epsilon)
        rmsnorm1_out = rmsnorm1_out.astype(h.dtype)

        tblock_out, masks, caches, z_loss, normalized_histogram = TransformerBlock._forward(x=rmsnorm1_out, causal_mask=causal_mask,doc_id=doc_id, attn_configs=attn_configs, attn_params=attn_params, ff_configs=ff_configs, ff_params=ffn_params, 
                                                                                              attention=attention, epsilon=epsilon,gamma1=block_gamma1, gamma2=block_gamma2, p=0.1, is_training=is_training, quantization=tblock_quants, recompute_activation=recompute_activation)

        rmsnorm2_out, rmsnorm2_cache = RMSNorm._forward(tblock_out, gamma2, epsilon)

        all_caches = (z, rmsnorm1_cache, rmsnorm2_cache, masks, caches, z_loss, normalized_histogram)
        return rmsnorm2_out, all_caches

    @staticmethod
    def backward(gradient, all_caches, all_params,all_configs, attention, gradient_scale, quantization, *, recompute_activation=True):
        z, rmsnorm1_cache, rmsnorm2_cache, masks, tblock_caches, z_loss, normalized_histogram = all_caches
        tblock_caches_attn, tblock_caches_ff, tblock_caches_rmsnorm1, tblock_caches_rmsnorm2 = tblock_caches

        wproj, gamma1, gamma2, tblock_attn_params, tblock_ff_params, tblock_gamma1,tblock_gamma2 = all_params
        mask1, mask2 = masks
        attn_configs, ff_configs = all_configs
        wproj_scale, wproj_bias,tblock_quants = quantization

        dx_rms2,d_gamma2 = RMSNorm._backward(gradient, rmsnorm2_cache, gamma2)
        dx, dWout, dWcombined, d_router,dW_dense_combined, dW_dense_out, dWqkv,dWo, tblock_d_gamma1, tblock_d_gamma2, Q_norm_d_gamma, K_norm_d_gamma= TransformerBlock._backward(dx_rms2, mask1, mask2,attention,0.1,tblock_caches_attn,tblock_caches_ff, tblock_caches_rmsnorm1, tblock_caches_rmsnorm2,  
                                   attn_configs=attn_configs,attn_params=tblock_attn_params, ff_configs=ff_configs, gradient_scale=gradient_scale,
                                   ff_params=tblock_ff_params,gamma1=tblock_gamma1,gamma2=tblock_gamma2, quantization=tblock_quants, recompute_activation=recompute_activation)

        dx_rms1, d_gamma1 = RMSNorm._backward(dx, rmsnorm1_cache, gamma1)

        D = gradient.shape[-1]
        flatten_z = z.reshape(-1, 2*D)
        flatten_dproj = dx_rms1.reshape(-1, D)

        dWproj = flatten_z.transpose() @ flatten_dproj #D,D

        if wproj_scale is not None:
            dz = nx.quantized_matmul(dx_rms1, wproj, scales=wproj_scale, biases=wproj_bias, transpose=True) #B,T,2D 
        else:
            dz = dx_rms1 @ wproj.transpose()

        de = dz[...,D:]
        dh = dz[...,:D]
        tblock_grads = {"dx":dx, "dwout":dWout, "dwcombined":dWcombined, "d_router":d_router,"dw_dense_combined":dW_dense_combined, "dw_dense_out":dW_dense_out, "dwqkv":dWqkv,"dwo":dWo, "tblock_d_gamma1":tblock_d_gamma1, "tblock_d_gamma2":tblock_d_gamma2, "q_d_gamma":Q_norm_d_gamma, "k_d_gamma":K_norm_d_gamma}

        return dh, de, dWproj, d_gamma1, d_gamma2, tblock_grads

    def inference_forward(self, h, next_token_embed, max_cache_len, cached_k, cached_v, position):
        z = nx.concatenate([h, next_token_embed], axis=-1) #B,T,2D
        projected = nx.mul(z, self.Wproj, scales= self.scale, biases=self.bias, quantized=self.quantized)
        projected = projected.astype(nx.float32) #type:ignore
        rms1_out, _ = self.rmsnorm1._forward(projected, self.rmsnorm1.gamma, self.rmsnorm1.epsilon)
        rms1_out = rms1_out.astype(self.dtype)
        tblock_out, cached_k, cached_v= self.transformer_block.inference_forward(rms1_out, max_cache_len, cached_k, cached_v, position)
        tblock_out = tblock_out.astype(nx.float32)
        rms2_out,_ = self.rmsnorm2._forward(tblock_out, self.rmsnorm2.gamma, self.rmsnorm2.epsilon)

        return rms2_out, cached_k, cached_v

    def quantize(self):
        if not self.quantized:
            self.quantized = True
            self.Wproj, self.scale, self.bias = nx.quantize(self.Wproj)
            self.transformer_block.quantize()

    def copy(self):
        wproj = nx.copy(self.Wproj)
        tblock = self.transformer_block.copy()
        rmsnorm1 = self.rmsnorm1.copy()
        rmsnorm2 = self.rmsnorm2.copy()

        copy = MTPBlock(self.embed_dim, tblock, rmsnorm1, rmsnorm2, self.dtype,quantized=self.quantized, init=False)
        copy.Wproj = wproj
        if self.quantized:
            copy.scale = nx.copy(self.scale)
            copy.bias = nx.copy(self.bias)

        return copy

    def get_things(self):
        tblock_attn = self.transformer_block.attention
        causal_mask = self.transformer_block.causal_mask
        tblock_ff = self.transformer_block.ff
        tblock_rmsnorm1 = self.transformer_block.rmsnorm1
        tblock_rmsnorm2 = self.transformer_block.rmsnorm2
        epsilon = self.rmsnorm1.epsilon

        attn_params = tblock_attn.Wqkv, tblock_attn.Wo, tblock_attn.Q_norm.gamma, tblock_attn.K_norm.gamma
        ff_params = tblock_ff.router, tblock_ff.Wcombined, tblock_ff.Wout, tblock_ff.W_dense_combined, tblock_ff.W_dense_out, tblock_ff.router_bias
        tblock_rms1 = tblock_rmsnorm1.gamma
        tblock_rms2 = tblock_rmsnorm2.gamma
        
        params = (self.Wproj, self.rmsnorm1.gamma, self.rmsnorm2.gamma, attn_params, ff_params ,tblock_rms1 ,tblock_rms2)
        configs = tblock_attn.configs, tblock_ff.configs

        quants = self.scale, self.bias , ((tblock_attn.scales + tblock_attn.biases), (tblock_ff.scales + tblock_ff.biases))

        return params, configs, quants, causal_mask, epsilon

    def set_gradient(self, **kwargs):
        self.dWproj += kwargs["dwproj"]
        self.rmsnorm1.d_gamma += kwargs["d_gamma1"]
        self.rmsnorm2.d_gamma += kwargs["d_gamma2"]
        self.transformer_block.set_gradient(**kwargs)
