import copy
from typing import Any

import engine.attention as attn
from engine.ffn import FFN
import engine.backend as nx
import engine.initializers as init
import engine.optimizer as optim
from engine.rmsnorm import RMSNorm
from engine.dataloader import DataLoader
from engine.embedding import Embedding
from engine.mtpblock import MTPBlock
from engine.losses import fused_cross_entropy
from engine.transformer_block import TransformerBlock
from helper.logger import Logger
from helper.validate_and_raise import validate_choice
from helper.singleton import sleep

optimizers = optim.Adam | optim.AdamW | optim.SGD

default_block_configs = {
    "ff_hidden_width": 1024,
    "ff_n_experts":24,
    "ff_topk":2,
    "ff_cf":1.25,
    "ff_init":"glorot_uniform",
    "ff_dense_hidden_width":1024, 
    "ff_n_dense":2,
    "ff_jitter_eps":0.02,
    "ff_bias_gamma":1e-3,
    "attn_dropout":0.1,
    "attn_type":"chunked",
    "attn_variant":"gqa",
    "attn_n_heads":16,
    "attn_init":"glorot_uniform",
}

ATTN_TYPE = {
    "chunked": {"attn": attn.AttentionChunked, "attn_chunk_size":32},
    "full": {"attn":attn.AttentionFull,},
}

ATTN_VARIANT = ["mqa", "gqa", "mha"]

INITIALIZERS = {
    "glorot_normal": init.glorot_normal,
    "glorot_uniform": init.glorot_uniform
}

attn_extra = {
    "chunked": {"chunk_size":256},
    "full":{}
}

class Transformer:
    #TODO: bug when config isnt in dict
    def __init__(self, configs: dict[str, Any] | None = None, blocks:list|None=None,mtp_blocks:list|None=None, *, init=True):
        self.blocks = []
        self.mtp_blocks = []
        configs =  {} if configs is None else configs
        self.configs = configs
        self.cross_entropy_chunk_size = 512

        self.vocab_size = configs.get("vocab_size", None)
        assert self.vocab_size is not None, "vocab size can't be None"

        self.embed_dim = configs.get("embed_dim", 128)
        self.dtype = configs.get("dtype", "float32")
        validate_choice(self.dtype, "dtype", nx.floating_type_str)
        if isinstance(self.dtype, str):
            self.dtype = nx.str_to_dtype[self.dtype]
        if nx.issubdtype(self.dtype, nx.integer):
            raise ValueError(f"please use floating type for dtype initialization, got {self.dtype} instead.")

        self.quantized = configs.get("quantized", False)
        self.recompute_activation = configs.get("recompute_activation", True)
        self.doc_id = configs.get("doc_id", True)
        self.mtp_lambda = configs.get("mtp_lambda", .5)

        self.check_non_finite = configs.get("check_non_finite", True)
        
        self.embedding = Embedding(self.vocab_size, self.embed_dim, self.dtype, self.quantized, init=init)
        self.rmsnorm_final = RMSNorm(self.embed_dim, init=init)

        self.gradient_scale = configs.get("gradient_scale", 4096)
        self.max_gradient_scale = self.gradient_scale

        assert self.gradient_scale > 0, "gradient scale cant be less than 1"
        no_class_attn_type = copy.deepcopy(ATTN_TYPE)
        no_class_attn_type[default_block_configs["attn_type"]].pop('attn')
        self.block_configs =  default_block_configs | configs.get("block_configs", {})
        self.individual_block_configs = []

        if blocks is None:
            n_blocks =  configs.get("n_blocks",4)
            block_overrides = configs.get("block_overrides", {})
            if block_overrides:
                for value in block_overrides.values():
                    if any(i in value for i in ["quantize_to_int8", "dtype"]):
                        raise ValueError('individual block config cant have dtype or quantization configuration.')

            for i in range(n_blocks):
                override = block_overrides.get(i, {})
                this = self.block_configs
                overrided = this | override

                attn_variant = overrided["attn_variant"]
                validate_choice(attn_variant, "attn_variant", ATTN_VARIANT)

                attn_type_str = overrided["attn_type"]
                validate_choice(attn_type_str, "attn_type", ATTN_TYPE)

                overrided = overrided | ATTN_TYPE[overrided["attn_type"]] | this  | override
                overrided.pop('attn')

                validate_choice(overrided["attn_init"], "attn_init", INITIALIZERS)
                validate_choice(overrided["ff_init"], "ff_init", INITIALIZERS)

                self.individual_block_configs.append(overrided)

                D = self.embed_dim
                H = overrided["ff_hidden_width"]
                attn_init = INITIALIZERS[overrided["attn_init"]]
                E = overrided["ff_n_experts"]
                CF = overrided["ff_cf"]
                topk = overrided["ff_topk"]
                ff_init = INITIALIZERS[overrided["ff_init"]]
                ff_dense_hidden_width = overrided["ff_dense_hidden_width"]
                ff_n_dense = overrided["ff_n_dense"]
                ff_jitter_eps = overrided["ff_jitter_eps"]
                ff_bias_gamma = overrided["ff_bias_gamma"]

                if "attn_chunk_size" in override and override.get("attn_type", None) == "full":
                    raise ValueError(f"[block {i}] attention type of {attn_type_str} doesn't accept \"attn_chunk_size\"")
                n_heads = overrided["attn_n_heads"]
                chunk_size = overrided.get("attn_chunk_size", None)

                head_dim = D // n_heads
                Q_norm = RMSNorm(head_dim)
                K_norm = RMSNorm(head_dim)

                if attn_variant == "mha":
                    n_kv_heads = n_heads
                elif attn_variant == "mqa":
                    n_kv_heads = 1
                else:
                    n_kv_heads = overrided["attn_n_kv_heads"]

                attn_type = ATTN_TYPE[attn_type_str]["attn"]

                a = copy.deepcopy(attn_extra)
                if attn_type_str == "chunked":
                    a["chunked"]["chunk_size"] = chunk_size

                attn = attn_type(embed_dim=D, n_heads=n_heads, Q_norm = Q_norm, K_norm=K_norm, n_kv_heads=n_kv_heads, dtype=self.dtype, initializer=attn_init, quantized=self.quantized, init=init, **a[attn_type_str])
                ff = FFN(CF, topk, E, D, H, ff_dense_hidden_width, ff_n_dense,ff_jitter_eps,ff_bias_gamma, dtype=self.dtype, initializer=ff_init, quantized=self.quantized, init=init)
                rmsnorm1 = RMSNorm(D, init=init)
                rmsnorm2 = RMSNorm(D, init=init)
                transformer_block = TransformerBlock(attn, ff, rmsnorm1, rmsnorm2)
               
                self.blocks.append(transformer_block)
        else:
            self.blocks = blocks

        if mtp_blocks is None:
            n_mtp = configs.get("n_mtp", 1)
            for i in range(n_mtp):
                last_block = self.blocks[-1].copy()
                rmsnorm1 = RMSNorm(self.embed_dim, init=init)
                rmsnorm2 = RMSNorm(self.embed_dim, init=init)
                mtp = MTPBlock(self.embed_dim, last_block, rmsnorm1, rmsnorm2, dtype=self.dtype, quantized=self.quantized, init=init)
                self.mtp_blocks.append(mtp)
        else:
            self.mtp_blocks = mtp_blocks
                
            
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        self.logger:Logger = kwds["logger"]
        return self
        
    def __str__(self) -> str:
        return self.get_configs_str()

    def count_params(self, dense=False) -> Any:
        """
        whole architecture number of (trainable) params
        """
        total = 0
        total_dense = 0
        for i in self.blocks:
            total += i.count_params(quantized=self.quantized)
            total_dense += i.count_params(dense=True, quantized=self.quantized)

        for mtp in self.mtp_blocks:
            total += mtp.count_params()
            total_dense += mtp.count_params(dense=True)

        embedding_size = self.embedding.lookup_table.size
        if self.quantized:
            embedding_size *= 4

        total += embedding_size
        total_dense += embedding_size

        total += self.rmsnorm_final.gamma.size
        total_dense += self.rmsnorm_final.gamma.size
        return total, total_dense

    @staticmethod
    def compute_mask(attn_str, block, T):
        if attn_str == "chunked":
            chunk_size = block.attention.chunk_size
            chunk_size = min(chunk_size, T-1)
            remainder = (chunk_size - (T % chunk_size)) % chunk_size
            n_chunk = (T + remainder) // chunk_size
            if block.causal_mask is None or block.causal_mask.shape != (n_chunk, chunk_size, 2 * chunk_size):
                block.causal_mask = block.attention.compute_mask(chunk_size,T)
        elif attn_str == "full":
            if block.causal_mask is None or block.causal_mask.shape != (T, T):
                block.causal_mask = block.attention.compute_mask(T)
        elif attn_str == "swa":
            W = block.attention.W
            W = min(W, T-1)
            if block.causal_mask is None or block.causal_mask.shape != (T, W + 1):
                block.causal_mask = block.attention.compute_mask(T)

    def forward(self, inputs:Any, doc_id:Any=None,*, is_training=True) -> Any:
        output = inputs.astype(self.dtype)
        all_masks = []
        all_caches = []
        total_z_loss = nx.array(0.0, dtype=nx.float32)
        histograms = [None for _ in range(len(self.blocks))]
        for idx, block in enumerate(self.blocks):
            try:
                B,T,_ = output.shape
                epsilon = block.rmsnorm1.epsilon
                gamma1 = block.rmsnorm1.gamma
                gamma2 = block.rmsnorm2.gamma

                P = nx.array(0.1, dtype=self.dtype)

                attn_str = block.attention.self_type()

                self.compute_mask(attn_str, block, T)

                attn_params = block.attention.Wqkv, block.attention.Wo, block.attention.Q_norm.gamma, block.attention.K_norm.gamma
                ff_params = block.ff.router, block.ff.Wcombined, block.ff.Wout, block.ff.W_dense_combined, block.ff.W_dense_out, block.ff.router_bias
                quants = (block.attention.scales + block.attention.biases, block.ff.scales + block.ff.biases)
                ff_out ,masks, caches, z_loss, normalized_histogram = block._forward(output, block.causal_mask,doc_id, attn_str ,block.attention.configs, attn_params, block.ff.configs, ff_params, epsilon, gamma1, gamma2, P, is_training, quants, recompute_activation=self.recompute_activation)
                total_z_loss += z_loss
                output = ff_out
                all_masks.append(masks)
                all_caches.append(caches)
                if normalized_histogram is not None:
                    histograms[idx] = nx.zeros_like(normalized_histogram)
                    histograms[idx] += normalized_histogram
            except TypeError as e:
                print(f"[block {idx}] TypeError")
                raise TypeError(e)
            except ValueError as e:
                print(f"[block {idx}] ValueError")
                raise ValueError(e)

        last_output = output.astype(self.dtype)
        rmsnorm_out, rms_cache = self.rmsnorm_final._forward(last_output, self.rmsnorm_final.gamma,self.rmsnorm_final.epsilon)

        # if self.quantized:
        #     scores = nx.quantized_matmul(rmsnorm_out, self.embedding.lookup_table, self.embedding.table_scale, self.embedding.bias, transpose=True)
        # else:
        #     scores = rmsnorm_out.astype(self.dtype) @ self.embedding.lookup_table.T # B,T,V

        all_caches += rms_cache,
        # return scores, rmsnorm_out, all_masks, all_caches, total_z_loss, histograms, last_output
        return rmsnorm_out, all_masks, all_caches, total_z_loss, histograms, last_output

    def backward(self, err_signal:Any,  all_masks, all_caches:list, doc_id, total_dh) -> Any:
        current_grad = err_signal
        rmsnorm_final_cache = all_caches.pop()
        current_grad, rmsnorm_final_d_gamma = self.rmsnorm_final._backward(current_grad, rmsnorm_final_cache, self.rmsnorm_final.gamma)
        current_grad[:,:-1] += total_dh
        self.rmsnorm_final.d_gamma = rmsnorm_final_d_gamma if getattr(self.rmsnorm_final, "d_gamma", None) is None else self.rmsnorm_final.d_gamma + rmsnorm_final_d_gamma
        for block, masks,caches in zip(self.blocks[::-1], all_masks[::-1],all_caches[::-1]):
            current_grad = current_grad.astype(self.dtype)
            _,T,_ = current_grad.shape
            caches_attn, caches_ff, caches_rmsnorm1, caches_rmsnorm2 = caches
            mask1, mask2 = masks
            ff_configs = block.ff.configs
            P = nx.array(0.1, dtype=self.dtype)

            ff_params = (block.ff.router,block.ff.Wcombined, block.ff.Wout,  block.ff.W_dense_combined, block.ff.W_dense_out, block.ff.router_bias)
            attn_str = block.attention.self_type()
            attn_configs = block.attention.configs
            attn_params = block.attention.Wqkv, block.attention.Wo, block.attention.Q_norm.gamma, block.attention.K_norm.gamma
            scales = (block.attention.scales + block.attention.biases, block.ff.scales + block.ff.biases)

            dx, dWout, dWcombined, d_router,dW_dense_combined, dW_dense_out, dWqkv,dWo, d_gamma1, d_gamma2, Q_norm_d_gamma, K_norm_d_gamma = block._backward(current_grad, mask1=mask1, mask2=mask2, p=P, attention=attn_str,
                                                                caches_attn=caches_attn, caches_ff=caches_ff, caches_rmsnorm1=caches_rmsnorm1, caches_rmsnorm2=caches_rmsnorm2,
                                                                attn_configs = attn_configs, attn_params=attn_params, gamma1=block.rmsnorm1.gamma, gamma2=block.rmsnorm2.gamma, ff_params=ff_params, ff_configs=ff_configs, gradient_scale=self.gradient_scale, quantization=scales, recompute_activation=self.recompute_activation)
            if block.ff.n_experts > 0:
                block.ff.dWout += dWout
                block.ff.dWcombined += dWcombined
                block.ff.d_router += d_router

            if block.ff.n_dense > 0:
                block.ff.dW_dense_combined += dW_dense_combined
                block.ff.dW_dense_out += dW_dense_out

            block.attention.dWqkv += dWqkv
            block.attention.dWo += dWo
            block.attention.Q_norm.d_gamma +=  Q_norm_d_gamma
            block.attention.K_norm.d_gamma +=  K_norm_d_gamma

            block.rmsnorm1.d_gamma += d_gamma1
            block.rmsnorm2.d_gamma += d_gamma2

            current_grad = dx

        return current_grad

    def mtp_forward(self, input, next_tokens, doc_id, *, is_training=True):
        output = input.astype(self.dtype)
        all_caches = []
        B,T,D = input.shape
        for mtp in self.mtp_blocks:
            attn_type = mtp.transformer_block.attention.self_type()
            self.compute_mask(attn_type, mtp.transformer_block, T)
            params, configs, quants, causal_mask, epsilon = mtp.get_things()
            rmsnorm2_out, caches = MTPBlock.forward(output, next_tokens, params, configs, attn_type, causal_mask, doc_id, epsilon, quants, is_training=is_training, recompute_activation=self.recompute_activation)
            output = rmsnorm2_out
            all_caches.append(caches)
        return output, all_caches

    def mtp_backward(self, gradient, all_caches):
        dh = gradient
        de = None
        for mtp,caches in zip(self.mtp_blocks[::-1], all_caches[::-1]):
            attn_type = mtp.transformer_block.attention.self_type()
            params, configs, quants, causal_mask, epsilon = mtp.get_things()
            dh, de, dWproj, d_gamma1, d_gamma2, tblock_grads = MTPBlock.backward(dh, caches, params, configs, attn_type, self.gradient_scale, quants, recompute_activation=self.recompute_activation)
            mtp.set_gradient(dwproj = dWproj, d_gamma1=d_gamma1, d_gamma2=d_gamma2, **tblock_grads)

        return dh,de
        
    def eval_networks(self, others:list|None = None, include_gradients:bool=True, optimizer:optimizers|None=None):
        to_eval = []
        if others is not None:
            to_eval.extend(others)

        for layer_obj, w_param_name, dw_param_name, _, _ in self.get_parameters():
            weight_obj = getattr(layer_obj, w_param_name)
            to_eval.append(weight_obj)
            if include_gradients:
                if dw_param_name is not None:
                    dweight_obj = getattr(layer_obj, dw_param_name)
                    to_eval.append(dweight_obj)

        if not include_gradients:
            if optimizer is not None:
                to_eval.append(optimizer.lr)
                if hasattr(optimizer, "state"):
                    to_eval.append(optimizer.state)
                if hasattr(optimizer, "masters"):
                    to_eval.append(optimizer.masters) #type:ignore

        nx.eval(*to_eval)

    def get_block_weights(self, layers, params, iterable:list|None=None, prefix=None):
        iterable = self.blocks if iterable is None else iterable
        for i, block in enumerate(iterable):
            prefix_ = i if prefix is None else prefix
            for layer_i, layer in enumerate(layers):
                layer_obj = getattr(block, layer, None)
                if layer_obj is not None:
                    for param_name in params[layer_i]:
                        if len(param_name) == 2:
                            w,dw = param_name 
                            w_ = getattr(layer_obj, w, None)
                            dw_full_str = None if dw is None else f"{prefix_}.{layer}.{dw}"
                            if w_ is not None:
                                yield layer_obj, w, dw,f"{prefix_}.{layer}.{w}", dw_full_str

                        elif len(param_name) == 3:
                            sublayer, w, dw = param_name
                            subobj = getattr(layer_obj, sublayer, None)
                            if subobj is not None:
                                w_ = getattr(subobj, w, None)
                                dw_full_str = None if dw is None else f"{prefix_}.{layer}.{sublayer}.{dw}"
                                if w_ is not None:
                                    yield subobj, w, dw, f"{prefix_}.{layer}.{sublayer}.{w}", dw_full_str
                  
    def get_mtp_things(self):
        upper = [None, None, None , "transformer_block"]
        middle = [None, "rmsnorm1", "rmsnorm2", ["ff", "attention", "rmsnorm1", "rmsnorm2"]]
        lower = [("Wproj", "dWproj"), ("gamma", "d_gamma"), ("gamma", "d_gamma"), [[("router", "d_router"), ("router_bias", None), ("Wcombined", "dWcombined"), ("Wout", "dWout"), ("W_dense_combined", "dW_dense_combined"), ("W_dense_out", "dW_dense_out")], [("Wqkv", "dWqkv"), ("Wo", "dWo"), ("Q_norm", "gamma", "d_gamma"), ("K_norm", "gamma", "d_gamma")], [("gamma", "d_gamma")], [("gamma", "d_gamma")]]]

        for i, mtp in enumerate(self.mtp_blocks):
            for idx, thing in enumerate(upper):
                layer = middle[idx]
                param = lower[idx]
                if thing is None:
                    param_w = param[0]
                    param_dw = param[1]
                    if layer is None:
                        param_ = getattr(mtp, param_w, None)
                        if param_ is not None:
                            yield mtp, param_w, param_dw, f"mtp.{i}.{param_w}", f"mtp.{i}.{param_dw}"
                    else:
                        layer_ = getattr(mtp, layer, None)
                        if layer_ is not None:
                            param_ = getattr(layer_, param_w, None)
                            if param_ is not None:
                                yield layer_, param_w, param_dw, f"mtp.{i}.{layer}.{param_w}", f"mtp.{i}.{layer}.{param_dw}"
                else:
                    for a,b,c,d,e in self.get_block_weights(layer, param, iterable=[mtp.transformer_block], prefix=f"mtp.{i}.transformer_block"):
                        yield a,b,c,d,e
                

    def get_parameters(self, *, block_only:bool=False, include_mtp=True):
        if not block_only:
            if getattr(self.embedding, "lookup_table", None) is not None:
                yield self.embedding, "lookup_table", "d_lookup_table", "embedding.lookup_table", "embedding.d_lookup_table"

            if getattr(self.rmsnorm_final, "gamma", None) is not None:
                yield self.rmsnorm_final, "gamma","d_gamma", "rmsnorm_final.gamma","rmsnorm_final.d_gamma"

        layers = ["ff", "attention", "rmsnorm1", "rmsnorm2"]
        params = [[("router", "d_router"),("router_bias", None),("Wcombined", "dWcombined"),("Wout", "dWout"),("W_dense_combined", "dW_dense_combined"),("W_dense_out", "dW_dense_out"),],[("Wqkv", "dWqkv"),("Wo", "dWo"),("Q_norm", "gamma", "d_gamma"),("K_norm", "gamma", "d_gamma"),],[("gamma", "d_gamma"),],[("gamma", "d_gamma"),],]
        for a,b,c,d,e in self.get_block_weights(layers, params):
            yield a,b,c,d,e

        if include_mtp:
            for a,b,c,d,e in self.get_mtp_things():
                yield a,b,c,d,e

    def gradient_clipping_factor(self, microbatch_size, max_norm=nx.float_32(0.5)):
        summed = nx.array(0, nx.float32)

        for layer, _, dw, _,_ in self.get_parameters(): 
            if dw is not None:
                param = getattr(layer, dw) 
                if dw == "d_lookup_table":
                    summed += nx.sum(nx.square(param.astype(nx.float32) / microbatch_size))
                else:
                    summed += nx.sum(nx.square(param.astype(nx.float32) / self.gradient_scale / microbatch_size))

        l2_norm = nx.sqrt(summed)
        raw_scale = max_norm / (l2_norm + nx.float_32(1e-6))
        scale = nx.minimum(nx.float_32(1.0), raw_scale)

        return scale

    def network_non_finite_check(self, additional:dict[str,list[Any]]|None=None):
        texts = ""
        non_finite = False

        failures = {}
        failures["-"] = []
        for i in range(self.configs["n_blocks"]):
            failures[i] = []

        if additional:
            for key,val in additional.items():
                layer = getattr(self, key)
                for weight in val:
                    weight_ = getattr(layer, weight)
                    if not nx.isfinite(weight_).all():
                        failures["-"].append(f"{weight}")
                        non_finite = True

        parameters = self.get_parameters()
        for layer_obj, w_param_name, d_param_name, w_name, dw_name in parameters:
            weight = getattr(layer_obj, w_param_name)
            if d_param_name is not None:
                d_weight = getattr(layer_obj, d_param_name)

                is_w_finite = nx.isfinite(weight).all()
                is_d_finite = nx.isfinite(d_weight).all() 
                if not is_w_finite or not is_d_finite:
                    non_finite = True
                    block_i = w_name.split(".")[0]
                    try:
                        block_i = int(block_i)
                        failures[block_i].append(w_name)
                        if not is_d_finite:
                            failures[block_i].append(dw_name)
                    except ValueError:
                        failures["-"].append(w_name)
                        if not is_d_finite:
                            failures["-"].append(dw_name)

        for key, val in failures.items():
            texts += f"block_{key}: {val}"
        return non_finite, texts

    def zeroes_gradient(self):
        for block in self.blocks:
            block.zeroes_gradient()

        for mtp in self.mtp_blocks:
            mtp.zeroes_gradient()

        self.rmsnorm_final.zeroes_gradient()
        self.embedding.zeroes_gradient()

    def train(self, dataloader:DataLoader, optimizer:optimizers, total_epoch:int, max_step:int=50000, eval_every:int=5, microbatch_size:int=16):
        total_loss = nx.float_32(0.0)
        count = 0
        microstep = 0
        step = 0
        total_histograms = None
        clean_step = 0

        def reset_and_halve_grad_scale():
            nonlocal total_loss, count, microstep, total_histograms, clean_step
            total_loss = nx.float_32(0)
            count = 0
            microstep = 0
            total_histograms = None
            clean_step = 0
            self.gradient_scale = max(1, self.gradient_scale//2)

            for block in self.blocks:
                block.zeroes_gradient()

        for contexts, next_tokens in dataloader.stream_batches(dataloader.train_files):
            B = dataloader.batch_size
            T = dataloader.context_size
            contexts = nx.array(contexts).reshape(B,T)

            doc_id = nx.cumsum(contexts == 2, axis=-1) if self.doc_id else None #B,T

            next_tokens = nx.array(next_tokens).reshape(dataloader.batch_size, dataloader.context_size)

            if step >= max_step:
                break

            embedded = self.embedding.forward(contexts)  # shape (batch, context_size, embed_dim)
            last_output, all_masks, all_caches, total_z_loss, histograms, unnormalized_output = self.forward(embedded, doc_id)

            lookup_table = self.embedding.lookup_table
            if self.quantized:
                lookup_table = nx.dequantize(lookup_table, self.embedding.table_scale,biases=self.embedding.bias, dtype=self.dtype)

            mtp_loss = nx.array(0, nx.float32)
            total_dh = nx.array(0, nx.float32)
            if self.mtp_blocks:
                next_tokens_embedded = self.embedding.forward(next_tokens[:, :-1])
                doc_id_sliced = doc_id[:, :-1] if doc_id is not None else doc_id

                mtp_output, mtp_caches  = self.mtp_forward(unnormalized_output[:, :-1], next_tokens_embedded, doc_id_sliced, is_training=True)
                mtp_loss, mtp_block_gradient, mtp_d_table = fused_cross_entropy(mtp_output, lookup_table, next_tokens[:, 1:], gradient_scale=self.gradient_scale, chunk_size=self.cross_entropy_chunk_size)
                dh,de = self.mtp_backward(mtp_block_gradient, mtp_caches)
                total_dh += dh
                self.embedding.d_lookup_table = nx.add_at(self.embedding.d_lookup_table, next_tokens[:, :-1], de/self.gradient_scale)
                self.embedding.d_lookup_table += mtp_d_table * self.mtp_lambda

            if total_histograms is None:
                total_histograms = histograms
            else:
                for i in range(len(self.blocks)):
                    if histograms[i] is not None:
                        total_histograms[i] += histograms[i]

            loss, block_gradient, d_table = fused_cross_entropy(last_output, lookup_table, next_tokens, gradient_scale=self.gradient_scale, chunk_size=self.cross_entropy_chunk_size)
            loss = loss  + total_z_loss + (nx.mean(mtp_loss) * self.mtp_lambda)

            current_grad = self.backward(block_gradient, all_masks, all_caches, doc_id, total_dh)
            current_grad = current_grad.astype(nx.float32) / self.gradient_scale

            self.embedding.d_lookup_table = nx.add_at(self.embedding.d_lookup_table, contexts, current_grad) 
            self.embedding.d_lookup_table += d_table

            total_loss += loss * next_tokens.size
            count += next_tokens.size

            for idx, block in enumerate(self.blocks):
                if histograms[idx] is not None:
                    block.ff.update_bias(histograms[idx], 1e-3)
            microstep += 1

            if microstep % eval_every == 0 or microstep == microbatch_size:
                to_eval = [total_loss, self.embedding.lookup_table, self.embedding.d_lookup_table, self.rmsnorm_final.gamma, self.rmsnorm_final.d_gamma, total_histograms]
                self.eval_networks(to_eval)

                if self.check_non_finite:
                    gradient_mean = nx.mean(current_grad)
                    if not nx.isfinite(loss).item() or not nx.isfinite(gradient_mean).item():
                        if self.gradient_scale <= 1:
                            self.logger.error("non-finite :(", category=FloatingPointError)
                        
                        forward_nan = nx.isnan(loss)
                        forward_inf = nx.isinf(loss)
                        nan_weights = self.network_non_finite_check()[1]

                        backward_nan = nx.isnan(gradient_mean)
                        backward_inf = nx.isinf(gradient_mean)

                        reset_and_halve_grad_scale()
                        self.logger.warn(f"[NON-FINITE step: {step}] non finite loss at microstep {microstep}. isnan forward/backward: {forward_nan}/{backward_nan} | isinf forward/backward: {forward_inf}/{backward_inf}\ngradient_scale is halved: {self.gradient_scale}", f"\n non-finite weights:\n{nan_weights}", category= UserWarning)
                        continue                        

            if microstep == microbatch_size:
                if self.check_non_finite:
                    check = self.network_non_finite_check()
                    if check[0]:
                        reset_and_halve_grad_scale()
                        self.logger.warn(f"[NON-FINITE step: {step}] non finite loss at microstep {microstep}. \ngradient_scale is halved: {self.gradient_scale}", f"\n non-finite weights:\n{check[1]}")
                        continue         
                    
                all_network_params = []
                gscale = self.gradient_clipping_factor(microbatch_size)

                for obj, w, dw, full_w_name, _ in self.get_parameters():
                    if dw is not None:
                        w_ = getattr(obj, w)
                        if w == "lookup_table":
                            dw_ = getattr(obj, dw).astype(nx.float32) /  microbatch_size * gscale
                        else:
                            dw_ = getattr(obj, dw).astype(nx.float32) / self.gradient_scale / microbatch_size * gscale

                        decay = True
                        if w in ["lookup_table", "gamma", "router"]:
                            decay = False

                        if self.quantized and w not in ["gamma", "router"]:
                            quants = obj.get_quants(name=w) #type:ignore
                            w_ = nx.dequantize(w_, scales=quants[0], biases=quants[1])

                        all_network_params.append((full_w_name, w_.astype(nx.float32), dw_, decay)) #type:ignore

                optimized = optimizer.step_many(all_network_params, max_step, total_epoch)
                del all_network_params

                for obj, w, dw, full_w_name, _ in self.get_parameters():
                    if dw is not None:
                        new_param = optimized[full_w_name]

                        exc = ["router", "gamma"]
                        if self.quantized and w not in exc:
                            new_param, new_scale, new_bias = nx.quantize(new_param)
                            obj.set_quants(scale=new_scale, bias=new_bias, name=w) #type:ignore
                            setattr(obj, w, new_param)
                        else:
                            if w in exc:
                                setattr(obj, w, new_param.astype(nx.float32))
                            else:
                                setattr(obj, w, new_param.astype(dtype=self.dtype))
                
                del optimized
                self.zeroes_gradient()
                step += 1

                clean_step += 1
                if clean_step > 0 and clean_step % 1000 == 0:
                    self.gradient_scale = min(self.gradient_scale * 2, self.max_gradient_scale)

                self.eval_networks([self.rmsnorm_final.gamma], include_gradients=False, optimizer=optimizer)

                yield total_loss.item(), count, total_histograms, step
                total_loss = nx.float_32(0)
                count = 0
                microstep = 0
                total_histograms = None
                # nx.clear_cache()

    def validate(self, dataloader:DataLoader, val_step:int|None=None):
        total_loss = nx.float_32(0.0)
        total_mtp_loss = nx.float_32(0.0)
        count = 0
        step_counter = 0

        if not dataloader.validation_files:
            return None, None

        for contexts, next_tokens in dataloader.stream_batches(dataloader.validation_files):
            if isinstance(val_step, int) and step_counter >= val_step:
                break
            contexts = nx.array(contexts).reshape(dataloader.batch_size, dataloader.context_size)
            next_tokens = nx.array(next_tokens).reshape(dataloader.batch_size, dataloader.context_size)
        
            embedded = self.embedding.forward(contexts)

            doc_id = nx.cumsum(contexts == 2, axis=-1) if self.doc_id else None #B,T

            lookup_table = self.embedding.lookup_table
            if self.quantized:
                lookup_table = nx.dequantize(lookup_table, self.embedding.table_scale, self.embedding.bias, self.dtype)
            output, _, _, total_z_loss, _, unnormalized_output = self.forward(embedded,doc_id=doc_id, is_training=False)
            val_loss = fused_cross_entropy(output, lookup_table, next_tokens, is_training=False, gradient_scale=self.gradient_scale, chunk_size=self.cross_entropy_chunk_size)

            mtp_loss = nx.array(0, nx.float32)
            if self.mtp_blocks:
                next_tokens_embedded = self.embedding.forward(next_tokens[:, :-1])
                doc_id_sliced = doc_id[:, :-1] if doc_id is not None else doc_id
                mtp_output, _  = self.mtp_forward(unnormalized_output[:, :-1], next_tokens_embedded, doc_id_sliced, is_training=False)
                mtp_loss = fused_cross_entropy(mtp_output,lookup_table, next_tokens[:, 1:], gradient_scale=self.gradient_scale, is_training=False, chunk_size=self.cross_entropy_chunk_size)

            val_loss = nx.mean(val_loss)  + total_z_loss 
            total_loss += val_loss * next_tokens.size
            total_mtp_loss += nx.mean(mtp_loss) * next_tokens.size
            count += next_tokens.size
            step_counter += 1

            nx.eval(*[val_loss, total_loss, total_mtp_loss])

        if count == 0:
            return None, None

        final_loss = total_loss / count
        final_mtp_loss = total_mtp_loss / count
        return final_loss.item(), final_mtp_loss.item()

    def inference(self, context:Any, max_cache_len, all_caches = None,  position = 0) -> Any:
        if all_caches is None:
            all_caches = []
            for _, block in enumerate(self.blocks):
                n_kv = block.attention.n_kv_heads
                head_dim = block.attention.head_dim
                kv_chace_shape = (context.shape[0], n_kv ,max_cache_len, head_dim)
                all_caches.append((nx.zeros(kv_chace_shape, dtype=self.dtype), nx.zeros(kv_chace_shape, dtype=self.dtype)))

        output = self.embedding.forward(context)
        for idx, block in enumerate(self.blocks):
            cached_k, cached_v = all_caches[idx]
            ff_out, cache_k, cache_v = block.inference_forward(output,max_cache_len, cached_k, cached_v, position)
            all_caches[idx] = (cache_k, cache_v)
            output = ff_out

        rmsfinal_out, _ = self.rmsnorm_final._forward(output, self.rmsnorm_final.gamma, self.rmsnorm_final.epsilon)
        if self.quantized:
            scores = nx.quantized_matmul(rmsfinal_out, self.embedding.lookup_table, self.embedding.table_scale, self.embedding.bias, transpose=True) #type:ignore
        else:
            scores = rmsfinal_out @ self.embedding.lookup_table.T

        return scores, all_caches, output

    def mtp_inference(self, h, next_tokens,max_cache_len, all_caches, position):
        next_token = nx.array([[next_tokens]], dtype=nx.int32).reshape(1, 1)
        if all_caches is None:
            all_caches = []
            for _, mtp in enumerate(self.mtp_blocks):
                n_kv = mtp.transformer_block.attention.n_kv_heads
                head_dim = mtp.transformer_block.attention.head_dim
                kv_cache_shape = (next_token.shape[0], n_kv, max_cache_len, head_dim)
                all_caches.append((nx.zeros(kv_cache_shape, dtype=self.dtype), nx.zeros(kv_cache_shape, dtype=self.dtype)))

        next_token_embed = self.embedding.forward(next_token)
        output = None
        for idx, mtp in enumerate(self.mtp_blocks):
            cached_k, cached_v = all_caches[idx]
            output, cached_k, cached_v = mtp.inference_forward(h[:, -1:, :] ,next_token_embed, max_cache_len,cached_k, cached_v, position)
            all_caches[idx] = (cached_k, cached_v)

        scores = nx.mul(output, self.embedding.lookup_table, self.embedding.table_scale, self.embedding.bias, transpose=True)
        return scores, all_caches
    
    def get_configs(self):
        configs = {}
        configs["vocab_size"] = self.vocab_size
        configs["embed_dim"] = self.embed_dim
        configs["dtype"] = nx.dtype_to_srt[self.dtype]
        configs["quantized"] =  self.quantized
        configs["gradient_scale"] = self.gradient_scale
        configs["recompute_activation"] = self.recompute_activation
        configs["n_mtp_block"] = len(self.mtp_blocks)
        return configs

    def get_configs_str(self):
        configs = ""
        for k,v in self.get_configs().items():
            if k not in ["dtype"]:
                configs += f"{k}: {str(v)}\n"

        configs += "precision: full (float32)\n" if self.dtype == nx.float32 else f"precision: mixed precision ({self.dtype})\n"
        configs += f"block configs: {self.block_configs}\n"
        configs += f"check_non_finite: {self.check_non_finite}\n"
        configs += "individual block configs (only difference is shown): \n"
        similar_count = 0
        for i, block in  enumerate(self.individual_block_configs):
            if block == self.block_configs:
                similar_count += 1
                continue
            ind_con = f"block {i}: "
            for key, val in block.items():
                if self.block_configs.get(key) != val:
                    ind_con += f"{key}: {val} | "
            ind_con += "\n"
            configs += ind_con

        if similar_count == len(self.individual_block_configs):
            configs += "None\n"
        return configs

    def get_all_weights(self) -> dict[str,nx.ArrayLike]:
        all_weights = {}
        weights = self.get_parameters()
        for layer_obj, param_name,_, name, _ in weights:
            weight = getattr(layer_obj, param_name)
            all_weights[name] = weight

        return all_weights

    def quantize(self):
        if self.quantized:
            return
        else:
            self.quantized = True
            self.configs["quantized"] = True

            self.embedding.quantize()
            for block in self.blocks:
                block.quantize()

            for mtpblock in self.mtp_blocks:
                mtpblock.quantize()

    def get_quant_params(self):
        quant_params = {}
        
        for obj, d, _, wfull, _ in self.get_parameters():
            if d not in ["gamma", "router_bias", "router"]:
                scales, biases = obj.get_quants(name=d) #type:ignore
                sname = wfull +".scales"
                bname = wfull +".biases"
                quant_params[sname] = scales
                quant_params[bname] = biases

        return quant_params

    def get_block_configs(self):
        configs = {}
        for idx, block in enumerate(self.blocks):
            configs[idx] = block.get_configs()

        return configs

    def discard_mtp(self):
        self.mtp_blocks = []
        self.configs["n_mtp"] = 0

    def copy(self) -> "Transformer":
        configs = copy.deepcopy(self.configs)
        block_copy = []
        mtp_copy = []
        for block in self.blocks:
            block_copy.append(block.copy())

        for mtp in self.mtp_blocks:
            mtp_copy.append(mtp.copy())

        embedding_copy = self.embedding.copy()
        rmsfinal_copy = self.rmsnorm_final.copy()
        transformer_copy = Transformer(configs, block_copy,mtp_copy, init=False)
        transformer_copy.embedding = embedding_copy
        transformer_copy.rmsnorm_final = rmsfinal_copy

        return transformer_copy