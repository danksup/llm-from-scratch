import copy
from typing import Any, Literal, overload

import engine.attention as attn
from engine.moe import MoE
import engine.backend as nx
import engine.initializers as init
import engine.optimizer as optim
from engine.rmsnorm import RMSNorm
from engine.dataloader import DataLoader
from engine.embedding import Embedding
from engine.losses import cross_entropy, cross_entropy_gradient
from engine.transformer_block import TransformerBlock
from helper.singleton import sleep, colorize
import warnings
from helper.logger import Logger
from helper.validate_and_raise import validate_choice

optimizers = optim.Adam | optim.AdamW | optim.SGD

default_block_configs = {
    "ff_hidden_width": 1024,
    "ff_n_experts":24,
    "ff_topk":2,
    "ff_cf":1.25,
    "ff_moe_lambda":1e-2,
    "ff_init":"glorot_uniform",
    "attn_type":"chunked",
    "attn_variant":"gqa",
    "attn_n_heads":16,
    "attn_init":"glorot_uniform",
}

ATTN_TYPE = {
    "chunked": {"attn": attn.AttentionChunked, "attn_chunk_size":32},
    "full": {"attn":attn.AttentionFull,},
}

ATTN_VARIANT = {
    "gqa": {"attn_n_kv_heads":default_block_configs["attn_n_heads"]//2},
    "mqa":{},
    "mha":{}
}

INITIALIZERS = {
    "glorot_normal": init.glorot_normal,
    "glorot_uniform": init.glorot_uniform
}

class Transformer:
    def __init__(self, configs: dict[str, Any] | None = None, blocks:list|None=None, *, embedding:bool|Embedding=False, rmsfinal:bool|RMSNorm=False):
        self.blocks = []
        configs =  {} if configs is None else configs
        self.configs = configs

        self.vocab_size = configs.get("vocab_size", None)
        assert self.vocab_size is not None, "vocab size can't be None"

        self.embed_dim = configs.get("embed_dim", 128)
        self.dtype = configs.get("dtype", nx.float32)
        validate_choice(self.dtype, "dtype", nx.floating_type_str)
        if isinstance(self.dtype, str):
            self.dtype = nx.str_to_dtype[self.dtype]
        if nx.issubdtype(self.dtype, nx.integer):
            raise ValueError(f"please use floating type for dtype initialization, got {self.dtype} instead.")

        self.quantized = configs.get("quantized", False)
        self.quantized = self.quantized.lower() if isinstance(self.quantized, str) else self.quantized
        self.symmetric_quant = True if self.quantized == "symmetric" else False

        self.check_non_finite = configs.get("check_non_finite", True)
        
        validate_choice(self.quantized, "quantized", [True, False, "symmetric"])

        if not isinstance(embedding, Embedding):
            self.embedding = Embedding(self.vocab_size, self.embed_dim, self.dtype, self.quantized, use_symmetric=self.symmetric_quant)
        else:
            self.embedding = embedding
        if not isinstance(self.embedding, Embedding):
            raise ValueError(",")

        if not isinstance(rmsfinal, RMSNorm):
            self.rmsnorm_final = RMSNorm(self.embed_dim)
        else:
            self.rmsnorm_final = rmsfinal
        
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

                overrided = overrided | ATTN_TYPE[overrided["attn_type"]] | ATTN_VARIANT[overrided["attn_variant"]] | this  | override
                overrided.pop('attn')

                attn_type = ATTN_TYPE[attn_type_str]["attn"]

                validate_choice(overrided["attn_init"], "attn_init", INITIALIZERS)
                validate_choice(overrided["ff_init"], "ff_init", INITIALIZERS)

                check = default_block_configs | ATTN_TYPE[this["attn_type"]] | ATTN_VARIANT[this["attn_variant"]] | ATTN_TYPE[overrided["attn_type"]] | ATTN_VARIANT[overrided["attn_variant"]]
                for config in overrided:
                    validate_choice(config, "block_overrides", check, f"[block {i}]")
                self.individual_block_configs.append(overrided)

                D = self.embed_dim
                H = overrided["ff_hidden_width"]
                attn_init = INITIALIZERS[overrided["attn_init"]]
                E = overrided["ff_n_experts"]
                CF = overrided["ff_cf"]
                topk = overrided["ff_topk"]
                ff_init = INITIALIZERS[overrided["ff_init"]]
                ff_moe_lambda = overrided["ff_moe_lambda"]

                if "attn_chunk_size" in override and override.get("attn_type", None) == "full":
                    raise ValueError(f"[block {i}] attention type of {attn_type_str} doesn't accept \"attn_chunk_size\"")
                n_heads = overrided["attn_n_heads"]
                attn = None
                chunk_size = overrided.get("attn_chunk_size", None)

                head_dim = D // n_heads
                Q_norm = RMSNorm(head_dim)
                K_norm = RMSNorm(head_dim)
                match (attn_type_str, attn_variant):
                    case ("chunked", "gqa"):
                        n_kv_heads = overrided["attn_n_kv_heads"]
                        attn = attn_type(embed_dim=D, n_heads=n_heads, Q_norm=Q_norm, K_norm=K_norm, n_kv_heads=n_kv_heads, chunk_size=chunk_size, dtype=self.dtype, initializer=attn_init, quantized=self.quantized, use_symmetric=self.symmetric_quant)
                    case ("chunked", "mha"):
                        attn = attn_type.multihead(D, n_heads,chunk_size , self.dtype, attn_init, quantized=self.quantized, use_symmetric=self.symmetric_quant, Q_norm=Q_norm, K_norm=K_norm,)
                    case ("chunked", "mqa"):
                        attn = attn_type.multiquery(D, n_heads, chunk_size, self.dtype, attn_init,quantized=self.quantized, use_symmetric=self.symmetric_quant, Q_norm=Q_norm, K_norm=K_norm,)
                    case ("chunked", invalid):
                        raise ValueError(f"[block {i}] invalid variant of \"{invalid}\". valid variants: {", ".join(ATTN_VARIANT)}")
                    case ("full", "gqa"):
                        n_kv_heads = overrided["attn_n_kv_heads"]
                        attn = attn_type(embed_dim=D, n_heads=n_heads, n_kv_heads=n_kv_heads,  dtype=self.dtype, initializer=attn_init,quantized=self.quantized, use_symmetric=self.symmetric_quant, Q_norm=Q_norm, K_norm=K_norm,)
                    case ("full", "mha"):
                        attn = attn_type.multihead(embed_dim=D, n_heads=n_heads,  dtype=self.dtype, initializer=attn_init,quantized=self.quantized, use_symmetric=self.symmetric_quant, Q_norm=Q_norm, K_norm=K_norm,)
                    case ("full", "mqa"):
                        attn = attn_type.multiquery(embed_dim=D, n_heads=n_heads,  dtype=self.dtype, initializer=attn_init,quantized=self.quantized, use_symmetric=self.symmetric_quant, Q_norm=Q_norm, K_norm=K_norm,)
                    case ("full", invalid):
                        raise ValueError(f"[block {i}] invalid variant of \"{invalid}\". valid variants: {", ".join(ATTN_VARIANT)}")
                    case _:
                        raise ValueError(f"[block {i}] invalid variant of \"{attn_variant}\". valid variants: {", ".join(ATTN_VARIANT)}")

                ff = MoE(CF, topk, E, D, H,ff_moe_lambda, dtype=self.dtype, initializer=ff_init, quantized=self.quantized, as_symmetric=self.symmetric_quant)
                rmsnorm1 = RMSNorm(D)
                rmsnorm2 = RMSNorm(D)
                transformer_block = TransformerBlock(attn, ff, rmsnorm1, rmsnorm2)
               
                self.blocks.append(transformer_block)
        else:
            self.blocks = blocks
            if not self.blocks:
                raise ValueError("this transformer doesnt have any block.")

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        self.logger:Logger = kwds["logger"]
        return self
        
    def __str__(self) -> str:
        return self.get_configs_str()

    def count_params(self) -> int:
        """
        whole architecture number of (trainable) params
        """
        total = 0
        for i in self.blocks:
            total += i.count_param(quantized=self.quantized, use_symmetric=self.symmetric_quant)

        embedding_size = self.embedding.lookup_table.size
        if self.quantized and not self.symmetric_quant:
            embedding_size *= 4
        total += embedding_size
        total += self.rmsnorm_final.gamma.size
        return total

    def forward(self, inputs:Any, return_cache= True, is_training=True) -> Any:
        '''
        inputs = list of inputs
        '''
        output = inputs.astype(self.dtype)
        all_masks = []
        all_caches = []
        total_router_loss = nx.array(0.0, dtype=nx.float32)
        histograms = [None for _ in range(len(self.blocks))]
        for idx, block in enumerate(self.blocks):
            try:
                output = output.astype(self.dtype)
                B,T,_ = output.shape
                epsilon = block.rmsnorm1.epsilon
                gamma1 = block.rmsnorm1.gamma
                gamma2 = block.rmsnorm2.gamma

                P = nx.array(0.1, dtype=self.dtype)
                attn_str = block.attention.self_type()

                if attn_str == "chunked":
                    chunk_size = block.attention.chunk_size
                    chunk_size = min(chunk_size, T-1)
                    if block.causal_mask is None or block.causal_mask.shape != (T, chunk_size + 1):
                        block.causal_mask = block.attention.compute_mask()
                elif attn_str == "full":
                    if block.causal_mask is None or block.causal_mask.shape != (T, T):
                        block.causal_mask = block.attention.compute_mask(T)
                elif attn_str == "swa":
                    W = block.attention.W
                    W = min(W, T-1)
                    if block.causal_mask is None or block.causal_mask.shape != (T, W + 1):
                        block.causal_mask = block.attention.compute_mask(T)

                attn_params = block.attention.Wqkv, block.attention.Wo, block.attention.Q_norm.gamma, block.attention.K_norm.gamma
                ff_params = block.ff.Wcombined, block.ff.Wout, block.ff.router
                scales = (block.attention.scales + block.attention.biases, block.ff.scales + block.ff.biases)
                ff_out ,masks, caches, router_loss, normalized_histogram = block._forward(output, block.causal_mask, attn_str ,block.attention.configs, attn_params, block.ff.configs, ff_params, epsilon, gamma1, gamma2, P, is_training, scales, use_symmetric=self.symmetric_quant)
                total_router_loss += router_loss
                output = ff_out
                all_masks.append(masks)
                all_caches.append(caches)
                histograms[idx] = nx.zeros_like(normalized_histogram)
                histograms[idx] += normalized_histogram
            except TypeError as e:
                print(f"[block {idx}] TypeError")
                raise TypeError(e)
            except ValueError as e:
                print(f"[block {idx}] ValueError")
                raise ValueError(e)

        last_output = output.astype(self.dtype)
        lookup_table = self.embedding.lookup_table
        if self.quantized:
            lookup_table = nx.dequantize(lookup_table, self.embedding.table_scale,self.embedding.bias, self.dtype, regular=self.symmetric_quant)

        rmsnorm_out, rms_cache = self.rmsnorm_final._forward(last_output, self.rmsnorm_final.gamma,self.rmsnorm_final.epsilon)
        scores = rmsnorm_out.astype(self.dtype) @ lookup_table.T
        del lookup_table

        if return_cache:
            all_caches += rms_cache,
            return scores, rmsnorm_out, all_masks, all_caches, total_router_loss, histograms
        return scores, total_router_loss

    def backward(self, err_signal:Any,  all_masks, all_caches:list) -> Any:
        '''
        Args:
            traces error contribution and then optimize
        '''
        current_grad = err_signal
        rmsnorm_final_cache = all_caches.pop()
        current_grad, rmsnorm_final_d_gamma = self.rmsnorm_final._backward(current_grad, rmsnorm_final_cache, self.rmsnorm_final.gamma)
        self.rmsnorm_final.d_gamma = rmsnorm_final_d_gamma if getattr(self.rmsnorm_final, "d_gamma", None) is None else self.rmsnorm_final.d_gamma + rmsnorm_final_d_gamma
        for block, masks,caches in zip(self.blocks[::-1], all_masks[::-1],all_caches[::-1]):
            current_grad = current_grad.astype(self.dtype)
            _,T,_ = current_grad.shape
            caches_attn, caches_ff, caches_rmsnorm1, caches_rmsnorm2 = caches
            mask1, mask2 = masks
            scaled_lambda = block.ff.LAMBDA * self.gradient_scale
            moe_configs = block.ff.cf, block.ff.n_experts, block.ff.hidden_width, block.ff.router, scaled_lambda
            P = nx.array(0.1, dtype=self.dtype)

            ff_params = (block.ff.Wout, block.ff.Wcombined)
            attn_str = block.attention.self_type()
            attn_configs = block.attention.configs
            attn_params = block.attention.Wqkv, block.attention.Wo, block.attention.Q_norm.gamma, block.attention.K_norm.gamma
            scales = (block.attention.scales + block.attention.biases, block.ff.scales + block.ff.biases)
            dx, dWout, dWcombined, d_router, dWqkv, dWo, d_gamma1, d_gamma2,Q_norm_d_gamma, K_norm_d_gamma = block._backward(current_grad, mask1=mask1, mask2=mask2, p=P, attention=attn_str,
                                                                caches_attn=caches_attn, caches_ff=caches_ff, caches_rmsnorm1=caches_rmsnorm1, caches_rmsnorm2=caches_rmsnorm2,
                                                                attn_configs = attn_configs, attn_params=attn_params, gamma1=block.rmsnorm1.gamma, gamma2=block.rmsnorm2.gamma, ff_params=ff_params, moe_configs=moe_configs, gradient_scale=self.gradient_scale, quantization=scales, use_symmetric=self.symmetric_quant)


            block.ff.dWout += dWout
            block.ff.dWcombined += dWcombined
            block.ff.d_router += d_router

            block.attention.dWqkv += dWqkv
            block.attention.dWo += dWo
            block.attention.Q_norm.d_gamma +=  Q_norm_d_gamma
            block.attention.K_norm.d_gamma +=  K_norm_d_gamma

            block.rmsnorm1.d_gamma += d_gamma1
            block.rmsnorm2.d_gamma += d_gamma2

            current_grad = dx

        return current_grad

    def eval_networks(self, others:list|None = None, include_gradients:bool=True, optimizer:optimizers|None=None):
        to_eval = []
        if others is not None:
            to_eval.extend(others)

        for layer_obj, param_name, _ in self.get_weights():
            weight_obj = getattr(layer_obj, param_name)
            to_eval.append(weight_obj)

        if include_gradients:
            for layer_obj, param_name,_ in self.get_gradients():
                dweight_obj = getattr(layer_obj, param_name)
                to_eval.append(dweight_obj)
        else:
            if optimizer is not None:
                to_eval.append(optimizer.lr)
                if hasattr(optimizer, "state"):
                    to_eval.append(optimizer.state)
                if hasattr(optimizer, "masters"):
                    to_eval.append(optimizer.masters) #type:ignore

        nx.eval(*to_eval)

    def get_block_weights(self, layers, params):
        for i, block in enumerate(self.blocks):
            for layer_i, layer in enumerate(layers):
                layer_obj = getattr(block, layer)
                for param_name in params[layer_i]:
                    if isinstance(param_name, list):
                        attr, param_name_list = param_name
                        layer_obj_list = getattr(layer_obj, attr, None)
                        if layer_obj_list is not None:
                            if getattr(layer_obj_list, param_name_list, None) is not None:
                                yield layer_obj_list, param_name_list, f"{i}.{layer}.{attr}.{param_name_list}"
   
                    else:
                        if getattr(layer_obj, param_name, None) is not None:
                                yield layer_obj, param_name, f"{i}.{layer}.{param_name}"

    def get_weights(self, *, block_only:bool=False):
        if not block_only:
            if getattr(self.embedding, "lookup_table", None) is not None:
                yield self.embedding, "lookup_table", "embedding.lookup_table"

            if getattr(self.rmsnorm_final, "gamma", None) is not None:
                yield self.rmsnorm_final, "gamma", "rmsnorm_final.gamma"

        layers = ["ff", "attention", "rmsnorm1", "rmsnorm2"]
        weights = [["router", "Wcombined", "Wout"],[ "Wqkv", "Wo", ["Q_norm", "gamma"], ["K_norm", "gamma"]],[ "gamma"],[ "gamma"]]
        for a,b,c in self.get_block_weights(layers, weights):
            yield a,b,c

    def get_gradients(self):
        if getattr(self.embedding, "d_lookup_table", None) is not None:
            yield self.embedding, "d_lookup_table", "embedding.d_lookup_table"

        if getattr(self.rmsnorm_final, "d_gamma", None) is not None:
            yield self.rmsnorm_final, "d_gamma", "rmsnorm_final.d_gamma"

        layers = ["ff", "attention", "rmsnorm1", "rmsnorm2"]
        dweights = [ ["d_router", "dWcombined","dWout"], ["dWo", "dWqkv", ["K_norm", "d_gamma"], ["Q_norm", "d_gamma"]],[ "d_gamma"],["d_gamma"]]
        for a,b,c in self.get_block_weights(layers, dweights): 
            yield a,b,c

    def gradient_clipping_factor(self, microbatch_size, max_norm=nx.float_32(0.5)):
        summed = nx.array(0, nx.float32)

        for layer, param_name, _ in self.get_gradients(): 
            param = getattr(layer, param_name) 
            if param_name == "d_lookup_table":
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

        weights = self.get_weights()
        for layer_obj, param_name, name in weights:
            weight = getattr(layer_obj, param_name)
            if not nx.isfinite(weight).all():
                non_finite = True
                block_i = name.split(".")[0]
                try:
                    block_i = int(block_i)
                    failures[block_i].append(name)
                except ValueError:
                    failures["-"].append(name)

        gradients = self.get_gradients()
        for layer_obj, param_name,name  in gradients:
            gradient = getattr(layer_obj, param_name)
            if not nx.isfinite(gradient).all():
                non_finite = True
                block_i = name.split(".")[0]
                try:
                    block_i = int(block_i)
                    failures[block_i].append(name)
                except ValueError:
                    failures["-"].append(name)

        for key, val in failures.items():
            texts += f"block_{key}: {val}"
        return non_finite, texts

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

        for contexts, next_tokens in dataloader.prefetch_batch(dataloader.train_files):
            contexts = nx.array(contexts).reshape(dataloader.batch_size, dataloader.context_size)
            next_tokens = nx.array(next_tokens).reshape(dataloader.batch_size, dataloader.context_size)

            if step >= max_step:
                break

            embedded = self.embedding.forward(contexts)  # shape (batch, context_size, embed_dim)
            batch_scores, last_output, all_masks, all_caches, total_aux_loss, histograms = self.forward(embedded)

            if total_histograms is None:
                total_histograms = histograms
            else:
                for i in range(len(self.blocks)):
                    total_histograms[i] += histograms[i]

            loss = cross_entropy(batch_scores, next_tokens)
            loss = nx.mean(loss)  + total_aux_loss

            batch_gradient = cross_entropy_gradient(batch_scores, next_tokens)
            batch_gradient /= (batch_gradient.shape[0] * batch_gradient.shape[1])
            scaled_batch_gradient = batch_gradient * self.gradient_scale

            batch_gradient = scaled_batch_gradient.astype(self.dtype)

            lookup_table = self.embedding.lookup_table
            if self.quantized:
                lookup_table = nx.dequantize(lookup_table, self.embedding.table_scale,biases=self.embedding.bias, dtype=self.dtype, regular=self.symmetric_quant)

            block_gradient =  batch_gradient @ lookup_table #dtype

            d_table = batch_gradient.reshape(-1, self.vocab_size).T @ last_output.reshape(-1, self.embed_dim)
            d_table = d_table.astype(nx.float32) / self.gradient_scale

            current_grad = self.backward(block_gradient, all_masks, all_caches)
            current_grad = current_grad.astype(nx.float32) / self.gradient_scale

            embedding_gradient = nx.zeros_like(lookup_table, dtype=nx.float32)
            embedding_gradient = nx.add_at(embedding_gradient, contexts, current_grad)

            total_embedding_gradient = embedding_gradient + d_table
            self.embedding.d_lookup_table += total_embedding_gradient

            total_loss += loss * next_tokens.size
            count += next_tokens.size
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

                for i,block in enumerate(self.blocks):
                    
                    dWqkv = block.attention.dWqkv.astype(nx.float32) / self.gradient_scale / microbatch_size * gscale
                    dWo = block.attention.dWo.astype(nx.float32) / self.gradient_scale / microbatch_size * gscale
                    Q_norm_d_gamma = block.attention.Q_norm.d_gamma.astype(nx.float32) / self.gradient_scale / microbatch_size * gscale
                    K_norm_d_gamma = block.attention.K_norm.d_gamma.astype(nx.float32) / self.gradient_scale / microbatch_size * gscale

                    dWcombined = block.ff.dWcombined.astype(nx.float32) / self.gradient_scale / microbatch_size * gscale
                    dWout = block.ff.dWout.astype(nx.float32) / self.gradient_scale / microbatch_size * gscale
                    d_router = block.ff.d_router.astype(nx.float32) / self.gradient_scale / microbatch_size * gscale

                    d_gamma1 = block.rmsnorm1.d_gamma.astype(nx.float32) / self.gradient_scale / microbatch_size* gscale
                    d_gamma2 = block.rmsnorm2.d_gamma.astype(nx.float32) / self.gradient_scale / microbatch_size* gscale

                    if self.quantized:
                        Wqkv = nx.dequantize(block.attention.Wqkv, block.attention.scales[0], block.attention.biases[0], regular=self.symmetric_quant)
                        Wo = nx.dequantize(block.attention.Wo, block.attention.scales[1], block.attention.biases[1], regular=self.symmetric_quant)
                        Wcombined = nx.dequantize(block.ff.Wcombined, block.ff.scales[0], block.ff.biases[0], regular=self.symmetric_quant)
                        Wout = nx.dequantize(block.ff.Wout, block.ff.scales[1], block.ff.biases[1], regular=self.symmetric_quant)
                    else:
                        Wqkv = block.attention.Wqkv.astype(nx.float32)
                        Wo = block.attention.Wo.astype(nx.float32)
                        Wcombined = block.ff.Wcombined.astype(nx.float32)
                        Wout = block.ff.Wout.astype(nx.float32)

                    all_network_params.extend(
                        [(f"Wqkv_{i}", Wqkv, dWqkv, True),
                        (f"Wo_{i}", Wo, dWo, True),
                        (f"Q_norm_gamma_{i}", block.attention.Q_norm.gamma, Q_norm_d_gamma, False),
                        (f"K_norm_gamma_{i}", block.attention.K_norm.gamma, K_norm_d_gamma, False),
                        (f"ff_wcombined_{i}", Wcombined,dWcombined, True),
                        (f"ff_wout_{i}", Wout, dWout, True),
                        (f"ff_router_{i}", block.ff.router.astype(nx.float32), d_router, True),
                        (f"rmsnorm1_gamma_{i}", block.rmsnorm1.gamma.astype(nx.float32), d_gamma1, False),
                        (f"rmsnorm2_gamma_{i}", block.rmsnorm2.gamma.astype(nx.float32), d_gamma2, False)])
                    del Wqkv, Wo, Wcombined, Wout
                    del dWqkv, dWo, dWcombined, dWout, d_router, d_gamma1, d_gamma2
                    block.zeroes_gradient()

                lookup_table = self.embedding.lookup_table.astype(nx.float32)
                if self.quantized:
                    lookup_table = nx.dequantize(lookup_table, self.embedding.table_scale, self.embedding.bias, regular=self.symmetric_quant)
                all_network_params.extend([("embedding",lookup_table, self.embedding.d_lookup_table / microbatch_size * gscale, False)])

                if getattr(self.rmsnorm_final, "d_gamma", None) is not None:
                    d_gamma = self.rmsnorm_final.d_gamma.astype(nx.float32) / self.gradient_scale / microbatch_size * gscale #type:ignore
                    all_network_params.extend([("rmsnorm_final", self.rmsnorm_final.gamma.astype(nx.float32), d_gamma, False)])
                    del d_gamma
                    self.rmsnorm_final.zeroes_gradient()

                optimized = optimizer.step_many(all_network_params, max_step, total_epoch)

                for i,block in enumerate(self.blocks):
                    if self.quantized:
                        Wqkv = optimized[f"Wqkv_{i}"]
                        block.attention.Wqkv, wqkv_scale, wqkv_bias = nx.quantize(Wqkv, regular=self.symmetric_quant)
                        Wo = optimized[f"Wo_{i}"]
                        block.attention.Wo, wo_scale, wo_bias = nx.quantize(Wo, regular=self.symmetric_quant)
                        block.attention.scales = (wqkv_scale, wo_scale)
                        block.attention.biases = (wqkv_bias, wo_bias)

                        Wcombined = optimized[f"ff_wcombined_{i}"]
                        block.ff.Wcombined, wcombined_scale, wcombined_bias = nx.quantize(Wcombined, regular=self.symmetric_quant)
                        Wout = optimized[f"ff_wout_{i}"]
                        block.ff.Wout, wout_scale, wout_bias = nx.quantize(Wout, regular=self.symmetric_quant)
                        block.ff.scales = (wcombined_scale, wout_scale)
                        block.ff.biases = (wcombined_bias, wout_bias)

                        del Wqkv,Wo,Wcombined,Wout,wcombined_scale,wcombined_bias,wqkv_scale,wqkv_bias,wout_scale,wout_bias,wo_scale,wo_bias
                    else:
                        block.attention.Wqkv = optimized[f"Wqkv_{i}"].astype(self.dtype)
                        block.attention.Wo = optimized[f"Wo_{i}"].astype(self.dtype)
                        block.ff.Wcombined = optimized[f"ff_wcombined_{i}"].astype(self.dtype)
                        block.ff.Wout = optimized[f"ff_wout_{i}"].astype(self.dtype)

                    block.ff.router = optimized[f"ff_router_{i}"]
                    block.rmsnorm1.gamma = optimized[f"rmsnorm1_gamma_{i}"]
                    block.rmsnorm2.gamma = optimized[f"rmsnorm2_gamma_{i}"]
                    block.attention.Q_norm.gamma = optimized[f"Q_norm_gamma_{i}"]
                    block.attention.K_norm.gamma = optimized[f"K_norm_gamma_{i}"]

                if self.quantized:
                    embedding = optimized[f"embedding"]
                    self.embedding.lookup_table, self.embedding.table_scale, self.embedding.bias = nx.quantize(embedding, regular=self.symmetric_quant)
                    del embedding
                else:
                    self.embedding.lookup_table = optimized["embedding"].astype(self.dtype)
                self.embedding.zeroes_gradient()

                self.rmsnorm_final.gamma = optimized["rmsnorm_final"]
                step += 1

                #TODO: fix this hardcoding
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
        count = 0
        step_counter = 0

        if not dataloader.validation_files:
            return None

        for contexts, next_tokens in dataloader.prefetch_batch(dataloader.validation_files):
            if isinstance(val_step, int) and step_counter >= val_step:
                break
            contexts = nx.array(nx.tolist(contexts), nx.int32)
            next_tokens = nx.array(nx.tolist(next_tokens), nx.int32)

            embedded = self.embedding.forward(contexts)
            batch_validation_scores, total_router_loss = self.forward(embedded, False, False)

            val_loss = cross_entropy(batch_validation_scores, next_tokens)
            val_loss = nx.mean(val_loss)  + total_router_loss
            total_loss += val_loss * next_tokens.size
            count += next_tokens.size
            step_counter += 1

            nx.eval(*[val_loss, total_loss])

        if count == 0:
            return None

        final_loss = total_loss / count
        return final_loss.item()

    def inference(self, context:Any, max_cache_len, all_caches = None,  position = 0, *, use_symmetric) -> Any:
        if all_caches is None:
            all_caches = [(None, None) for _ in range(len(self.blocks))]
        as_symmetric = self.symmetric_quant or use_symmetric
        output = self.embedding.forward(context)
        for idx, block in enumerate(self.blocks):
            cached_k, cached_v = all_caches[idx]
            ff_out, cache_k, cache_v = block.inference_forward(output,max_cache_len, cached_k, cached_v, position, use_symmetric=as_symmetric)
            all_caches[idx] = (cache_k, cache_v)
            output = ff_out

        rmsfinal_out, _ = self.rmsnorm_final._forward(output, self.rmsnorm_final.gamma, self.rmsnorm_final.epsilon)
        if self.quantized:
            scores = nx.quantized_matmul(rmsfinal_out, self.embedding.lookup_table, self.embedding.table_scale, self.embedding.bias, transpose=True, regular=as_symmetric) #type:ignore
        else:
            scores = rmsfinal_out @ self.embedding.lookup_table.T

        return scores, all_caches

    def get_configs(self):
        configs = {}
        configs["vocab_size"] = self.vocab_size
        configs["embed_dim"] = self.embed_dim
        configs["dtype"] = nx.dtype_to_srt[self.dtype]
        configs["quantized"] =  self.quantized
        configs["symmetric_quant"] =  self.symmetric_quant
        configs["gradient_scale"] = self.gradient_scale
        return configs

    def get_configs_str(self):
        configs = ""
        for k,v in self.get_configs().items():
            if k not in ["dtype", "symmetric_quant"]:
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

    @overload
    def get_all_weights(self, flatten: Literal[False] = False) -> dict[int, dict[str, dict[str, nx.ArrayLike]]]: ...

    @overload
    def get_all_weights(self, flatten: Literal["dict"]) -> dict[str, nx.ArrayLike]: ...

    @overload
    def get_all_weights(self, flatten: Literal[True]) -> list[Any]: ...

    def get_all_weights(self, flatten:bool|Literal["dict"]=False) -> dict[int,dict[str,dict[str,nx.ArrayLike]]] | dict[str,nx.ArrayLike] | list[Any]:
            layers = ["ff", "attention", "rmsnorm1", "rmsnorm2"]
            weights = [["router", "Wcombined", "Wout"],[ "Wqkv", "Wo"],[ "gamma"],[ "gamma"]]

            if not flatten:
                all_weights = {}
                for idx, block in enumerate(self.blocks):
                    all_weights[idx] = {}
                    for layer_i, layer in enumerate(layers):
                        layer_ = getattr(block, layer)
                        all_weights[idx][layer] = {}
                        for weight in weights[layer_i]:
                            weight_ = getattr(layer_, weight)
                            all_weights[idx][layer][weight] = weight_
            else:
                if flatten == "dict":
                    all_weights = {}
                    weights = self.get_weights(block_only=True)
                    for layer_obj, param_name, name in weights:
                        weight = getattr(layer_obj, param_name)
                        all_weights[name] = weight
                else:
                    a:dict[str,nx.ArrayLike] = self.get_all_weights("dict") #type:ignore
                    return list(a.values())

            return all_weights

    def get_quant_params(self):
        quant_params = {}
        layers = ["ff", "attention"]
        weights = [["Wcombined", "Wout"],[ "Wqkv", "Wo"]]
        quants = ["scales", "biases"]

        for idx, block in enumerate(self.blocks):
            for layer_i, layer in enumerate(layers):
                layer_ = getattr(block, layer)
                for quant in quants:
                    quant_attr = getattr(layer_, quant)
                    for attr_i, attr in enumerate(quant_attr):
                        quant_params[f"{idx}.{layer}.{weights[layer_i][attr_i]}.{quant}"] = attr

        return quant_params

    def get_block_configs(self):
        configs = {}
        for idx, block in enumerate(self.blocks):
            configs[idx] = block.get_configs()

        return configs

    def copy(self) -> "Transformer":
        configs = copy.deepcopy(self.configs)
        block_copy = []
        for block in self.blocks:
            block_copy.append(block.copy())

        embedding_copy = self.embedding.copy()
        rmsfinal_copy = self.rmsnorm_final.copy()
        transformer_copy = Transformer(configs, block_copy, embedding=embedding_copy, rmsfinal=rmsfinal_copy)

        return transformer_copy