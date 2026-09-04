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
from helper.singleton import sleep
from helper.validate_and_raise import validate_choice
import warnings


optimizers = optim.Adam | optim.AdamW | optim.SGD

default_block_configs = {
    "ff_hidden_width": 1024,
    "ff_n_experts":24,
    "ff_topk":2,
    "ff_cf":1.25,
    "ff_moe_lambda":1e-2,
    "ff_init":"glorot_uniform",
    "attn_type":"swa",
    "attn_variant":"gqa",
    "attn_n_heads":16,
    "attn_init":"glorot_uniform",
}

ATTN_TYPE = {
    "swa": {"attn": attn.AttentionSWA, "attn_windows":32},
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
    def __init__(self, configs: dict[str, Any] | None = None, blocks:list|None=None, *, embedding:bool|Embedding=False):
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
        
        # assert self.quantized in [True, False, "symmetric"], f"True= mlx:affine, else symmetric; symmetric= use symmetric regardless of backend type; False = floats; got {self.quantized} of type {type(self.quantized)} instead."
        validate_choice(self.quantized, "quantized", [True, False, "symmetric"])

        if not isinstance(embedding, Embedding):
            self.embedding = Embedding(self.vocab_size, self.embed_dim, self.dtype, self.quantized, use_symmetric=self.symmetric_quant)
        else:
            self.embedding = embedding
        if not isinstance(self.embedding, Embedding):
            raise ValueError(",")
        
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
                # assert attn_variant in ATTN_VARIANT, f"[block {i}] invalid input: \"{attn_variant}\" of type {type(attn_variant)} for attn_variant. valid attn_variant: {", ".join(ATTN_VARIANT.keys())}"
                validate_choice(attn_variant, "attn_variant", ATTN_VARIANT)

                attn_type_str = overrided["attn_type"]
                validate_choice(attn_type_str, "attn_type", ATTN_TYPE)

                overrided = overrided | ATTN_TYPE[overrided["attn_type"]] | ATTN_VARIANT[overrided["attn_variant"]] | this  | override
                overrided.pop('attn')

                attn_type = ATTN_TYPE[attn_type_str]["attn"]

                # assert overrided["attn_init"] in INITIALIZERS, f"your configs contain invalid input: {overrided["attn_init"]} of type {type(overrided["attn_init"])}. expected for this config: {", ".join(list(INITIALIZERS))}"
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

                if "attn_windows" in override and override.get("attn_type", None) == "full":
                    raise ValueError(f"[block {i}] attention type of {attn_type_str} doesn't accept \"attn_windows\"")
                n_heads = overrided["attn_n_heads"]
                attn = None
                W = overrided.get("attn_windows", None)
                match (attn_type_str, attn_variant):
                    case ("swa", "gqa"):
                        n_kv_heads = overrided["attn_n_kv_heads"]
                        #def __init__(self,embed_dim:int, n_heads:int, n_kv_heads:int=-1, W=8, dtype:Any=nx.float16, initializer:Callable=initializer.glorot_uniform)
                        attn = attn_type(embed_dim=D, n_heads=n_heads, n_kv_heads=n_kv_heads, W=W, dtype=self.dtype, initializer=attn_init, quantized=self.quantized, use_symmetric=self.symmetric_quant)
                    case ("swa", "mha"):
                        attn = attn_type.multihead(D, n_heads, W, self.dtype, attn_init, quantized=self.quantized, use_symmetric=self.symmetric_quant)
                    case ("swa", "mqa"):
                        attn = attn_type.multiquery(D, n_heads, W, self.dtype, attn_init,quantized=self.quantized, use_symmetric=self.symmetric_quant)
                    case ("swa", invalid):
                        raise ValueError(f"[block {i}] invalid variant of \"{invalid}\". valid variants: {", ".join(ATTN_VARIANT)}")
                    case ("full", "gqa"):
                        n_kv_heads = overrided["attn_n_kv_heads"]
                        attn = attn_type(embed_dim=D, n_heads=n_heads, n_kv_heads=n_kv_heads,  dtype=self.dtype, initializer=attn_init,quantized=self.quantized, use_symmetric=self.symmetric_quant)
                    case ("full", "mha"):
                        attn = attn_type.multihead(embed_dim=D, n_heads=n_heads,  dtype=self.dtype, initializer=attn_init,quantized=self.quantized, use_symmetric=self.symmetric_quant)
                    case ("full", "mqa"):
                        attn = attn_type.multiquery(embed_dim=D, n_heads=n_heads,  dtype=self.dtype, initializer=attn_init,quantized=self.quantized, use_symmetric=self.symmetric_quant)
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

                if attn_str == "swa":
                    W = block.attention.W
                    assert W is not None, f"[block {idx}] W is None"
                    W = min(W, T-1)
                    if block.causal_mask is None or block.causal_mask.shape != (T, W + 1):
                        block.causal_mask = block.attention.compute_mask(W, T)
                elif attn_str == "full":
                    if block.causal_mask is None or block.causal_mask.shape != (T, T):
                        block.causal_mask = block.attention.compute_mask(T)
                attn_params = block.attention.Wqkv, block.attention.Wo
                ff_params = block.ff.Wcombined, block.ff.Wout, block.ff.router
                scales = (block.attention.scales + block.attention.biases, block.ff.scales + block.ff.biases)
                ff_out ,masks, caches, router_loss, normalized_histogram = block._forward(output, block.causal_mask, attn_str ,block.attention.configs, attn_params, block.ff.configs, ff_params, epsilon, gamma1, gamma2, P, is_training, scales, use_symmetric=self.symmetric_quant)
                total_router_loss += router_loss
                output = ff_out
                all_masks.append(masks)
                all_caches.append(caches)
                histograms[idx] = nx.zeros_like(normalized_histogram)
                histograms[idx] += normalized_histogram
                # self.eval_networks()
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
        scores = last_output @ lookup_table.T
        del lookup_table

        if return_cache:
            return scores, last_output, all_masks, all_caches, total_router_loss, histograms
        return scores, total_router_loss

    def backward(self, err_signal:Any,  all_masks,all_caches) -> Any:
        '''
        Args:
            traces error contribution and then optimize
        '''
        current_grad = err_signal
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
            attn_params = block.attention.Wqkv, block.attention.Wo
            scales = (block.attention.scales + block.attention.biases, block.ff.scales + block.ff.biases)
            dx, dWout, dWcombined, d_router, dWqkv, dWo, d_gamma1, d_gamma2 = block._backward(current_grad, mask1=mask1, mask2=mask2, p=P, attention=attn_str,
                                                                caches_attn=caches_attn, caches_ff=caches_ff, caches_rmsnorm1=caches_rmsnorm1, caches_rmsnorm2=caches_rmsnorm2,
                                                                attn_configs = attn_configs, attn_params=attn_params, gamma1=block.rmsnorm1.gamma, gamma2=block.rmsnorm2.gamma, ff_params=ff_params, moe_configs=moe_configs, quantization=scales, use_symmetric=self.symmetric_quant)


            block.ff.dWout = dWout if getattr(block.ff, "dWout", None) is None else block.ff.dWout + dWout
            block.ff.dWcombined = dWcombined if getattr(block.ff, "dWcombined", None) is None else block.ff.dWcombined + dWcombined
            block.ff.d_router = d_router if getattr(block.ff, "d_router", None) is None else block.ff.d_router + d_router

            block.attention.dWqkv = dWqkv if getattr(block.attention, "dWqkv", None) is None else block.attention.dWqkv + dWqkv
            block.attention.dWo = dWo if getattr(block.attention, "dWo", None) is None else block.attention.dWo + dWo

            block.rmsnorm1.d_gamma = d_gamma1 if getattr(block.rmsnorm1, "d_gamma", None) is None else block.rmsnorm1.d_gamma + d_gamma1
            block.rmsnorm2.d_gamma = d_gamma2 if getattr(block.rmsnorm2, "d_gamma", None) is None else block.rmsnorm2.d_gamma + d_gamma2

            current_grad = dx

        return current_grad

    def eval_networks(self, others:list|None = None, include_gradients:bool=True, optimizer:optimizers|None=None):
        to_eval = []
        if others is not None:
            to_eval.extend(others)

        for block in self.blocks:
            to_eval.append(block.attention.Wqkv)
            to_eval.append(block.attention.Wo)
            to_eval.append(block.ff.Wcombined)
            to_eval.append(block.ff.Wout)
            to_eval.append(block.ff.router)
            to_eval.append(block.rmsnorm1.gamma)
            to_eval.append(block.rmsnorm2.gamma)

            if include_gradients:
                to_eval.append(block.attention.dWqkv)
                to_eval.append(block.attention.dWo)
                to_eval.append(block.ff.dWcombined)
                to_eval.append(block.ff.dWout)
                to_eval.append(block.ff.d_router)
                to_eval.append(block.rmsnorm1.d_gamma)
                to_eval.append(block.rmsnorm2.d_gamma)
            else:
                if optimizer is not None:
                    to_eval.append(optimizer.lr)
                    if hasattr(optimizer, "state"):
                        to_eval.append(optimizer.state)
                    if hasattr(optimizer, "masters"):
                        to_eval.append(optimizer.masters) #type:ignore

        nx.eval(*to_eval)

    
    def non_finite_check(self):
        # nan_weights = []
        texts = ""
        layers = ["ff", "attention", "rmsnorm1", "rmsnorm2"]
        weights = [["router", "Wcombined", "Wout"],[ "Wqkv", "Wo"],[ "gamma"],[ "gamma"]]
        dweights = [ ["d_router", "dWcombined","dWout"], ["dWo", "dWqkv"],[ "d_gamma"],["d_gamma"]]
        for idx, block in enumerate(self.blocks):
            text = f"block{idx}: "
            ltext = len(text)
            for layer_i, layer in enumerate(layers):
                layer_ = getattr(block, layer)
                for weight in weights[layer_i]:
                    weight_ = getattr(layer_, weight)
                    if not nx.isfinite(weight_).all():
                        text += f"{layer}_{weight}"

                for dweight in dweights[layer_i]:
                    dweight_ = getattr(layer_, dweight)
                    if not nx.isfinite(dweight_).all():
                        text += f"{layer}_{dweight} "
            if ltext == len(text):
                continue
            texts += f"{text}\n"
        return texts

    def reset_gradient(self):
        layers = ["ff", "attention", "rmsnorm1", "rmsnorm2"]
        dweights = [ ["d_router", "dWcombined","dWout"], ["dWo", "dWqkv"],[ "d_gamma"],["d_gamma"]]
        for block in self.blocks:
            for layer_i, layer in enumerate(layers):
                layer_ = getattr(block, layer)
                for dweight in dweights[layer_i]:
                    if hasattr(layer_, dweight):
                        delattr(layer_, dweight)

    def train(self, dataloader:DataLoader, optimizer:optimizers, total_epoch:int, max_step:int=50000, eval_every:int=5, microbatch_size:int=16):
        total_loss = nx.float_32(0.0)
        count = 0
        microstep = 0
        step = 0
        total_histograms = None
        embed_acc = nx.zeros((self.vocab_size, self.embed_dim), nx.float32)
        clean_step = 0

        for contexts, next_tokens in dataloader.prefetch_batch(dataloader.train_files):
            contexts = nx.array(nx.tolist(contexts), nx.int32)
            next_tokens = nx.array(nx.tolist(next_tokens), nx.int32)

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
            embed_acc += total_embedding_gradient

            total_loss += loss * next_tokens.size
            count += next_tokens.size
            microstep += 1

            if microstep % eval_every == 0 or microstep == microbatch_size:
                to_eval = [total_loss, self.embedding.lookup_table, embed_acc, total_histograms]
                self.eval_networks(to_eval)

                if self.check_non_finite:
                    gradient_mean = nx.mean(current_grad)
                    if not nx.isfinite(loss).item() or not nx.isfinite(gradient_mean).item():
                        if self.gradient_scale <= 1:
                            raise FloatingPointError("worthless")
                        
                        forward_nan = nx.isnan(loss)
                        forward_inf = nx.isinf(loss)
                        nan_weights = self.non_finite_check()

                        backward_nan = nx.isnan(gradient_mean)
                        backward_inf = nx.isinf(gradient_mean)

                        warnings.warn(f"[NON-FINITE step: {step}] non finite loss at microstep {microstep}. isnan forward/backward: {forward_nan}/{backward_nan} | isinf forward/backward: {forward_inf}/{backward_inf} |\n non-finite weights:\n{nan_weights}", UserWarning)
                        self.gradient_scale = max(1, self.gradient_scale // 2)
                        microstep = 0
                        total_loss = nx.float_32(0)
                        count = 0
                        clean_step = 0
                        embed_acc = nx.zeros_like(embed_acc)
                        total_histograms = None
                        self.reset_gradient()
                        continue                        

            if microstep == microbatch_size:
                all_network_params = []
                for i,block in enumerate(self.blocks):
                    dWqkv = block.attention.dWqkv.astype(nx.float32) / self.gradient_scale / microbatch_size
                    dWo = block.attention.dWo.astype(nx.float32) / self.gradient_scale / microbatch_size
                    dWcombined = block.ff.dWcombined.astype(nx.float32) / self.gradient_scale / microbatch_size
                    dWout = block.ff.dWout.astype(nx.float32) / self.gradient_scale / microbatch_size
                    d_router = block.ff.d_router.astype(nx.float32) / self.gradient_scale / microbatch_size
                    d_gamma1 = block.rmsnorm1.d_gamma.astype(nx.float32) / self.gradient_scale / microbatch_size
                    d_gamma2 = block.rmsnorm2.d_gamma.astype(nx.float32) / self.gradient_scale / microbatch_size
                    Wqkv = nx.dequantize(block.attention.Wqkv, block.attention.scales[0], block.attention.biases[0], regular=self.symmetric_quant)
                    Wo = nx.dequantize(block.attention.Wo, block.attention.scales[1], block.attention.biases[1], regular=self.symmetric_quant)
                    Wcombined = nx.dequantize(block.ff.Wcombined, block.ff.scales[0], block.ff.biases[0], regular=self.symmetric_quant)
                    Wout = nx.dequantize(block.ff.Wout, block.ff.scales[1], block.ff.biases[1], regular=self.symmetric_quant)
                    all_network_params.extend(
                        [(f"Wqkv_{i}", Wqkv, dWqkv),
                        (f"Wo_{i}", Wo, dWo),
                        (f"ff_wcombined_{i}", Wcombined,dWcombined),
                        (f"ff_wout_{i}", Wout, dWout),
                        (f"ff_router_{i}", block.ff.router.astype(nx.float32), d_router),
                        (f"rmsnorm1_gamma_{i}", block.rmsnorm1.gamma.astype(nx.float32), d_gamma1),
                        (f"rmsnorm2_gamma_{i}", block.rmsnorm2.gamma.astype(nx.float32), d_gamma2)])
                    del Wqkv, Wo, Wcombined, Wout
                    del dWqkv, dWo, dWcombined, dWout, d_router, d_gamma1, d_gamma2
                    del block.attention.dWqkv, block.attention.dWo, block.ff.dWcombined, block.ff.dWout, block.ff.d_router, block.rmsnorm1.d_gamma, block.rmsnorm2.d_gamma


                lookup_table = nx.dequantize(self.embedding.lookup_table, self.embedding.table_scale, self.embedding.bias, regular=self.symmetric_quant)
                all_network_params.extend([("embedding",lookup_table, embed_acc / microbatch_size)])

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

                if self.quantized:
                    embedding = optimized[f"embedding"]
                    self.embedding.lookup_table, self.embedding.table_scale, self.embedding.bias = nx.quantize(embedding, regular=self.symmetric_quant)
                    del embedding
                else:
                    self.embedding.lookup_table = optimized["embedding"].astype(self.dtype)
                embed_acc = nx.zeros_like(embed_acc)

                for i in range(len(total_histograms)):
                    total_histograms[i] = total_histograms[i]

                step += 1

                #TODO: fix this hardcoding
                clean_step += 1
                if clean_step > 0 and clean_step % 1000 == 0:
                    self.gradient_scale = min(self.gradient_scale * 2, self.max_gradient_scale)

                self.eval_networks(include_gradients=False, optimizer=optimizer)

                yield total_loss.item(), count, total_histograms, step
                total_loss = nx.float_32(0)
                count = 0
                microstep = 0
                total_histograms = None
                nx.clear_cache()


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

        if self.quantized:
            #TODO this becomes nan, tho the lookup table and the scale themselves arent (fixed)
            scores = nx.quantized_matmul(output, self.embedding.lookup_table, self.embedding.table_scale, self.embedding.bias, transpose=True, regular=as_symmetric) #type:ignore
        else:
            scores = output @ self.embedding.lookup_table.T

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
        # configs += f"vocab_size: {str(self.vocab_size)}" + "\n"
        # configs += f"embed_dim: {str(self.embed_dim)}" + "\n"
        # configs += f"quantized: {str(self.quantized)}" + "\n"
        # configs += f"gradient_scale: {str(self.gradient_scale)}" + "\n"
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
                    for idx, block in enumerate(self.blocks):
                        for layer_i, layer in enumerate(layers):
                            layer_ = getattr(block, layer)
                            for weight in weights[layer_i]:
                                weight_ = getattr(layer_, weight)
                                all_weights[f"{idx}.{layer}.{weight}"] = weight_
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
        transformer_copy = Transformer(configs, block_copy, embedding=embedding_copy)

        return transformer_copy
