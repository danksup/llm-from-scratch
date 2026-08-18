from engine.losses import cross_entropy_gradient, cross_entropy
from engine.embedding import Embedding
from engine.dataloader import DataLoader
import engine.optimizer as optim
from engine.transformer_block import TransformerBlock
import engine.attention as attn
import engine.initializers as init
import engine.backend as nx
from typing import Any, Literal, Union
import copy
from helper.singleton import sleep
from engine.quantization import quantize, dequantize

optimizers = Union[optim.Adam, optim.AdamW, optim.SGD]

default_block_configs = {
    "ff_hidden_width": 1024,
    "ff_n_experts":24,
    "ff_topk":2,
    "ff_cf":1.25,
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
    def __init__(self, configs: dict[str, Any] | None = None, blocks:list|None=None):
        #transformer_block:  
        # def __init__(self,embed_dim,ff_dim, n_heads, n_kv_heads, n_experts=1, cf=1.25, top_k =2, W=8, dtype=nx.float16) 
        self.blocks = []
        configs =  {} if configs is None else configs
        self.vocab_size = configs.get("vocab_size", None)
        assert self.vocab_size is not None, "vocab size can't be None"
        self.embed_dim = configs.get("embed_dim", 128)
        self.dtype = configs.get("dtype", nx.float32)
        self.quantized = configs.get("quantize", False)
        assert isinstance(self.quantized, bool), f"True= int8 weights, rest floats; False = floats; got {self.quantized} of type {type(self.quantized)} instead."
        self.embedding = Embedding(self.vocab_size, self.embed_dim, self.dtype, self.quantized)
        self.gradient_scale = configs.get("gradient_scale", 4096)
        self.moe_lambda = configs.get("moe_lambda", 0.01)
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
                assert attn_variant in ATTN_VARIANT, f"[block {i}] invalid input: \"{attn_variant}\" of type {type({attn_variant})} for attn_variant. valid attn_variant: {", ".join(ATTN_VARIANT.keys())}"

                attn_type_str = overrided["attn_type"]
                assert attn_type_str in ATTN_TYPE, f"[block {i}] invalid input: \"{attn_type_str}\" of type {type({attn_type_str})} for attn_type. valid attn_type: {", ".join(ATTN_TYPE.keys())}"

                overrided = overrided | ATTN_TYPE[overrided["attn_type"]] | ATTN_VARIANT[overrided["attn_variant"]] | this  | override 
                overrided.pop('attn')

                attn_type = ATTN_TYPE[attn_type_str]["attn"]

                assert overrided["attn_init"] in INITIALIZERS, f"your configs contain invalid input: {overrided["attn_init"]} of type {type(overrided["attn_init"])}. expected for this config: {", ".join(list(INITIALIZERS))}"
                assert overrided["ff_init"] in INITIALIZERS, f"your configs contain invalid input: {overrided["ff_init"]} of type {type(overrided["ff_init"])}. expected for this config: {", ".join(list(INITIALIZERS))}"
                check = default_block_configs | ATTN_TYPE[this["attn_type"]] | ATTN_VARIANT[this["attn_variant"]] | ATTN_TYPE[overrided["attn_type"]] | ATTN_VARIANT[overrided["attn_variant"]]
                for config in overrided:
                    if config not in check:
                        raise ValueError(f"[block {i}] {config} is invalid. valid override: {", ".join(check.keys())}")
                self.individual_block_configs.append(overrided)

                D = self.embed_dim
                H = overrided["ff_hidden_width"]
                attn_init = INITIALIZERS[overrided["attn_init"]]
                E = overrided["ff_n_experts"]
                CF = overrided["ff_cf"]
                topk = overrided["ff_topk"]
                ff_init = INITIALIZERS[overrided["ff_init"]]

                if "attn_windows" in override and override.get("attn_type", None) == "full":
                    raise ValueError(f"[block {i}] attention type of {attn_type_str} doesn't accept \"attn_windows\"")
                n_heads = overrided["attn_n_heads"]
                attn = None
                W = overrided.get("attn_windows", None)
                match (attn_type_str, attn_variant):
                    case ("swa", "gqa"):
                        n_kv_heads = overrided["attn_n_kv_heads"] 
                        #def __init__(self,embed_dim:int, n_heads:int, n_kv_heads:int=-1, W=8, dtype:Any=nx.float16, initializer:Callable=initializer.glorot_uniform)
                        attn = attn_type(embed_dim=D, n_heads=n_heads, n_kv_heads=n_kv_heads, W=W, dtype=self.dtype, initializer=attn_init, quantized=self.quantized)
                    case ("swa", "mha"):
                        attn = attn_type.multihead(D, n_heads, W, self.dtype, attn_init, quantized=self.quantized)
                    case ("swa", "mqa"):
                        attn = attn_type.multiquery(D, n_heads, W, self.dtype, attn_init,quantized=self.quantized)
                    case ("swa", invalid):
                        raise ValueError(f"[block {i}] invalid variant of \"{invalid}\". valid variants: {", ".join(ATTN_VARIANT)}")
                    case ("full", "gqa"):
                        n_kv_heads = overrided["attn_n_kv_heads"] 
                        attn = attn_type(embed_dim=D, n_heads=n_heads, n_kv_heads=n_kv_heads,  dtype=self.dtype, initializer=attn_init,quantized=self.quantized)
                    case ("full", "mha"):
                        attn = attn_type.multihead(embed_dim=D, n_heads=n_heads,  dtype=self.dtype, initializer=attn_init,quantized=self.quantized)
                    case ("full", "mqa"):
                        attn = attn_type.multiquery(embed_dim=D, n_heads=n_heads,  dtype=self.dtype, initializer=attn_init,quantized=self.quantized)
                    case ("full", invalid):
                        raise ValueError(f"[block {i}] invalid variant of \"{invalid}\". valid variants: {", ".join(ATTN_VARIANT)}")
                    case _:
                        raise ValueError(f"[block {i}] invalid variant of \"{attn_variant}\". valid variants: {", ".join(ATTN_VARIANT)}")

                transformer_block = TransformerBlock(D, attn, H, E, CF, topk, self.dtype, attn_init, ff_init, self.quantized)
                self.blocks.append(transformer_block) 
        else:
            self.blocks = blocks
            if not self.blocks:
                raise ValueError("lol")
            
            for i, block in enumerate(self.blocks):
                if block.embed_dim != self.embed_dim:
                    raise ValueError(f"block {i} embed dimension of {block.embed_dim} does not match the transformer's embed dimension of {self.embed_dim}")
    
    def __str__(self) -> str:
        return self.get_configs_str()

    def count_params(self) -> int:
        """
        whole architecture number of (trainable) params 
        """
        total = 0
        for i in self.blocks:
            total += i.count_param()
        total += self.embedding.lookup_table.size
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
                scales = (block.attention.scales, block.ff.scales)
                ff_out ,masks, caches, router_loss, normalized_histogram = block._forward(output, block.causal_mask, attn_str ,block.attention.configs, attn_params, block.ff.configs, ff_params, epsilon, gamma1, gamma2, P, is_training, scales)
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
            lookup_table = dequantize(lookup_table, self.embedding.table_scale, self.dtype)
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
            scaled_lambda = self.moe_lambda * self.gradient_scale
            moe_configs = block.ff.cf, block.ff.n_experts, block.ff.hidden_width, block.ff.router, scaled_lambda
            P = nx.array(0.1, dtype=self.dtype)

            ff_params = (block.ff.Wout, block.ff.Wcombined)
            attn_str = block.attention.self_type()
            attn_configs = block.attention.configs
            attn_params = block.attention.Wqkv, block.attention.Wo
            scales = (block.attention.scales, block.ff.scales)
            dx, dWout, dWcombined, d_router, dWqkv, dWo, d_gamma1, d_gamma2 = block._backward(current_grad, mask1=mask1, mask2=mask2, p=P, attention=attn_str,
                                                                caches_attn=caches_attn, caches_ff=caches_ff, caches_rmsnorm1=caches_rmsnorm1, caches_rmsnorm2=caches_rmsnorm2, 
                                                                attn_configs = attn_configs, attn_params=attn_params, gamma1=block.rmsnorm1.gamma, gamma2=block.rmsnorm2.gamma, ff_params=ff_params, moe_configs=moe_configs, quantization=scales)

            
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
        weights = [["router", "Wcombined", "Wout"],[ "Wo", "Wqkv"],[ "gamma"],[ "gamma"]]
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
                    layer_ = getattr(block, layer)
                    dweight_ = getattr(layer_, dweight)
                    if not nx.isfinite(dweight_).all():
                        text += f"{layer}_{dweight} "
            if ltext == len(text):
                continue
            texts += f"{text}\n"
        return texts
                    
    def train(self, dataloader:DataLoader, optimizer:optimizers, total_epoch:int, max_step:int=50000, eval_every:int=5, microbatch_size:int=16):
        total_loss = nx.float_32(0.0)
        count = 0
        microstep = 0 
        step = 0
        total_histograms = None
        embed_acc = nx.zeros((self.vocab_size, self.embed_dim), nx.float32)

        for contexts, next_tokens in dataloader.prefetch_batch(dataloader.train_files): 
            contexts = nx.array(contexts, nx.int32)
            next_tokens = nx.array(next_tokens, nx.int32)

            if step >= max_step:
                break

            embedded = self.embedding.forward(contexts)  # shape (batch, context_size, embed_dim)
            batch_scores, last_output, all_masks, all_caches, total_router_loss, histograms = self.forward(embedded)
           
            if total_histograms is None:
                total_histograms = histograms
            else:
                for i in range(len(self.blocks)):
                    total_histograms[i] += histograms[i]
            
            loss = cross_entropy(batch_scores, next_tokens)
            loss = nx.mean(loss) + self.moe_lambda * total_router_loss

            batch_gradient = cross_entropy_gradient(batch_scores, next_tokens)
            batch_gradient /= (batch_gradient.shape[0] * batch_gradient.shape[1])
            scaled_batch_gradient = batch_gradient * self.gradient_scale

            batch_gradient = scaled_batch_gradient.astype(self.dtype)

            lookup_table = self.embedding.lookup_table
            if self.quantized:
                lookup_table = dequantize(lookup_table, self.embedding.table_scale, self.dtype)

            block_gradient =  batch_gradient @ lookup_table #dtype
            
            d_table = batch_gradient.reshape(-1, self.vocab_size).T @ last_output.reshape(-1, self.embed_dim) 
            d_table = d_table.astype(nx.float32) / self.gradient_scale

            current_grad = self.backward(block_gradient, all_masks, all_caches)
            current_grad = current_grad.astype(nx.float32) / self.gradient_scale

            embedding_gradient = nx.zeros_like(self.embedding.lookup_table, dtype=nx.float32)
            embedding_gradient = nx.add_at(embedding_gradient, contexts, current_grad)

            total_embedding_gradient = embedding_gradient + d_table
            embed_acc += total_embedding_gradient

            total_loss += loss * next_tokens.size
            count += next_tokens.size
            microstep += 1
        
            if microstep % eval_every == 0:
                to_eval = [total_loss, self.embedding.lookup_table, embed_acc, total_histograms]
                self.eval_networks(to_eval)

                if not nx.isfinite(loss).item():
                    forward_nan = nx.isnan(loss)
                    forward_inf = nx.isinf(loss)
                    nan_weights = self.non_finite_check()

                    raise FloatingPointError(f"[FORWARD step: {step}] non finite loss at microstep {microstep}. isnan: {forward_nan} | isinf: {forward_inf} |\n non-finite weights:\n{nan_weights}")

                if not nx.isfinite(current_grad).all().item():
                    backward_max = nx.max(current_grad)
                    backward_min = nx.min(current_grad)
                    backward_nan = nx.isnan(current_grad).any()
                    backward_inf = nx.isinf(current_grad).any()
                    nan_weights = self.non_finite_check()
                    raise FloatingPointError(f"[BACKWARD step: {step}] non-finite gradient at microstep {microstep}. isnan: {backward_nan} | isinf: {backward_inf}.\nmin value: {backward_min}\nmax value: {backward_max}, | non-finite weights: {nan_weights}")
                
            if microstep > 0 and microstep % microbatch_size == 0:
                all_network_params = []
                for i,block in enumerate(self.blocks):
                    dWqkv = block.attention.dWqkv.astype(nx.float32) / self.gradient_scale / microbatch_size
                    dWo = block.attention.dWo.astype(nx.float32) / self.gradient_scale / microbatch_size
                    dWcombined = block.ff.dWcombined.astype(nx.float32) / self.gradient_scale / microbatch_size
                    dWout = block.ff.dWout.astype(nx.float32) / self.gradient_scale / microbatch_size
                    d_router = block.ff.d_router.astype(nx.float32) / self.gradient_scale / microbatch_size
                    d_gamma1 = block.rmsnorm1.d_gamma.astype(nx.float32) / self.gradient_scale / microbatch_size
                    d_gamma2 = block.rmsnorm2.d_gamma.astype(nx.float32) / self.gradient_scale / microbatch_size
                    Wqkv = dequantize(block.attention.Wqkv, block.attention.scales[0])
                    Wo = dequantize(block.attention.Wo, block.attention.scales[1])
                    Wcombined = dequantize(block.ff.Wcombined, block.ff.scales[0])
                    Wout = dequantize(block.ff.Wout, block.ff.scales[1])
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

                lookup_table = dequantize(self.embedding.lookup_table, self.embedding.table_scale)
                all_network_params.extend([("embedding",lookup_table, embed_acc / microbatch_size)])
                
                optimized = optimizer.step_many(all_network_params, max_step, total_epoch)
                
                for i,block in enumerate(self.blocks):
                    if self.quantized:
                        Wqkv = optimized[f"Wqkv_{i}"]
                        block.attention.Wqkv, wqkv_scale, _ = quantize(Wqkv, nx.int8)
                        Wo = optimized[f"Wo_{i}"]
                        block.attention.Wo, wo_scale, _ = quantize(Wo, nx.int8)
                        block.attention.scale = (wqkv_scale, wo_scale)

                        Wcombined = optimized[f"ff_wcombined_{i}"]
                        block.ff.Wcombined, wcombined_scale, _ = quantize(Wcombined, nx.int8)
                        Wout = optimized[f"ff_wout_{i}"]
                        block.ff.Wout, wout_scale, _ = quantize(Wout, nx.int8)
                        block.ff.scale = (wcombined_scale, wout_scale)

                        del Wqkv,Wo,Wcombined,Wout
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
                    self.embedding.lookup_table, self.embedding.table_scale, _ = quantize(embedding, nx.int8)
                    del embedding
                else:
                    self.embedding.lookup_table = optimized["embedding"].astype(self.dtype)
                embed_acc = nx.zeros_like(embed_acc)

                for i in range(len(total_histograms)):
                    total_histograms[i] = total_histograms[i] / microbatch_size

                step += 1

                self.eval_networks(include_gradients=False, optimizer=optimizer)
                
                yield total_loss.item(), count, total_histograms, step
                nx.clear_cache()
        
    def validate(self, dataloader:DataLoader, val_step:int|Literal["all"]="all"):
        total_loss = nx.float_32(0.0)
        count = 0
        step_counter = 0

        if not dataloader.validation_files:
            return None

        for contexts, next_tokens in dataloader.prefetch_batch(dataloader.validation_files):
            if isinstance(val_step, int) and step_counter >= val_step:
                break
            contexts = nx.array(contexts, nx.int32)
            next_tokens = nx.array(next_tokens, nx.int32)
            
            embedded = self.embedding.forward(contexts) 
            batch_validation_scores, total_router_loss = self.forward(embedded, False, False)
            
            val_loss = cross_entropy(batch_validation_scores, next_tokens)
            val_loss = nx.mean(val_loss) + self.moe_lambda * total_router_loss
            total_loss += val_loss * next_tokens.size
            count += next_tokens.size
            step_counter += 1

            nx.eval(*[val_loss, total_loss])

        if count == 0:
            return None
        
        final_loss = total_loss / count
        return final_loss.item()

    def to_dict(self) -> dict[str, Any]:
        """
        get dictionary
        """
        a:dict[str,Any] = {"transformer_configs":{}}
        transformer_configs = a["transformer_configs"]
        transformer_configs["vocab_size"] = self.vocab_size
        transformer_configs["embed_dim"] = self.embed_dim
        transformer_configs["dtype"] = nx.dtype_to_srt[self.dtype]
        a["embedding"] = self.embedding.to_dict()
        blocks = []
        for block in self.blocks:
            blocks.append(block.to_dict())
        a["blocks"] = blocks
        return a
    
    @classmethod
    def from_dict(cls,thing:dict[str, Any]) -> "Transformer":
        configs = thing["transformer_configs"]
        configs["dtype"] = nx.str_to_dtype[configs["dtype"]]
        raw_blocks = thing["blocks"]
        blocks = []
        for block in raw_blocks:
            a = TransformerBlock.from_dict(block)
            blocks.append(a)
    
        transformer = cls(configs, blocks=blocks)
        transformer.embedding = Embedding.from_dict(thing["embedding"])
        
        return transformer
       
    def inference(self, context:Any, max_cache_len, all_caches = None,  position = 0) -> Any:
        if all_caches is None:
            all_caches = [(None, None) for _ in range(len(self.blocks))]
        output = self.embedding.forward(context)
        for idx, block in enumerate(self.blocks):
            cached_k, cached_v = all_caches[idx]
            ff_out, cache_k, cache_v = block.inference_forward(output,max_cache_len, cached_k, cached_v, position)
            all_caches[idx] = (cache_k, cache_v)
            output = ff_out

        scores = output @ self.embedding.lookup_table.T
        return scores, all_caches
    
    def get_configs_str(self):
        configs = ""
        configs += f"vocab_size: {str(self.vocab_size)}" + "\n"
        configs += f"embed_dim: {str(self.embed_dim)}" + "\n"
        configs += f"gradient_scale: {str(self.gradient_scale)}" + "\n"
        configs += "precision: full (float32)\n" if self.dtype == nx.float32 else f"precision: mixed precision ({self.dtype})\n"
        configs += f"quantized: {str(self.quantized)}" + "\n"
        configs += f"block configs: {self.block_configs}\n"
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

    @staticmethod
    def create_checkpoint(to_checkpoint:"Transformer") -> "Transformer":
        return to_checkpoint.from_dict(to_checkpoint.to_dict())