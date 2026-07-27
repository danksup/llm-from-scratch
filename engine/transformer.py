from engine.losses import cross_entropy_gradient, cross_entropy
from engine.embedding import Embedding
from engine.dataloader import DataLoader
from engine.optimizer.adamw import  AdamW
from engine.transformer_block import TransformerBlock
import engine.backend as nx
from typing import Any
import time
LAMBDA = 1e-2
GRADIENT_SCALE = 4096

default_block_configs = {
    "ff_hidden_width": 1024,
    "ff_n_experts":24,
    "ff_topk":2,
    "ff_cf":1.25,
    "attn_n_heads":16,
    "attn_n_kv_heads":4,
    "attn_windows":32
}

class Transformer:
    def __init__(self, configs: dict[str, Any] | None = None, blocks:list|None=None):
        #transformer_block:  
        # def __init__(self,embed_dim,ff_dim, n_heads, n_kv_heads, n_experts=1, cf=1.25, top_k =2, W=8, dtype=nx.float16) 
        self.blocks = []
        configs =  {} if configs is None else configs
        self.vocab_size = configs.get("vocab_size", 8192)
        self.embed_dim = configs.get("embed_dim", 128)
        self.dtype = configs.get("dtype", nx.float32)
        self.embedding = Embedding(self.vocab_size, self.embed_dim, self.dtype)
        block_configs =  default_block_configs | configs.get("block_configs", {})

        if blocks is None:
            n_blocks =  configs.get("n_blocks",4)
            block_overrides = configs.get("block_overrides", {})
            if block_overrides:
                for value in block_overrides.values():
                    if "dtype" in value:
                        raise ValueError('no dtype')

            for i in range(n_blocks):
                override = block_overrides.get(i, {})
                overrided = block_configs | override
                D = self.embed_dim
                H = overrided["ff_hidden_width"]
                n_heads = overrided["attn_n_heads"]
                n_kv_heads = overrided["attn_n_kv_heads"]
                E = overrided["ff_n_experts"]
                CF = overrided["ff_cf"]
                topk = overrided["ff_topk"]
                W = overrided["attn_windows"]

                transformer_block = TransformerBlock(D, H, n_heads, n_kv_heads, E, CF, topk,W, self.dtype)
                self.blocks.append(transformer_block) 
        else:
            self.blocks = blocks
            if not self.blocks:
                raise ValueError("lol")
            
            for i, block in enumerate(self.blocks):
                if block.embed_dim != self.embed_dim:
                    raise ValueError(f"block {i} embed dimension of {block.embed_dim} does not match the transformer's embed dimension of {self.embed_dim}")
            
    @classmethod
    def build(cls, input_size:int, output_size:int, hidden_layer_size:int=1, base_width:int=512) -> "Transformer":
        '''
        deprecated.
        Args:
            n
        build
        '''
        raise DeprecationWarning("no")
        
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
            output = output.astype(self.dtype)
            B,T,_ = output.shape
            Wqkv = block.attention.Wqkv
            Wo = block.attention.Wo
            Wcombined = block.ff.Wcombined
            Wout = block.ff.Wout
            router = block.ff.router
            epsilon = block.rmsnorm1.epsilon
            gamma1 = block.rmsnorm1.gamma
            gamma2 = block.rmsnorm2.gamma
            W = block.W
            W = min(W, T-1)
            P = nx.array(0.1, dtype=self.dtype)
            if block.causal_mask is None or block.causal_mask.shape != (T, W + 1):
                causal_mask = block.attention.compute_mask(W, T)
                block.causal_mask = causal_mask
            ff_out ,masks, caches, router_loss, normalized_histogram = block._forward(output, block.causal_mask, self.embed_dim, block.n_heads, block.n_kv_heads, block.n_rep, block.head_dim, W, block.n_experts, block.cf, block.ff.top_k,
                                                   block.freqs, Wqkv, Wo, Wcombined, router, block.hidden_width, Wout, epsilon, gamma1, gamma2, P, is_training)

            total_router_loss += router_loss
            output = ff_out
            all_masks.append(masks)
            all_caches.append(caches)
            histograms[idx] = nx.zeros_like(normalized_histogram) #type:ignore
            histograms[idx] += normalized_histogram

        last_output = output.astype(self.dtype)
        scores = last_output @ self.embedding.lookup_table.T
        # print("forward t scores", scores.dtype)

        if return_cache:
            return scores, last_output, all_masks, all_caches, total_router_loss, histograms
        return scores, total_router_loss
    
    def backward(self, err_signal:Any, lookup_table, last_output, all_masks,all_caches) -> Any:
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
            scaled_lambda = LAMBDA * GRADIENT_SCALE
            moe_configs = block.ff.cf, block.ff.n_experts, block.ff.hidden_width, block.ff.router, scaled_lambda
            P = nx.array(0.1, dtype=self.dtype)
            W = block.W
            W = min(W, T-1)
            attn_params = (block.n_heads, block.head_dim, block.embed_dim, block.n_kv_heads, block.n_rep, block.W, block.attention.Wo, block.freqs, block.attention.Wqkv)
            ff_params = (block.ff.Wout, block.ff.Wcombined)

            dx, dWout, dWcombined, d_router, dWqkv, dWo, d_gamma1, d_gamma2 = block._backward(current_grad, mask1=mask1, mask2=mask2, p=P,
                                                                caches_attn=caches_attn, caches_ff=caches_ff, caches_rmsnorm1=caches_rmsnorm1, caches_rmsnorm2=caches_rmsnorm2, 
                                                                attn_params=attn_params, gamma1=block.rmsnorm1.gamma, gamma2=block.rmsnorm2.gamma, ff_params=ff_params, moe_configs=moe_configs)
            
            block.ff.dWout = dWout 
            block.ff.dWcombined = dWcombined 
            block.ff.d_router = d_router 
            
            block.attention.dWqkv=dWqkv 
            block.attention.dWo = dWo 

            block.rmsnorm1.d_gamma = d_gamma1 
            block.rmsnorm2.d_gamma = d_gamma2 
            current_grad = dx
        
        return current_grad
    
    def train(self, dataloader:DataLoader, optimizer:AdamW, total_epoch:int, batch_size:int=32):
        '''
        Args:
            dataloader: Dataloader object
            embedding: Embedding object
            batch_size = number of batch
        '''
        total_loss = nx.float_32(0.0)
        total_histograms = None
        count = 0
        batch_counter = 0
        for contexts, next_tokens in dataloader.get_pairs(batch_size):              
            embedded = self.embedding.forward(contexts)  # shape (batch, context_size, embed_dim)
            batch_scores, last_output, all_masks, all_caches, total_router_loss, histograms = self.forward(embedded)
            if total_histograms is None:
                total_histograms = histograms
            else:
                for i in range(len(self.blocks)):
                    total_histograms[i] += histograms[i]
            loss = cross_entropy(batch_scores, next_tokens)
            loss = nx.mean(loss) + LAMBDA * total_router_loss

            if not nx.isfinite(loss):
                forward_nan = nx.isnan(loss)
                forward_inf = nx.isinf(loss)
                raise FloatingPointError(f"[FORWARD] backward inf of value {loss.item()} at {batch_counter}. isnan: {forward_nan} | isinf: {forward_inf}")

            batch_gradient = cross_entropy_gradient(batch_scores, next_tokens)
            batch_gradient /= (batch_gradient.shape[0] * batch_gradient.shape[1])
            scaled_batch_gradient = batch_gradient * GRADIENT_SCALE

            batch_gradient = scaled_batch_gradient.astype(self.dtype)
            block_gradient =  batch_gradient @ self.embedding.lookup_table #dtype
            
            d_table = batch_gradient.reshape(-1, self.vocab_size).T @ last_output.reshape(-1, self.embed_dim) 
            d_table = d_table.astype(nx.float32) / GRADIENT_SCALE
            
            current_grad = self.backward(block_gradient, self.embedding.lookup_table, last_output, all_masks, all_caches)
            current_grad = current_grad.astype(nx.float32) / GRADIENT_SCALE

            if not nx.isfinite(current_grad).all().item():
                backward_max = max(current_grad)
                backward_min = min(current_grad)
                backward_nan = nx.isnan(current_grad)
                backward_inf = nx.isinf(current_grad)
                raise FloatingPointError(f"[BACKWARD] backward inf at {batch_counter}. isnan: {backward_nan} | isinf: {backward_inf}.\nmin value: {backward_min}\nmax value: {backward_max}")

            embedding_gradient = nx.zeros_like(self.embedding.lookup_table, dtype=nx.float32)
            embedding_gradient = nx.add_at(embedding_gradient, contexts, current_grad)

            total_embedding_gradient = embedding_gradient + d_table

            all_network_params = []
            for i,block in enumerate(self.blocks):
                block.attention.dWqkv = block.attention.dWqkv.astype(nx.float32) / GRADIENT_SCALE
                block.attention.dWo = block.attention.dWo.astype(nx.float32) / GRADIENT_SCALE
                block.ff.dWcombined = block.ff.dWcombined.astype(nx.float32) / GRADIENT_SCALE
                block.ff.dWout = block.ff.dWout.astype(nx.float32) / GRADIENT_SCALE
                block.ff.d_router = block.ff.d_router.astype(nx.float32) / GRADIENT_SCALE
                block.rmsnorm1.d_gamma = block.rmsnorm1.d_gamma.astype(nx.float32) / GRADIENT_SCALE
                block.rmsnorm2.d_gamma = block.rmsnorm2.d_gamma.astype(nx.float32) / GRADIENT_SCALE
                all_network_params.extend(
                    [(f"Wqkv_{i}", block.attention.Wqkv.astype(nx.float32), block.attention.dWqkv),
                    (f"Wo_{i}", block.attention.Wo.astype(nx.float32), block.attention.dWo),
                    (f"ff_wcombined_{i}", block.ff.Wcombined.astype(nx.float32), block.ff.dWcombined),
                    (f"ff_wout_{i}", block.ff.Wout.astype(nx.float32), block.ff.dWout),
                    (f"ff_router_{i}", block.ff.router.astype(nx.float32), block.ff.d_router),
                    (f"rmsnorm1_gamma_{i}", block.rmsnorm1.gamma.astype(nx.float32), block.rmsnorm1.d_gamma),
                    (f"rmsnorm2_gamma_{i}", block.rmsnorm2.gamma.astype(nx.float32), block.rmsnorm2.d_gamma)])
            all_network_params.extend([("embedding",self.embedding.lookup_table.astype(nx.float32), total_embedding_gradient)])
            
            optimized = optimizer.step_many(all_network_params,dataloader.train_contexts, batch_size, total_epoch)
            for i,block in enumerate(self.blocks):
                block.attention.Wqkv = optimized[f"Wqkv_{i}"].astype(self.dtype)
                block.attention.Wo = optimized[f"Wo_{i}"].astype(self.dtype)
                block.ff.Wcombined = optimized[f"ff_wcombined_{i}"].astype(self.dtype)
                block.ff.Wout = optimized[f"ff_wout_{i}"].astype(self.dtype)
                block.ff.router = optimized[f"ff_router_{i}"]
                block.rmsnorm1.gamma = optimized[f"rmsnorm1_gamma_{i}"]
                block.rmsnorm2.gamma = optimized[f"rmsnorm2_gamma_{i}"]
            self.embedding.lookup_table = optimized["embedding"].astype(self.dtype)

            total_loss += loss.item() * next_tokens.size
            count += next_tokens.size
            batch_counter += 1

            del (embedded,batch_scores,last_output,all_masks,all_caches,total_router_loss,loss,batch_gradient,block_gradient,d_table,current_grad,
                    embedding_gradient,total_embedding_gradient,all_network_params,optimized,)
            # if batch_counter % 10 == 0:
            #     gc.collect()

        final_loss = total_loss / count
        if total_histograms != None:
            for i in range(len(total_histograms)):
                total_histograms[i] /= batch_counter
        return nx.float_32(final_loss), total_histograms, batch_counter
    
    def benchmark(self, dataloader:DataLoader, optimizer:AdamW, batch_size:int=32, pass_ =1):
        total_loss = nx.float_32(0.0)
        count = 0
        batch_idx = 0
        total_epoch = 1

        loss_times = []
        backward_times = []
        network_optimizer_times = []
        total_histograms = None
        weighted_router_loss = nx.float_32(0.0)
        for contexts, next_tokens in dataloader.get_pairs(batch_size):  
            if batch_idx == pass_:
                break            
            embedded = self.embedding.forward(contexts)  # shape (batch, context_size, embed_dim)
            start = time.perf_counter()
            batch_scores, last_output, all_masks, all_caches, total_router_loss, histograms = self.forward(embedded)
            loss = cross_entropy(batch_scores, next_tokens) 
            nx.eval(loss)
            end = time.perf_counter()
            loss_times.append(end-start)

            histogram_loss = LAMBDA * total_router_loss
            loss = nx.mean(loss) + histogram_loss
            weighted_router_loss += histogram_loss
            if not nx.isfinite(loss):
                raise FloatingPointError("nan/inf")

            if total_histograms is None:
                total_histograms = histograms
            else:
                for i in range(len(self.blocks)):
                    total_histograms[i] += histograms[i]
           
            batch_gradient = cross_entropy_gradient(batch_scores, next_tokens)
            batch_gradient /= (batch_gradient.shape[0] * batch_gradient.shape[1])
            scaled_batch_gradient = batch_gradient * GRADIENT_SCALE

            batch_gradient = scaled_batch_gradient.astype(self.dtype) #cast
            block_gradient =  batch_gradient @ self.embedding.lookup_table #dtype

            d_table = batch_gradient.reshape(-1, self.vocab_size).T @ last_output.reshape(-1, self.embed_dim) #dtype
            
            d_table = d_table.astype(nx.float32) / GRADIENT_SCALE

            start = time.perf_counter()
            current_grad = self.backward(block_gradient, self.embedding.lookup_table, last_output, all_masks, all_caches)
            current_grad = current_grad.astype(nx.float32) / GRADIENT_SCALE

            nx.eval(current_grad)
            end = time.perf_counter()
            backward_times.append(end-start)

            embedding_gradient = nx.zeros_like(self.embedding.lookup_table, dtype=nx.float32)
            embedding_gradient = nx.add_at(embedding_gradient, contexts, current_grad)

            total_embedding_gradient = embedding_gradient + d_table

            all_network_params = []
            for i,block in enumerate(self.blocks):
                block.attention.dWqkv = block.attention.dWqkv.astype(nx.float32) / GRADIENT_SCALE
                block.attention.dWo = block.attention.dWo.astype(nx.float32) / GRADIENT_SCALE
                block.ff.dWcombined = block.ff.dWcombined.astype(nx.float32) / GRADIENT_SCALE
                block.ff.dWout = block.ff.dWout.astype(nx.float32) / GRADIENT_SCALE
                block.ff.d_router = block.ff.d_router.astype(nx.float32) / GRADIENT_SCALE
                block.rmsnorm1.d_gamma = block.rmsnorm1.d_gamma.astype(nx.float32) / GRADIENT_SCALE
                block.rmsnorm2.d_gamma = block.rmsnorm2.d_gamma.astype(nx.float32) / GRADIENT_SCALE
                all_network_params.extend(
                    [(f"Wqkv_{i}", block.attention.Wqkv.astype(nx.float32), block.attention.dWqkv),
                    (f"Wo_{i}", block.attention.Wo.astype(nx.float32), block.attention.dWo),
                    (f"ff_wcombined_{i}", block.ff.Wcombined.astype(nx.float32), block.ff.dWcombined),
                    (f"ff_wout_{i}", block.ff.Wout.astype(nx.float32), block.ff.dWout),
                    (f"ff_router_{i}", block.ff.router.astype(nx.float32), block.ff.d_router),
                    (f"rmsnorm1_gamma_{i}", block.rmsnorm1.gamma.astype(nx.float32), block.rmsnorm1.d_gamma),
                    (f"rmsnorm2_gamma_{i}", block.rmsnorm2.gamma.astype(nx.float32), block.rmsnorm2.d_gamma)])
            all_network_params.extend([("embedding",self.embedding.lookup_table.astype(nx.float32), total_embedding_gradient)])
            
            start = time.perf_counter()
            optimized = optimizer.step_many(all_network_params,dataloader.train_contexts, batch_size, total_epoch)
            for i,block in enumerate(self.blocks):
                block.attention.Wqkv = optimized[f"Wqkv_{i}"].astype(self.dtype)
                block.attention.Wo = optimized[f"Wo_{i}"].astype(self.dtype)
                block.ff.Wcombined = optimized[f"ff_wcombined_{i}"].astype(self.dtype)
                block.ff.Wout = optimized[f"ff_wout_{i}"].astype(self.dtype)
                block.ff.router = optimized[f"ff_router_{i}"]
                block.rmsnorm1.gamma = optimized[f"rmsnorm1_gamma_{i}"]
                block.rmsnorm2.gamma = optimized[f"rmsnorm2_gamma_{i}"]
            self.embedding.lookup_table = optimized["embedding"].astype(self.dtype)

            to_eval = []
            for block in self.blocks:
                to_eval.append(block.attention.Wqkv)
                to_eval.append( block.attention.Wo)
                to_eval.append(block.ff.Wcombined)
                to_eval.append(block.ff.Wout)
                to_eval.append(block.ff.router)
                to_eval.append(block.rmsnorm1.gamma)
                to_eval.append(block.rmsnorm2.gamma)
            to_eval.append(self.embedding.lookup_table)

            nx.eval(*to_eval)
            end = time.perf_counter()
            network_optimizer_times.append(end-start)

            total_loss += loss.item() * next_tokens.size
            count += next_tokens.size
            batch_idx += 1

            del (embedded,batch_scores,last_output,all_masks,all_caches,total_router_loss,loss,batch_gradient,block_gradient,d_table,current_grad,
                    embedding_gradient,total_embedding_gradient,all_network_params,optimized,)
            
        final_loss = total_loss / count
        if total_histograms != None:
            for i in range(len(total_histograms)):
                total_histograms[i] /= batch_idx
        weighted_router_loss /= batch_idx
        return nx.float_32(final_loss), loss_times, backward_times, network_optimizer_times, total_histograms, weighted_router_loss
    
    def validate(self, dataloader:DataLoader, batch_size:int, train_split=.9):
        total_loss = nx.float_32(0.0)
        count = 0
        dataloader.train_split = train_split
    
        for contexts, next_tokens in dataloader.get_validation_pairs(batch_size):
            embedded = self.embedding.forward(contexts) 
            batch_validation_scores, total_router_loss = self.forward(embedded, False, False)
            
            val_loss = cross_entropy(batch_validation_scores, next_tokens)
            val_loss = nx.mean(val_loss) + LAMBDA * total_router_loss
            total_loss += val_loss.item() * next_tokens.size
            count += next_tokens.size
            
        final_loss = total_loss / count
        return nx.float_32(final_loss)

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
        configs += "precision: float32" if self.dtype == nx.float32 else f"precision: mixed precision ({self.dtype})" 
        configs += "\n"

        for i,block in enumerate(self.blocks):
            H = block.hidden_width
            n_heads = block.n_heads
            Ne = block.n_experts
            n_kv_heads = block.n_kv_heads
            topk = block.ff.top_k
            W = block.W
            configs += f"block {i}: n_heads: {n_heads} | n_kv_heads: {n_kv_heads} | windows: {W} | n_experts: {Ne} | hidden_width: {H} | topk: {topk}\n"

        return configs