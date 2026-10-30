import math
from typing import Any, Callable

import engine.backend as nx
import engine.initializers as initializer
from engine.activations import softmax_derivative, swish, swish_derivative
from helper.singleton import sleep

C = nx.array(1e-4)

class FFN:
    def __init__(self, capacity_factor, top_k, n_experts, embed_dim, hidden_width, dense_hidden_width, n_dense=2,jitter_eps=0.02,LAMBDA =1e-2, dtype:Any=nx.float16, initializer:Callable=initializer.glorot_uniform, quantized:bool|str=False, *, init=True) -> None:
        self.hidden_width = hidden_width
        self.embed_dim = embed_dim #D
        self.n_experts = n_experts #E
        self.cf = capacity_factor
        self.top_k = top_k
        self.dtype = dtype
        self.n_dense = n_dense
        self.dense_hidden_width = dense_hidden_width
        self.jitter_eps = jitter_eps
        self.LAMBDA = LAMBDA

        self.configs = (hidden_width, embed_dim, n_experts, capacity_factor, top_k,dense_hidden_width, n_dense,jitter_eps, LAMBDA)
        self.quantized = quantized

        self.scales = (None, None, None, None)
        self.biases = (None, None, None, None)

        wcombined_shape =  (n_experts, embed_dim, hidden_width * 2)
        wout_shape =  (n_experts, hidden_width, embed_dim)
        router_shape = embed_dim, n_experts
        dense_combined_shape = embed_dim, n_dense * dense_hidden_width * 2
        dense_out_shape = (n_dense * dense_hidden_width, embed_dim)
        
        if init:
            if self.n_experts > 0:
                self.router = initializer(router_shape, dtype= nx.float32)
                self.Wcombined = initializer(wcombined_shape, dtype=dtype)
                self.Wout = initializer(wout_shape, dtype = dtype)
            else:
                self.router = None
                self.Wcombined = None
                self.Wout = None

            if n_dense > 0:
                self.W_dense_combined = initializer(dense_combined_shape, dtype=dtype)
                self.W_dense_out = initializer(dense_out_shape, dtype=dtype)
            else:
                self.W_dense_combined = None
                self.W_dense_out = None

            if quantized:
                if n_experts > 0:
                    self.Wcombined, wcombined_scale, wcombined_biases = nx.quantize(self.Wcombined)
                    self.Wout, wout_scale, wout_biases = nx.quantize(self.Wout)
                    self.scales = (wcombined_scale, wout_scale, self.scales[2], self.scales[3])
                    self.biases = (wcombined_biases, wout_biases, self.biases[2], self.biases[3])

                if n_dense > 0:
                    self.W_dense_combined, dense_combined_scale, dense_combined_bias = nx.quantize(self.W_dense_combined)
                    self.W_dense_out, dense_out_scale, dense_out_bias = nx.quantize(self.W_dense_out)
                    self.scales = (self.scales[0], self.scales[1], dense_combined_scale, dense_out_scale)
                    self.biases = (self.biases[0], self.biases[1], dense_combined_bias, dense_out_bias)

        if n_experts > 0:
            self.dWcombined = nx.zeros(wcombined_shape)
            self.dWout = nx.zeros(wout_shape)
            self.d_router = nx.zeros(router_shape, dtype=nx.float32)
        else:
            self.dWcombined = None
            self.dWout = None
            self.d_router = None

        if n_dense > 0:
            self.dW_dense_combined = nx.zeros(dense_combined_shape)
            self.dW_dense_out = nx.zeros(dense_out_shape)
        else:
            self.dW_dense_combined = None
            self.dW_dense_out = None

    def zeroes_gradient(self):
        if self.n_experts > 0:
            self.dWcombined = nx.zeros_like(self.dWcombined)
            self.dWout = nx.zeros_like(self.dWout)
            self.d_router = nx.zeros_like(self.d_router, dtype=nx.float32)

        if self.n_dense > 0:
            self.dW_dense_combined = nx.zeros_like(self.dW_dense_combined)
            self.dW_dense_out = nx.zeros_like(self.dW_dense_out)

    def count_param(self):
        total = 0
        total_dense = 0
        total_param = 0

        if self.n_experts > 0:
            total += 0 if not hasattr(self.router, "size") else getattr(self.router, "size")

            wcombined_size = 0 if not hasattr(self.Wcombined, "size") else getattr(self.Wcombined, "size")
            total  += wcombined_size
            total_dense += wcombined_size // self.n_experts

            wout_size = 0 if not hasattr(self.Wout, "size") else getattr(self.Wout, "size") 
            total  += wout_size
            total_dense += wout_size // self.n_experts

        if self.n_dense > 0:
            total_param += 0 if not hasattr(self.W_dense_combined, "size") else getattr(self.W_dense_combined, "size") 
            total_param += 0 if not hasattr(self.W_dense_out, "size") else getattr(self.W_dense_out, "size")

        if self.quantized:
            total_param *= 4

        total += total_param
        total_dense += total_param

        return total, total_dense

    @staticmethod 
    def compute_expert_input(n_experts,top_k, N, flatten_x, valid, safe_slot, capacity, D, flatten_top_expert_indices,dtype):
        assignement_tokens = nx.repeat( nx.arange(N, dtype=nx.int32), top_k) #(N*K,)
        masked_tokens = flatten_x[assignement_tokens] * valid[:, None] 
        expert_input = nx.zeros((n_experts, capacity, D), dtype=dtype)
        expert_input = nx.add_at(expert_input, (flatten_top_expert_indices, safe_slot), masked_tokens)
        return expert_input

    @staticmethod
    def compute_projected(expert_input, Wcombined, wcombined_quants):
        wcombined_scale, wcombined_bias = wcombined_quants
        if wcombined_scale is not None:

            projected = nx.quantized_matmul(expert_input, Wcombined, scales=wcombined_scale, biases=wcombined_bias) #(E, capacity, 2H)
        else:
            projected = expert_input @ Wcombined

        return projected

    @staticmethod 
    def compute_hidden(projected, dtype):
        H = projected.shape[-1] // 2
        gate_half = projected[..., :H]
        value_half = projected[..., H:]
        s = swish(gate_half, dtype)
        return s * value_half #(E, capacity, H) 

    @staticmethod
    def forward(x:nx.ArrayLike, ff_configs, ff_params, quantization:tuple[Any,...]|None=None, *, recompute_activation:bool=True, is_training=True):
        #routing
        # print("moe forward x", x.dtype)
        B, T, D = x.shape
        N = B * T
        _, _, n_experts, capacity_factor, top_k, _, n_dense,jitter_eps, LAMBDA = ff_configs
        Wcombined, Wout, router, W_dense_combined, W_dense_out  = ff_params

        wcombined_scale, wout_scale, dense_combined_scale, dense_out_scale, wcombined_bias, wout_bias, dense_combined_bias,  dense_out_bias = quantization  #type:ignore
        flatten_x = x.reshape(-1, D)

        final_output = nx.zeros((B,T,D), dtype=x.dtype)

        if n_experts > 0:
            top_k = min(top_k, n_experts)

            capacity = math.ceil(capacity_factor * N * top_k / n_experts)
                
            scores =  flatten_x.astype(nx.float32) @ router #(N, E)
            if is_training and jitter_eps > 0:
                scores += nx.uniform(-jitter_eps, jitter_eps,size=scores.shape, dtype=nx.float32)
            z_loss = C * nx.mean(nx.square(nx.logsumexp(scores, -1)))
            router_prob = nx.softmax(scores, -1) #(N, E)

            #top-k
            top_expert_indices = nx.topk(router_prob, top_k) #(N, K)
            row_idx = nx.arange(N, dtype=nx.int32)[:, None]  #(N,1)
            top_gates = router_prob[row_idx, top_expert_indices] # (N, K)
            top_gates32 = top_gates / nx.sum(top_gates, axis=-1, keepdims=True, dtype=nx.float32)#, dtype=DTYPE)
            top_gates = top_gates32.astype(x.dtype)

            flatten_top_expert_indices = top_expert_indices.reshape(-1) #(N*K,)
            flatten_top_gates = top_gates.reshape(-1) #(N*K,)
            assignement_tokens = nx.repeat( nx.arange(N, dtype=nx.int32), top_k) #(N*K,)

            histogram = nx.zeros(n_experts, dtype=nx.int32)
            histogram = nx.add_at(histogram, flatten_top_expert_indices, 1)
            avg_prob = nx.mean(router_prob, axis=0) #P
            normalized_histogram = histogram / ( N * top_k) #f
            router_loss = n_experts * nx.sum(normalized_histogram * avg_prob) * LAMBDA #L, fp32

            #dispatch
            M = N * top_k
            assignment_rows = nx.arange(M, dtype=nx.int32)
            routing_mask = nx.zeros((M,n_experts), nx.int32)
            routing_mask = nx.add_at(routing_mask, (assignment_rows, flatten_top_expert_indices), 1)

            cum_assignment = nx.cumsum(routing_mask, axis=0, dtype=nx.int32)

            slot_idx = cum_assignment[assignment_rows, flatten_top_expert_indices] - 1 #(N,)
            safe_slot = nx.clip(slot_idx, 0, capacity - 1, dtype=nx.int32)
            valid = slot_idx < capacity

            expert_input = FFN.compute_expert_input(n_experts, top_k, N, flatten_x, valid, safe_slot, capacity, D, flatten_top_expert_indices, x.dtype)

            expert_gate = nx.zeros((n_experts, capacity), dtype=top_gates.dtype)
            safe_gates = nx.where(valid, flatten_top_gates, nx.zeros_like(flatten_top_gates))
            expert_gate = nx.add_at(expert_gate, (flatten_top_expert_indices, safe_slot), safe_gates)

            projected = FFN.compute_projected(expert_input, Wcombined, (wcombined_scale, wcombined_bias)) #(E, capacity, 2H)

            hidden = FFN.compute_hidden(projected, x.dtype) #(E, capacity, H)

            if wout_scale is not None:
                raw_output = nx.quantized_matmul(hidden, Wout, scales=wout_scale, biases=wout_bias) #(E, capacity, D)
            else:
                raw_output = hidden @ Wout #(E, capacity, D)

            gated_output = raw_output * expert_gate[..., None]
            moe_output = gated_output[flatten_top_expert_indices, safe_slot]
            moe_output = moe_output * valid[..., None]
            moe_output = moe_output.reshape(N,top_k,D)
            moe_output = nx.sum(moe_output, axis=1, dtype=nx.float32).reshape(B,T,D) #check

            final_output += moe_output

            total_aux_loss = router_loss + z_loss

            if not recompute_activation:
                moe_cache = (router_prob, top_expert_indices, top_gates32, flatten_top_expert_indices, assignement_tokens, valid, safe_slot, expert_input, expert_gate, projected, hidden, raw_output, normalized_histogram, scores)
            else:
                moe_cache = (router_prob, top_expert_indices, top_gates32, flatten_top_expert_indices, assignement_tokens, valid, safe_slot, None,         expert_gate, None,      None,   raw_output, normalized_histogram, scores)

        else:
            moe_output = nx.zeros((B,T,D), dtype=x.dtype)
            total_aux_loss = nx.zeros((), dtype=nx.float32)
            normalized_histogram = None
            moe_cache = (None, None, None, None, None, None, None, None, None, None, None, None, None, None)

        if n_dense > 0:
            dense_projected = FFN.compute_projected(flatten_x, W_dense_combined, (dense_combined_scale, dense_combined_bias)) #(N, ns * 2Hs)

            dense_hidden = FFN.compute_hidden(dense_projected, x.dtype) #N, ns * H

            if dense_out_scale is not None:
                raw_dense_output = nx.quantized_matmul(dense_hidden,W_dense_out, scales=dense_out_scale, biases=dense_out_bias)
            else:
                raw_dense_output = dense_hidden @ W_dense_out #N, D

            dense_output = raw_dense_output.reshape(B,T,D)

            final_output += dense_output

            if recompute_activation:
                dense_cache = (dense_projected, dense_hidden)
            else:
                dense_cache = (None, None)
        else:
            dense_cache = (None, None)

        cache = (flatten_x,) + moe_cache + dense_cache
        return final_output, cache, total_aux_loss, normalized_histogram

    @staticmethod
    def backward(gradient , caches, moe_configs, ff_params, gradient_scale, quantization:tuple[Any,...]|None=None, *, recompute_activation:bool=True):
        B,T,D = gradient.shape
        N = B*T

        flatten_gradient = gradient.reshape(-1, D)
        
        wcombined_scale, wout_scale,shared_expert_scale,shared_expert_out_scale, wcombined_bias, wout_bias, shared_expert_bias,  shared_expert_out_bias = quantization  #type:ignore

        router, Wout, Wcombined, W_shared_combined, W_shared_out = ff_params

        hidden_width, D, n_experts, capacity_factor, top_k,shared_hidden_width, n_shared_experts, _, LAMBDA = moe_configs

        flatten_x, router_prob, top_expert_indices, top_gates32, flatten_top_expert_indices, assignement_tokens, valid, safe_slot, expert_input, expert_gate, projected, hidden, raw_output, normalized_histogram, scores, shared_projected, shared_hidden = caches

        dx_flat = nx.zeros((N,D), dtype=gradient.dtype)

        if n_shared_experts > 0:
            if recompute_activation:
                shared_projected = FFN.compute_projected(flatten_x, W_shared_combined, (shared_expert_scale,shared_expert_bias)) #(N,2H)
                shared_hidden = FFN.compute_hidden(shared_projected, gradient.dtype) #N, H

            dW_shared_out = shared_hidden.T @ flatten_gradient #H, D
            if shared_expert_out_scale is not None:
                    d_shared_hidden = nx.quantized_matmul(flatten_gradient, W_shared_out, shared_expert_out_scale, shared_expert_out_bias, True)
            else:
                d_shared_hidden = flatten_gradient @ W_shared_out.T #N,H
            
            shared_gate_half = shared_projected[..., :shared_hidden_width * n_shared_experts] #N,H
            shared_value_half = shared_projected[..., shared_hidden_width* n_shared_experts:]

            d_shared_gate_half = d_shared_hidden * shared_value_half * swish_derivative(shared_gate_half, dtype=shared_gate_half.dtype)
            d_shared_value_half = d_shared_hidden * swish(shared_gate_half, dtype=shared_value_half.dtype)
            d_shared_projected = nx.concatenate([d_shared_gate_half, d_shared_value_half], axis=-1) #N,2H

            dW_shared_combined = flatten_x.T @ d_shared_projected #D, 2H
            if shared_expert_scale is not None:
                d_x_shared = nx.quantized_matmul(d_shared_projected, W_shared_combined, shared_expert_scale, shared_expert_bias, transpose=True)
            else:
                d_x_shared = d_shared_projected @ W_shared_combined.T #N,D

            dx_flat += d_x_shared
            ret_dense = (dW_shared_combined, dW_shared_out)
        else:
            ret_dense = (None, None)

        if n_experts > 0:
            capacity = math.ceil(capacity_factor * N * top_k / n_experts)

            if recompute_activation:
                expert_input = FFN.compute_expert_input(n_experts, top_k, N, flatten_x, valid, safe_slot, capacity, D, flatten_top_expert_indices, gradient.dtype)
                projected = FFN.compute_projected(expert_input, Wcombined, (wcombined_scale,wcombined_bias))
                hidden = FFN.compute_hidden(projected, gradient.dtype)

            M = N *top_k
        
            assignment_gradient = flatten_gradient[assignement_tokens] #(M,D)
            d_masked_output = assignment_gradient * valid[...,None]

            d_gated_output = nx.zeros((n_experts, capacity, D), dtype=gradient.dtype)

            d_gated_output = nx.add_at(d_gated_output, (flatten_top_expert_indices, safe_slot), d_masked_output)

            d_raw_output = d_gated_output * expert_gate[..., None] #(E, capacity, D) #fp16
            d_expert_gate = nx.sum(d_gated_output * raw_output, axis=-1, dtype=nx.float32,) #(E, capacity)

            dWout = hidden.transpose(0, 2, 1) @ d_raw_output
            
            if wout_scale is not None:
                d_hidden = nx.quantized_matmul(d_raw_output, Wout, wout_scale,biases=wout_bias, transpose=(0,2,1))  #(E,C,H) fp16
            else:
                d_hidden = d_raw_output @ Wout.transpose(0, 2, 1)

            del hidden, d_gated_output, Wout

            gate_half = projected[..., :hidden_width]
            value_half = projected[..., hidden_width:]

            d_gate_half = d_hidden * value_half * swish_derivative(gate_half, dtype=gate_half.dtype)
            d_value_half = d_hidden * swish(gate_half, dtype=value_half.dtype)
            d_projected = nx.concatenate([d_gate_half, d_value_half], axis=-1) #(E, C, 2H)  fp16

            del projected, d_hidden, gate_half, value_half, d_gate_half, d_value_half

            dWcombined = expert_input.transpose(0, 2, 1) @ d_projected #(E,D,2H) fp16


            if wcombined_scale is not None:
                d_expert_input = nx.quantized_matmul(d_projected, Wcombined, wcombined_scale,biases=wcombined_bias, transpose=(0,2,1))
            else:
                d_expert_input = d_projected @ Wcombined.transpose(0,2,1) #(E, C, D) fp16

            d_x_expert = d_expert_input[flatten_top_expert_indices, safe_slot]
            d_x_expert *= valid[...,None]

            d_chosen_gate = d_expert_gate[flatten_top_expert_indices, safe_slot]
            d_chosen_gate *= valid

            del d_expert_input, d_projected, expert_input, d_raw_output, raw_output, expert_gate, Wcombined

            d_chosen_gate = d_chosen_gate.reshape(N,top_k)
            token_rows = nx.arange(N, dtype=nx.int32)[:,None]
            selected_prob = router_prob[token_rows, top_expert_indices] #N,K
            gate_sum = nx.sum(selected_prob, -1, keepdims=True, dtype=nx.float32) #(N,1)
            coupling = nx.sum(d_chosen_gate * top_gates32, -1, keepdims=True, dtype=nx.float32) #(N,1)
            d_selected_prob = (d_chosen_gate - coupling)/gate_sum #(N,K)
            d_selected_prob = d_selected_prob.reshape(-1,)

            d_router_prob = nx.zeros((N,n_experts), dtype=d_selected_prob.dtype) #fp32
            d_router_prob[assignement_tokens, flatten_top_expert_indices] = d_selected_prob

            del coupling, gate_sum, token_rows, selected_prob, top_gates32, d_chosen_gate,d_selected_prob, assignement_tokens, flatten_top_expert_indices

            d_avg_prob = n_experts * normalized_histogram
            d_router_prob += LAMBDA * (d_avg_prob / N)

            d_scores = softmax_derivative(router_prob, d_router_prob) #(N,E)
            d_z_loss = ((2 * C) / N) * nx.logsumexp(scores, -1, keepdims=True) * router_prob
            d_scores += d_z_loss * gradient_scale


            d_router = flatten_x.astype(nx.float32).T @ d_scores #(D,E) #fp32
            d_x_router = d_scores @ router.T #(N, D)

            del normalized_histogram, d_avg_prob,flatten_x, d_scores, router

            d_x_expert = d_x_expert.reshape(N, top_k, D)
            d_x_expert = nx.sum(d_x_expert, axis=1, dtype=nx.float32) #(N,D)
            dx_flat += d_x_expert + d_x_router #(N,D)

            ret_moe = (dWcombined, dWout, d_router,)
        else:
            ret_moe = (None, None, None)

        dx = dx_flat.reshape(B,T,D)

        del dx_flat, top_expert_indices, valid, safe_slot

        return (dx,) + ret_moe + ret_dense

    @staticmethod
    def inference_forward(x:nx.ArrayLike, ff_configs, ff_params, quantization:tuple[Any,...]|None=None):
        #routing
        B, T, D = x.shape
        N = B * T

        _, _, n_experts, capacity_factor, top_k, _, n_dense, _, _ = ff_configs

        Wcombined, Wout, router, W_shared_combined, W_shared_out  = ff_params

        wcombined_scale, wout_scale,shared_expert_scale,shared_expert_out_scale, wcombined_bias, wout_bias, shared_expert_bias,  shared_expert_out_bias = quantization  #type:ignore

        flatten_x = x.reshape(-1, D)
        final_output = nx.zeros((B,T,D), dtype=x.dtype )

        if n_experts > 0:

            top_k = min(top_k, n_experts)

            scores =  flatten_x.astype(nx.float32) @ router #(N, E)
            router_prob = nx.softmax(scores, -1) #(N, E)

            #top-k
            top_expert_indices = nx.topk(router_prob, top_k) #(N, K)
            row_idx = nx.arange(N, dtype=nx.int32)[:, None]  #(N,1)
            top_gates = router_prob[row_idx, top_expert_indices] # (N, K)
            top_gates32 = top_gates / nx.sum(top_gates, axis=-1, keepdims=True, dtype=nx.float32)#, dtype=DTYPE)
            top_gates = top_gates32.astype(x.dtype)

            flatten_top_expert_indices = top_expert_indices.reshape(-1) #(N*K,)
            flatten_top_gates = top_gates.reshape(-1) #(N*K,)

            if N == 1:
                Wcombined = Wcombined[flatten_top_expert_indices] #K,D,2H
                Wout = Wout[flatten_top_expert_indices] #K H D

                if wcombined_scale is not None:
                    wcombined_scale = wcombined_scale[flatten_top_expert_indices]
                    wcombined_bias = wcombined_bias[flatten_top_expert_indices]
                    wout_scale = wout_scale[flatten_top_expert_indices]
                    wout_bias = wout_bias[flatten_top_expert_indices]
                    
                projected = FFN.compute_projected(flatten_x[None,:,:], Wcombined, (wcombined_scale, wcombined_bias))

                hidden = FFN.compute_hidden(projected, x.dtype) #(E, capacity, H)

                if wout_scale is not None:
                    raw_output = nx.quantized_matmul(hidden, Wout, scales=wout_scale, biases=wout_bias) #(E, capacity, D)
                else:
                    raw_output = hidden @ Wout #(E, capacity, D)

                moe_output = raw_output * flatten_top_gates[:, None, None]
            else:
                capacity = math.ceil(capacity_factor * N * top_k / n_experts)
                #dispatch
                M = N * top_k
                assignment_rows = nx.arange(M, dtype=nx.int32)
                routing_mask = nx.zeros((M,n_experts), nx.int32)
                routing_mask = nx.add_at(routing_mask, (assignment_rows, flatten_top_expert_indices), 1)
        
                cum_assignment = nx.cumsum(routing_mask, axis=0, dtype=nx.int32)
        
                slot_idx = cum_assignment[assignment_rows, flatten_top_expert_indices] - 1 #(N,)
                safe_slot = nx.clip(slot_idx, 0, capacity - 1, dtype=nx.int32)
                valid = slot_idx < capacity
        
                expert_input = FFN.compute_expert_input(n_experts, top_k, N, flatten_x, valid, safe_slot, capacity, D, flatten_top_expert_indices, x.dtype) #(n_experts, capacity, D)
        
                expert_gate = nx.zeros((n_experts, capacity), dtype=top_gates.dtype)
                safe_gates = nx.where(valid, flatten_top_gates, nx.zeros_like(flatten_top_gates))
                expert_gate = nx.add_at(expert_gate, (flatten_top_expert_indices, safe_slot), safe_gates)
                
                projected = FFN.compute_projected(expert_input, Wcombined, (wcombined_scale, wcombined_bias)) #(E, capacity, 2H)

                hidden = FFN.compute_hidden(projected, x.dtype) #(E, capacity, H)

                if wout_scale is not None:
                    raw_output = nx.quantized_matmul(hidden, Wout, scales=wout_scale, biases=wout_bias) #(E, capacity, D)
                else:
                    raw_output = hidden @ Wout #(E, capacity, D)

                gated_output = raw_output * expert_gate[..., None]
                moe_output = gated_output[flatten_top_expert_indices, safe_slot]
                moe_output = moe_output * valid[..., None]
            
            moe_output = moe_output.reshape(N,top_k,D)
            moe_output = nx.sum(moe_output, axis=1, dtype=nx.float32).reshape(B,T,D) #check

            final_output += moe_output
        if n_dense > 0:
            shared_projected = FFN.compute_projected(flatten_x, W_shared_combined, (shared_expert_scale,shared_expert_bias)) #(N, ns * 2Hs)

            shared_hidden = FFN.compute_hidden(shared_projected, x.dtype) #N, ns * H

            if shared_expert_out_scale is not None:
                raw_shared_output = nx.quantized_matmul(shared_hidden,W_shared_out, scales=shared_expert_out_scale, biases=shared_expert_out_bias)
            else:
                raw_shared_output = shared_hidden @ W_shared_out #N, D

            shared_output = raw_shared_output.reshape(B,T,D)

            final_output += shared_output.astype(x.dtype)

        return final_output

    @classmethod
    def from_weight(cls, configs, weights, quants, dtype) -> "FFN":
        hidden_width, embed_dim, n_experts, capacity_factor, top_k, shared_hidden_width, n_shared_experts, jitter_eps, LAMBDA = configs
        router, wcombined, wout,W_dense_combined,W_dense_out = weights

        moe = cls(capacity_factor, top_k, n_experts, embed_dim, hidden_width, shared_hidden_width, n_shared_experts,jitter_eps,LAMBDA, dtype, init=False)
        moe.router = router
        moe.Wcombined = wcombined
        moe.Wout = wout
        moe.W_dense_combined = W_dense_combined
        moe.W_dense_out = W_dense_out

        if quants is not None:
            scales, biases = quants
            moe.scales = scales
            moe.biases = biases

        return moe

    def copy(self):
        moe_copy = FFN(self.cf, self.top_k, self.n_experts, self.embed_dim, self.hidden_width, self.dense_hidden_width, self.n_dense, self.jitter_eps,self.LAMBDA, dtype=self.dtype, quantized=self.quantized, init=False)

        if self.n_experts > 0:
            moe_copy.router = nx.copy(self.router)
            moe_copy.Wcombined = nx.copy(self.Wcombined)
            moe_copy.Wout = nx.copy(self.Wout)
            if self.quantized:
                moe_copy.scales = (nx.copy(self.scales[0]), nx.copy(self.scales[1]), self.scales[2], self.scales[3])
                moe_copy.biases = (nx.copy(self.biases[0]), nx.copy(self.biases[1]), self.biases[2], self.biases[3])


        if self.n_dense > 0:
            moe_copy.W_dense_combined = nx.copy(self.W_dense_combined)
            moe_copy.W_dense_out = nx.copy(self.W_dense_out)
            if self.quantized:
                moe_copy.scales = (self.scales[0], self.scales[1], nx.copy(self.scales[2]), nx.copy(self.scales[3]))
                moe_copy.biases = (self.biases[0], self.biases[1], nx.copy(self.biases[2]), nx.copy(self.biases[3]))

        return moe_copy

    def quantize(self):
        if self.quantized:
            return
        
        self.quantized = True

        if self.n_experts > 0:
            self.Wcombined, scale_wcombined, bias_wcombined = nx.quantize(self.Wcombined)
            self.Wout, scale_wout, bias_out = nx.quantize(self.Wout)
            self.scales = (scale_wcombined, scale_wout, self.scales[2], self.scales[3])
            self.biases = (bias_wcombined, bias_out, self.biases[2], self.biases[3])

        if self.n_dense > 0:
            self.W_dense_combined, scale_wsharedcombined, bias_wsharedcombined = nx.quantize(self.W_dense_combined)
            self.W_dense_out, scale_wsharedout, bias_wsharedout = nx.quantize(self.W_dense_out)

            self.scales = (self.scales[0], self.scales[1], scale_wsharedcombined, scale_wsharedout)
            self.biases = (self.biases[0], self.biases[1], bias_wsharedcombined, bias_wsharedout)