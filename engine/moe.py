import math
from typing import Any, Callable

import engine.backend as nx
import engine.initializers as initializer
from engine.activations import softmax_derivative, swish, swish_derivative
from helper.singleton import sleep

C = nx.array(1e-4)

class MoE:
    def __init__(self, capacity_factor, top_k, n_experts, embed_dim, hidden_width, shared_hidden_width, n_shared_experts=2,jitter_eps=0.02,LAMBDA =1e-2, dtype:Any=nx.float16, initializer:Callable=initializer.glorot_uniform, quantized:bool|str=False, *, init=True) -> None:
        self.hidden_width = hidden_width
        self.embed_dim = embed_dim #D
        self.n_experts = n_experts #E
        self.cf = capacity_factor
        self.top_k = top_k
        self.dtype = dtype
        self.n_shared_experts = n_shared_experts
        self.shared_hidden_width = shared_hidden_width
        self.jitter_eps = jitter_eps
        self.LAMBDA = LAMBDA

        self.configs = (hidden_width, embed_dim, n_experts, capacity_factor, top_k,shared_hidden_width, n_shared_experts,jitter_eps, LAMBDA)
        self.quantized = quantized

        self.scales = (None, None, None, None)
        self.biases = (None, None, None, None)

        wcombined_shape =  (n_experts, embed_dim, hidden_width * 2)
        wout_shape =  (n_experts, hidden_width, embed_dim)
        router_shape = embed_dim, n_experts
        shared_expert_shape = embed_dim, n_shared_experts * shared_hidden_width * 2
        shared_out_shape = (n_shared_experts * shared_hidden_width, embed_dim)
        
        if init:
            self.router = initializer(router_shape, dtype= nx.float32)

            self.Wcombined = initializer(wcombined_shape, dtype=dtype)
            assert nx.isfinite(self.Wcombined).all(), f"non-finite detected when initializing moe.Wcombined."

            self.Wout = initializer(wout_shape, dtype = dtype)
            assert nx.isfinite(self.Wout).all(), f"non-finite detected when initializing moe.Wout."

            self.W_shared_combined = initializer(shared_expert_shape, dtype=dtype)
            self.W_shared_out = initializer(shared_out_shape, dtype=dtype)

            if quantized:
                self.Wcombined, wcombined_scale, wcombined_biases = nx.quantize(self.Wcombined)
                self.Wout, wout_scale, wout_biases = nx.quantize(self.Wout)
                self.W_shared_combined, shared_expert_scale, shared_expert_bias = nx.quantize(self.W_shared_combined)
                self.W_shared_out, shared_expert_out_scale, shared_expert_out_bias = nx.quantize(self.W_shared_out)
                self.scales = (wcombined_scale, wout_scale, shared_expert_scale, shared_expert_out_scale)
                self.biases = (wcombined_biases, wout_biases, shared_expert_bias,shared_expert_out_bias)

        self.dWcombined = nx.zeros(wcombined_shape)
        self.dWout = nx.zeros(wout_shape)
        self.d_router = nx.zeros(router_shape, dtype=nx.float32)
        self.dW_shared_combined = nx.zeros(shared_expert_shape)
        self.dW_shared_out = nx.zeros(shared_out_shape)

    def zeroes_gradient(self):
        self.dWcombined = nx.zeros_like(self.dWcombined)
        self.dWout = nx.zeros_like(self.dWout)
        self.d_router = nx.zeros_like(self.d_router, dtype=nx.float32)
        self.dW_shared_combined = nx.zeros_like(self.dW_shared_combined)
        self.dW_shared_out = nx.zeros_like(self.dW_shared_out)


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

        _, _, n_experts, capacity_factor, top_k, _, _,jitter_eps, LAMBDA = ff_configs
        Wcombined, Wout, router, W_shared_combined, W_shared_out  = ff_params

        wcombined_scale, wout_scale,shared_expert_scale,shared_expert_out_scale, wcombined_bias, wout_bias, shared_expert_bias,  shared_expert_out_bias = quantization  #type:ignore

        top_k = min(top_k, n_experts)

        capacity = math.ceil(capacity_factor * N * top_k / n_experts)
            
        flatten_x = x.reshape(-1, D)
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

        expert_input = MoE.compute_expert_input(n_experts, top_k, N, flatten_x, valid, safe_slot, capacity, D, flatten_top_expert_indices, x.dtype)

        expert_gate = nx.zeros((n_experts, capacity), dtype=top_gates.dtype)
        safe_gates = nx.where(valid, flatten_top_gates, nx.zeros_like(flatten_top_gates))
        expert_gate = nx.add_at(expert_gate, (flatten_top_expert_indices, safe_slot), safe_gates)

        projected = MoE.compute_projected(expert_input, Wcombined, (wcombined_scale, wcombined_bias)) #(E, capacity, 2H)
        shared_projected = MoE.compute_projected(flatten_x, W_shared_combined, (shared_expert_scale,shared_expert_bias)) #(N, ns * 2Hs)

        hidden = MoE.compute_hidden(projected, x.dtype) #(E, capacity, H)
        shared_hidden = MoE.compute_hidden(shared_projected, x.dtype) #N, ns * H

        if wout_scale is not None:
            raw_output = nx.quantized_matmul(hidden, Wout, scales=wout_scale, biases=wout_bias) #(E, capacity, D)
        else:
            raw_output = hidden @ Wout #(E, capacity, D)

        if shared_expert_out_scale is not None:
            raw_shared_output = nx.quantized_matmul(shared_hidden,W_shared_out, scales=shared_expert_out_scale, biases=shared_expert_out_bias)
        else:
            raw_shared_output = shared_hidden @ W_shared_out #N, D

        shared_output = raw_shared_output.reshape(B,T,D)

        gated_output = raw_output * expert_gate[..., None]
        final_output = gated_output[flatten_top_expert_indices, safe_slot]
        final_output = final_output * valid[..., None]
        final_output = final_output.reshape(N,top_k,D)
        final_output = nx.sum(final_output, axis=1, dtype=nx.float32).reshape(B,T,D) #check
        final_output = final_output.astype(x.dtype) + shared_output

        total_aux_loss = router_loss + z_loss

        if not recompute_activation:
            cache = (flatten_x, router_prob, top_expert_indices, top_gates32, flatten_top_expert_indices, assignement_tokens, valid, safe_slot, expert_input, expert_gate, projected, hidden, raw_output, normalized_histogram, scores, shared_projected, shared_hidden)
        else:
            cache = (flatten_x, router_prob, top_expert_indices, top_gates32, flatten_top_expert_indices, assignement_tokens, valid, safe_slot, expert_gate, raw_output, normalized_histogram, scores)

        return final_output, cache, total_aux_loss, normalized_histogram

    @staticmethod
    def backward(gradient , caches, moe_configs, ff_params, gradient_scale, quantization:tuple[Any,...]|None=None, *, recompute_activation:bool=True):
        B,T,D = gradient.shape
        N = B*T
        
        wcombined_scale, wout_scale,shared_expert_scale,shared_expert_out_scale, wcombined_bias, wout_bias, shared_expert_bias,  shared_expert_out_bias = quantization  #type:ignore

        router, Wout, Wcombined, W_shared_combined, W_shared_out = ff_params

        hidden_width, D, n_experts, capacity_factor, top_k,shared_hidden_width, n_shared_experts, _, LAMBDA = moe_configs
        capacity = math.ceil(capacity_factor * N * top_k / n_experts)

        if not recompute_activation:
            flatten_x, router_prob, top_expert_indices, top_gates32, flatten_top_expert_indices, assignement_tokens, valid, safe_slot, expert_input, expert_gate, projected, hidden, raw_output, normalized_histogram, scores, shared_projected, shared_hidden = caches
        else:
            flatten_x, router_prob, top_expert_indices, top_gates32 , flatten_top_expert_indices, assignement_tokens, valid, safe_slot, expert_gate, raw_output, normalized_histogram, scores = caches
            expert_input = MoE.compute_expert_input(n_experts, top_k, N, flatten_x, valid, safe_slot, capacity, D, flatten_top_expert_indices, gradient.dtype)
            projected = MoE.compute_projected(expert_input, Wcombined, (wcombined_scale,wcombined_bias))
            hidden = MoE.compute_hidden(projected, gradient.dtype)

            shared_projected = MoE.compute_projected(flatten_x, W_shared_combined, (shared_expert_scale,shared_expert_bias)) #(N,2H)
            shared_hidden = MoE.compute_hidden(shared_projected, gradient.dtype) #N, H

        M = N *top_k
       
        flatten_gradient = gradient.reshape(-1, D)
        assignment_gradient = flatten_gradient[assignement_tokens] #(M,D)
        d_masked_output = assignment_gradient * valid[...,None]

        if shared_expert_out_scale is not None:
            d_shared_hidden = nx.quantized_matmul(flatten_gradient, W_shared_out, shared_expert_out_scale, shared_expert_out_bias, True)
        else:
            d_shared_hidden = flatten_gradient @ W_shared_out.T #N,H

        d_gated_output = nx.zeros((n_experts, capacity, D), dtype=gradient.dtype)

        d_gated_output = nx.add_at(d_gated_output, (flatten_top_expert_indices, safe_slot), d_masked_output)

        d_raw_output = d_gated_output * expert_gate[..., None] #(E, capacity, D) #fp16
        d_expert_gate = nx.sum(d_gated_output * raw_output, axis=-1, dtype=nx.float32,) #(E, capacity)

        dWout = hidden.transpose(0, 2, 1) @ d_raw_output

        dW_shared_out = shared_hidden.T @ flatten_gradient #H, D
        
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

        shared_gate_half = shared_projected[..., :shared_hidden_width * n_shared_experts] #N,H
        shared_value_half = shared_projected[..., shared_hidden_width* n_shared_experts:]

        d_shared_gate_half = d_shared_hidden * shared_value_half * swish_derivative(shared_gate_half, dtype=shared_gate_half.dtype)
        d_shared_value_half = d_shared_hidden * swish(shared_gate_half, dtype=shared_value_half.dtype)
        d_shared_projected = nx.concatenate([d_shared_gate_half, d_shared_value_half], axis=-1) #N,2H

        del projected, d_hidden, gate_half, value_half, d_gate_half, d_value_half

        dWcombined = expert_input.transpose(0, 2, 1) @ d_projected #(E,D,2H) fp16

        dW_shared_combined = flatten_x.T @ d_shared_projected #D, 2H

        if wcombined_scale is not None:
            d_expert_input = nx.quantized_matmul(d_projected, Wcombined, wcombined_scale,biases=wcombined_bias, transpose=(0,2,1))
        else:
            d_expert_input = d_projected @ Wcombined.transpose(0,2,1) #(E, C, D) fp16

        if shared_expert_scale is not None:
            d_x_shared = nx.quantized_matmul(d_shared_projected, W_shared_combined, shared_expert_scale, shared_expert_bias, transpose=True)
        else:
            d_x_shared = d_shared_projected @ W_shared_combined.T #N,D

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
        # print("d_scores", d_scores.dtype)
        d_scores += d_z_loss * gradient_scale


        d_router = flatten_x.astype(nx.float32).T @ d_scores #(D,E) #fp32
        # print("droter", d_router.dtype)
        d_x_router = d_scores @ router.T #(N, D)

        del normalized_histogram, d_avg_prob,flatten_x, d_scores, router

        # print("router", router.dtype)
        # print("dxrouter", d_x_router.dtype)
        d_x_expert = d_x_expert.reshape(N, top_k, D)
        d_x_expert = nx.sum(d_x_expert, axis=1, dtype=nx.float32) #(N,D)
        # print("dxpert", d_x_expert.dtype)
        dx_flat = d_x_expert + d_x_router + d_x_shared #(N,D)

        dx = dx_flat.reshape(B,T,D)

        del dx_flat, top_expert_indices, valid, safe_slot

        return dx, dWcombined, dWout, d_router, dW_shared_combined, dW_shared_out

    @staticmethod
    def inference_forward(x:nx.ArrayLike, ff_configs, ff_params, quantization:tuple[Any,...]|None=None):
        #routing
        B, T, D = x.shape
        N = B * T

        _, _, n_experts, capacity_factor, top_k, _, _, _, _ = ff_configs

        Wcombined, Wout, router, W_shared_combined, W_shared_out  = ff_params

        wcombined_scale, wout_scale,shared_expert_scale,shared_expert_out_scale, wcombined_bias, wout_bias, shared_expert_bias,  shared_expert_out_bias = quantization  #type:ignore

        top_k = min(top_k, n_experts)

        flatten_x = x.reshape(-1, D)
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
                
            projected = MoE.compute_projected(flatten_x[None,:,:], Wcombined, (wcombined_scale, wcombined_bias))

            hidden = MoE.compute_hidden(projected, x.dtype) #(E, capacity, H)

            if wout_scale is not None:
                raw_output = nx.quantized_matmul(hidden, Wout, scales=wout_scale, biases=wout_bias) #(E, capacity, D)
            else:
                raw_output = hidden @ Wout #(E, capacity, D)

            final_output = raw_output * flatten_top_gates[:, None, None]
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
    
            expert_input = MoE.compute_expert_input(n_experts, top_k, N, flatten_x, valid, safe_slot, capacity, D, flatten_top_expert_indices, x.dtype) #(n_experts, capacity, D)
    
            expert_gate = nx.zeros((n_experts, capacity), dtype=top_gates.dtype)
            safe_gates = nx.where(valid, flatten_top_gates, nx.zeros_like(flatten_top_gates))
            expert_gate = nx.add_at(expert_gate, (flatten_top_expert_indices, safe_slot), safe_gates)
            
            projected = MoE.compute_projected(expert_input, Wcombined, (wcombined_scale, wcombined_bias)) #(E, capacity, 2H)

            hidden = MoE.compute_hidden(projected, x.dtype) #(E, capacity, H)

            if wout_scale is not None:
                raw_output = nx.quantized_matmul(hidden, Wout, scales=wout_scale, biases=wout_bias) #(E, capacity, D)
            else:
                raw_output = hidden @ Wout #(E, capacity, D)

            gated_output = raw_output * expert_gate[..., None]
            final_output = gated_output[flatten_top_expert_indices, safe_slot]
            final_output = final_output * valid[..., None]
        
        final_output = final_output.reshape(N,top_k,D)

        final_output = nx.sum(final_output, axis=1, dtype=nx.float32).reshape(B,T,D) #check

        shared_projected = MoE.compute_projected(flatten_x, W_shared_combined, (shared_expert_scale,shared_expert_bias)) #(N, ns * 2Hs)
        
        shared_hidden = MoE.compute_hidden(shared_projected, x.dtype) #N, ns * H

        if shared_expert_out_scale is not None:
            raw_shared_output = nx.quantized_matmul(shared_hidden,W_shared_out, scales=shared_expert_out_scale, biases=shared_expert_out_bias)
        else:
            raw_shared_output = shared_hidden @ W_shared_out #N, D

        shared_output = raw_shared_output.reshape(B,T,D)
        final_output = final_output.astype(x.dtype) + shared_output

        return final_output

    @classmethod
    def from_weight(cls, configs, weights, quants, dtype) -> "MoE":
        hidden_width, embed_dim, n_experts, capacity_factor, top_k, shared_hidden_width, n_shared_experts, jitter_eps, LAMBDA = configs
        router, wcombined, wout,W_shared_combined,W_shared_out = weights

        moe = cls(capacity_factor, top_k, n_experts, embed_dim, hidden_width, shared_hidden_width, n_shared_experts,jitter_eps,LAMBDA, dtype, init=False)
        moe.router = router
        moe.Wcombined = wcombined
        moe.Wout = wout
        moe.W_shared_combined = W_shared_combined
        moe.W_shared_out = W_shared_out

        if quants is not None:
            scales, biases = quants
            moe.scales = scales
            moe.biases = biases

        return moe

    def copy(self):
        moe_copy = MoE(self.cf, self.top_k, self.n_experts, self.embed_dim, self.hidden_width,self.shared_hidden_width,self.n_shared_experts,self.jitter_eps,self.LAMBDA, dtype=self.dtype, quantized=self.quantized, init=False)
        moe_copy.router = nx.copy(self.router)
        moe_copy.Wcombined = nx.copy(self.Wcombined)
        moe_copy.Wout = nx.copy(self.Wout)
        moe_copy.W_shared_combined = nx.copy(self.W_shared_combined)
        moe_copy.W_shared_out = nx.copy(self.W_shared_out)

        if self.quantized:
            moe_copy.scales = (nx.copy(self.scales[0]), nx.copy(self.scales[1]), nx.copy(self.scales[2]), nx.copy(self.scales[3]))
            moe_copy.biases = (nx.copy(self.biases[0]), nx.copy(self.biases[1]), nx.copy(self.biases[2]), nx.copy(self.biases[3]))

        return moe_copy