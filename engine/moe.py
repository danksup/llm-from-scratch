import math
from typing import Any, Callable

import engine.backend as nx
import engine.initializers as initializer
from engine.activations import softmax, softmax_derivative, swish, swish_derivative


class MoE:
    def __init__(self, capacity_factor, top_k, n_experts, embed_dim, hidden_width, dtype:Any=nx.float16, initializer:Callable=initializer.glorot_uniform, quantized:bool=False) -> None:
        self.hidden_width = hidden_width
        self.embed_dim = embed_dim #D
        self.n_experts = n_experts #E
        self.cf = capacity_factor
        self.top_k = top_k
        self.dtype = dtype

        self.configs = (hidden_width, embed_dim, n_experts, capacity_factor, top_k)

        router_shape = embed_dim, n_experts
        self.router = initializer(router_shape, dtype= nx.float32)
        self.d_router = None

        wcombined_shape =  (n_experts, embed_dim, hidden_width * 2)
        self.Wcombined = initializer(wcombined_shape, dtype=dtype)
        assert nx.isfinite(self.Wcombined).all(), f"non-finite detected when initializing moe.Wcombined."

        wout_shape =  (n_experts, hidden_width, embed_dim)
        self.Wout = initializer(wout_shape, dtype = dtype)
        assert nx.isfinite(self.Wout).all(), f"non-finite detected when initializing moe.Wout."

        self.scales = (None, None)
        self.quantized = quantized
        if quantized:
            self.Wcombined, wcombined_scale, _ = nx.quantize(self.Wcombined)
            self.Wout, wout_scale, _ = nx.quantize(self.Wout)
            self.scales = (wcombined_scale, wout_scale)

        self.dWcombined = None
        self.dWout = None

    @staticmethod
    def forward(x:nx.ArrayLike, ff_configs, ff_params, quantization:tuple[Any,...]|None=None):
        #routing
        # print("moe forward x", x.dtype)
        B, T, D = x.shape
        N = B * T

        H, _, n_experts, capacity_factor, top_k = ff_configs
        Wcombined, Wout, router = ff_params

        wcombined_scale, wout_scale = quantization  #type:ignore
        # Wcombined = dequantize(Wcombined, wcombined_scale, x.dtype)
        # Wout = dequantize(Wout, wout_scale, x.dtype)

        top_k = min(top_k, n_experts)
        capacity = math.ceil(capacity_factor * N * top_k / n_experts)
        flatten_x = x.reshape(-1, D)
        scores =  flatten_x.astype(nx.float32) @ router #(N, E)
        router_prob = softmax(scores, -1) #(N, E)

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
        router_loss = n_experts * nx.sum(normalized_histogram * avg_prob) #L, fp32

        #dispatch
        M = N * top_k
        assignment_rows = nx.arange(M, dtype=nx.int32)
        routing_mask = nx.zeros((M,n_experts), nx.int32)
        routing_mask = nx.add_at(routing_mask, (assignment_rows, flatten_top_expert_indices), 1)

        cum_assignment = nx.cumsum(routing_mask, axis=0, dtype=nx.int32)

        slot_idx = cum_assignment[assignment_rows, flatten_top_expert_indices] - 1 #(N,)

        valid = slot_idx < capacity

        masked_tokens = flatten_x[assignement_tokens] * valid[:, None] #type:ignore
        safe_slot = nx.clip(slot_idx, 0, capacity - 1, dtype=nx.int32)
        expert_input = nx.zeros((n_experts, capacity, D), dtype=x.dtype)
        expert_input = nx.add_at(expert_input, (flatten_top_expert_indices, safe_slot), masked_tokens)

        expert_gate = nx.zeros((n_experts, capacity), dtype=top_gates.dtype)
        safe_gates = nx.where(valid, flatten_top_gates, nx.zeros_like(flatten_top_gates))
        expert_gate = nx.add_at(expert_gate, (flatten_top_expert_indices, safe_slot), safe_gates)

        # projected = expert_input @ Wcombined
        projected = nx.quantized_matmul(expert_input, Wcombined, scales=wcombined_scale) #(E, capacity, 2H)
        gate_half = projected[..., :H]
        value_half = projected[..., H:]
        s = swish(gate_half, x.dtype)

        hidden = s * value_half #(E, capacity, H)
        # raw_output = hidden @ Wout #(E, capacity, D)
        raw_output = nx.quantized_matmul(hidden, Wout, scales=wout_scale) #(E, capacity, D)

        gated_output = raw_output * expert_gate[..., None]
        final_output = gated_output[flatten_top_expert_indices, safe_slot]
        final_output = final_output * valid[..., None]
        final_output = final_output.reshape(N,top_k,D)
        final_output = nx.sum(final_output, axis=1, dtype=nx.float32).reshape(B,T,D) #check
        final_output = final_output.astype(x.dtype)

        cache = (flatten_x, router_prob, top_expert_indices, top_gates32, flatten_top_expert_indices, assignement_tokens, valid, safe_slot, expert_input, expert_gate, projected, hidden, raw_output, normalized_histogram)
        return final_output, cache, router_loss, normalized_histogram


    @staticmethod
    def backward(gradient , caches, moe_configs, ff_params, quantization:tuple[Any,...]|None=None):
        flatten_x, router_prob, top_expert_indices, top_gates32 , flatten_top_expert_indices, assignement_tokens, valid, safe_slot, expert_input, expert_gate, projected, hidden, raw_output, normalized_histogram = caches
        Wout, Wcombined = ff_params

        wcombined_scale, wout_scale = quantization  #type:ignore

        capacity_factor, n_experts, hidden_width, router, LAMBDA = moe_configs
        top_k = top_expert_indices.shape[1]
        B,T,D = gradient.shape
        N = B*T
        M = N *top_k
        flatten_gradient = gradient.reshape(-1, D)
        assignment_gradient = flatten_gradient[assignement_tokens] #(M,D)
        capacity = math.ceil(capacity_factor * N * top_k / n_experts)
        d_masked_output = assignment_gradient * valid[...,None]

        del flatten_gradient, assignment_gradient

        d_gated_output = nx.zeros((n_experts, capacity, D), dtype=gradient.dtype)

        d_gated_output = nx.add_at(d_gated_output, (flatten_top_expert_indices, safe_slot), d_masked_output)

        del d_masked_output

        d_raw_output = d_gated_output * expert_gate[..., None] #(E, capacity, D) #fp16
        d_expert_gate = nx.sum(d_gated_output * raw_output, axis=-1, dtype=nx.float32,) #(E, capacity)

        dWout = hidden.transpose(0, 2, 1) @ d_raw_output

        if wout_scale is not None:
            d_hidden = nx.quantized_matmul(d_raw_output, Wout, wout_scale, transpose=True)  #(E,C,H) fp16
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
            d_expert_input = nx.quantized_matmul(d_projected, Wcombined, wcombined_scale, transpose=True)
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
        # print("d_scores", d_scores.dtype)


        d_router = flatten_x.astype(nx.float32).T @ d_scores #(D,E) #fp32
        # print("droter", d_router.dtype)
        d_x_router = d_scores @ router.T #(N, D)

        del normalized_histogram, d_avg_prob,flatten_x, d_scores, router

        # print("router", router.dtype)
        # print("dxrouter", d_x_router.dtype)
        d_x_expert = d_x_expert.reshape(N, top_k, D)
        d_x_expert = nx.sum(d_x_expert, axis=1, dtype=nx.float32) #(N,D)
        # print("dxpert", d_x_expert.dtype)
        dx_flat = d_x_expert + d_x_router #(N,D)

        dx = dx_flat.reshape(B,T,D)

        del dx_flat, top_expert_indices, valid, safe_slot

        return dx, dWcombined, dWout, d_router

    def to_dict(self) -> dict:
        moe_dict =  {
            "moe_configs":(self.cf, self.top_k, self.n_experts, self.hidden_width,self.embed_dim, nx.dtype_to_srt[self.dtype], self.quantized),
            "router": self.router.tolist(),
            "Wcombined":self.Wcombined.tolist(),
            "Wout":self.Wout.tolist(),
        }

        if self.quantized:
             moe_dict["scales"] = (self.scales[0].tolist(), self.scales[1].tolist()) #type:ignore
        else:
             moe_dict["scales"] = self.scales

        return moe_dict

    @classmethod
    def from_dict(cls, thing:dict) -> "MoE":
        capacity_factor, top_k, n_experts, hidden_width, embed_dim, dtype, is_quantized = thing["moe_configs"]
        dtype = nx.str_to_dtype[dtype]
        scales = thing["scales"]
        moe = cls(capacity_factor, top_k, n_experts, embed_dim, hidden_width, dtype, quantized=is_quantized)
        Wcombined = thing["Wcombined"]
        Wout = thing["Wout"]
        moe.router = nx.array(thing["router"], nx.float32)

        if is_quantized:
             moe.Wcombined = nx.array(Wcombined, dtype=nx.int8)
             moe.Wout = nx.array(Wout, dtype=nx.int8)
             moe.scales = (nx.array(scales[0], dtype=dtype), nx.array(scales[1], dtype=dtype))
        else:
             moe.Wcombined = nx.array(Wcombined, dtype=dtype)
             moe.Wout = nx.array(Wout, dtype=dtype)
        return moe
