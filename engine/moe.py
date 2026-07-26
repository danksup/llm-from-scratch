import engine.backend as nx
from engine.activations import softmax, softmax_derivative
from engine.activations import swish,swish_derivative
import time
import math
from typing import Any

class MoE:
    def __init__(self, capacity_factor, top_k, n_experts, embed_dim, hidden_width, dtype:Any=nx.float16) -> None:
        self.hidden_width = hidden_width 
        self.embed_dim = embed_dim #D
        self.n_experts = n_experts #E
        self.cf = capacity_factor
        self.top_k = top_k
        self.dtype = dtype

        init = nx.sqrt(6 / (embed_dim + n_experts), dtype=nx.float32) 
        self.router = nx.uniform(-init,init,(self.embed_dim, self.n_experts), dtype=init.dtype)
        self.d_router = None

        scale = nx.sqrt(6 / (embed_dim+hidden_width), dtype=dtype)
        self.Wcombined = nx.uniform(-scale,scale, (n_experts, hidden_width * 2, embed_dim),dtype=dtype)
        self.Wout = nx.uniform(-scale,scale, (n_experts, hidden_width, embed_dim), dtype=dtype)

        self.dWcombined = None
        self.dWout = None

    @staticmethod
    def forward(x:nx.ArrayLike, capacity_factor, top_k, router:nx.ArrayLike, n_expert:int,H:int,  Wcombined, Wout):
        #routing
        # print("moe forward x", x.dtype)
        B, T, D = x.shape 
        N = B * T
        top_k = min(top_k, n_expert)
        capacity = math.ceil(capacity_factor * N * top_k / n_expert)
        flatten_x = x.reshape(-1, D)
        scores =  flatten_x.astype(nx.float32) @ router #(N, E) #type:ignore
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
        
        histogram = nx.zeros(n_expert, dtype=nx.int32)
        histogram = nx.add_at(histogram, flatten_top_expert_indices, 1) 
        avg_prob = nx.mean(router_prob, axis=0) #P
        normalized_histogram = histogram / ( N * top_k) #f
        router_loss = n_expert * nx.sum(normalized_histogram * avg_prob) #L, fp32

        #dispatch
        M = N * top_k
        assignment_rows = nx.arange(M, dtype=nx.int32)
        routing_mask = nx.zeros((M,n_expert), nx.int32)
        routing_mask = nx.add_at(routing_mask, (assignment_rows, flatten_top_expert_indices), 1)

        cum_assignment = nx.cumsum(routing_mask, axis=0, dtype=nx.int32)

        slot_idx = cum_assignment[assignment_rows, flatten_top_expert_indices] - 1 #type:ignore #(N,)

        valid = slot_idx < capacity
        masked_tokens = flatten_x[assignement_tokens] * valid[:, None] #type:ignore
        safe_slot = nx.clip(slot_idx, 0, capacity - 1, dtype=nx.int32)
        expert_input = nx.zeros((n_expert, capacity, D), dtype=x.dtype)
        expert_input = nx.add_at(expert_input, (flatten_top_expert_indices, safe_slot), masked_tokens)

        # start = time.perf_counter()
        expert_gate = nx.zeros((n_expert, capacity), dtype=top_gates.dtype)
        safe_gates = nx.where(valid, flatten_top_gates, nx.zeros_like(flatten_top_gates))
        expert_gate = nx.add_at(expert_gate, (flatten_top_expert_indices, safe_slot), safe_gates)
        # nx.eval(expert_gate)
        # end = time.perf_counter()
        # print(f"gate {end-start}")

        projected = expert_input @ Wcombined.transpose(0,2,1) #(E, capacity, 2H)
        gate_half = projected[..., :H]
        value_half = projected[..., H:]
        s = swish(gate_half, x.dtype)
        # print("swish", s.dtype)
        hidden = s * value_half #(E, capacity, H) 
        raw_output = hidden @ Wout #(E, capacity, D) 
        # print("raw output", raw_output.dtype)
        gated_output = raw_output * expert_gate[..., None]
        final_output = gated_output[flatten_top_expert_indices, safe_slot]
        final_output = final_output * valid[..., None]
        final_output = final_output.reshape(N,top_k,D)
        final_output = nx.sum(final_output, axis=1, dtype=nx.float32).reshape(B,T,D) #check
        final_output = final_output.astype(x.dtype)
        # print("final output", final_output.dtype)
        # print("expert_input",expert_input.dtype)
        # print("gate",expert_gate.dtype)
        # print("projected",projected.dtype)
        # print("hidden",hidden.dtype)
        # print("router forward", router.dtype)
        cache = (flatten_x, router_prob, top_expert_indices, top_gates32, flatten_top_expert_indices, assignement_tokens, valid, safe_slot, expert_input, expert_gate, projected, hidden, raw_output, normalized_histogram)
        return final_output, cache, router_loss, normalized_histogram

    @staticmethod
    def backward(gradient , caches, moe_configs, ff_params):
        flatten_x, router_prob, top_expert_indices, top_gates32 , flatten_top_expert_indices, assignement_tokens, valid, safe_slot, expert_input, expert_gate, projected, hidden, raw_output, normalized_histogram = caches
        Wout, Wcombined = ff_params
        capacity_factor, n_experts, hidden_width, router, LAMBDA = moe_configs
        top_k = top_expert_indices.shape[1]
        B,T,D = gradient.shape
        N = B*T
        M = N *top_k
        flatten_gradient = gradient.reshape(-1, D)
        assignment_gradient = flatten_gradient[assignement_tokens] #(M,D)
        capacity = math.ceil(capacity_factor * N * top_k / n_experts)
        d_masked_output = assignment_gradient * valid[...,None]

        d_gated_output = nx.zeros((n_experts, capacity, D), dtype=gradient.dtype)
        # start = time.perf_counter()
        d_gated_output = nx.add_at(d_gated_output, (flatten_top_expert_indices, safe_slot), d_masked_output)
        # nx.eval(d_gated_output)
        # end = time.perf_counter()
        # print("d_gated_output", end-start)

        d_raw_output = d_gated_output * expert_gate[..., None] #(E, capacity, D) #fp16
        d_expert_gate = nx.sum(d_gated_output * raw_output, axis=-1, dtype=nx.float32,) #(E, capacity)
       
        dWout = hidden.transpose(0, 2, 1) @ d_raw_output
        
        # start = time.perf_counter()
        d_hidden = d_raw_output @ Wout.transpose(0, 2, 1) #(E,C,H) fp16
        # nx.eval(d_hidden)
        # end = time.perf_counter()
        # print("d_hidden", end-start)

        gate_half = projected[..., :hidden_width]
        value_half = projected[..., hidden_width:]

        d_gate_half = d_hidden * value_half * swish_derivative(gate_half, dtype=gate_half.dtype)
        d_value_half = d_hidden * swish(gate_half, dtype=value_half.dtype)
        d_projected = nx.concatenate([d_gate_half, d_value_half], axis=-1) #(E, C, 2H)  fp16


        # start = time.perf_counter()
        #expertinput = (E, C, D)
        dWcombined = d_projected.transpose(0, 2, 1) @ expert_input #(E, 2H, D) fp16
        # nx.eval(dWcombined)
        # end = time.perf_counter()
        # print("dWcombined", end-start)

        # start = time.perf_counter()
        d_expert_input = d_projected @ Wcombined #(E, C, D) fp16
        # nx.eval(d_expert_input)
        # end = time.perf_counter()
        # print("d_expert_input", end-start)

        d_x_expert = d_expert_input[flatten_top_expert_indices, safe_slot]
        d_x_expert *= valid[...,None]

        d_chosen_gate = d_expert_gate[flatten_top_expert_indices, safe_slot]
        d_chosen_gate *= valid

        d_chosen_gate = d_chosen_gate.reshape(N,top_k)
        token_rows = nx.arange(N, dtype=nx.int32)[:,None]
        selected_prob = router_prob[token_rows, top_expert_indices] #N,K
        gate_sum = nx.sum(selected_prob, -1, keepdims=True, dtype=nx.float32) #(N,1)
        coupling = nx.sum(d_chosen_gate * top_gates32, -1, keepdims=True, dtype=nx.float32) #(N,1)
        d_selected_prob = (d_chosen_gate - coupling)/gate_sum #(N,K)
        d_selected_prob = d_selected_prob.reshape(-1,)

        d_router_prob = nx.zeros((N,n_experts), dtype=d_selected_prob.dtype) #fp32
        d_router_prob[assignement_tokens, flatten_top_expert_indices] = d_selected_prob

        d_avg_prob = n_experts * normalized_histogram
        d_router_prob += LAMBDA * (d_avg_prob / N)

        d_scores = softmax_derivative(router_prob, d_router_prob) #(N,E)
        # print("d_scores", d_scores.dtype)

        d_router = flatten_x.astype(nx.float32).T @ d_scores #(D,E) #fp32
        # print("droter", d_router.dtype)
        d_x_router = d_scores @ router.T #(N, D)
        # print("router", router.dtype)
        # print("dxrouter", d_x_router.dtype)
        d_x_expert = d_x_expert.reshape(N, top_k, D)
        d_x_expert = nx.sum(d_x_expert, axis=1, dtype=nx.float32) #(N,D)
        # print("dxpert", d_x_expert.dtype)
        dx_flat = d_x_expert + d_x_router #(N,D)

        dx = dx_flat.reshape(B,T,D) 
        return dx, dWcombined, dWout, d_router

    def to_dict(self) -> dict:
        return {
            "moe_configs":(self.cf, self.top_k, self.n_experts, self.hidden_width,self.embed_dim, nx.dtype_to_srt[self.dtype]),
            "router": self.router.tolist(),
            "Wcombined":self.Wcombined.tolist(),
            "Wout":self.Wout.tolist(),
        }
    
    @classmethod
    def from_dict(cls, thing:dict) -> "MoE":
        capacity_factor, top_k, n_experts, hidden_width, embed_dim, dtype = thing["moe_configs"]
        dtype = nx.str_to_dtype[dtype]
        moe = cls(capacity_factor, top_k, n_experts, embed_dim, hidden_width, dtype)
        moe.Wcombined = nx.array(thing["Wcombined"], dtype=moe.dtype)
        moe.Wout = nx.array(thing["Wout"], dtype=moe.dtype)
        moe.router = nx.array(thing["router"], dtype=nx.float32)
        return moe
