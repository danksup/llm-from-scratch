from typing import Any
import engine.backend as nx

def fused_cross_entropy(input:Any, embedding:Any, target_logits:Any, chunk_size=512, is_training=True, gradient_scale=1):
    B,T,D = input.shape
    N= B*T
    input = input.astype(nx.float32)
    vocab_size = embedding.shape[0]
    flatten_input = input.reshape(-1, D)
    flat_target = target_logits.reshape(-1)

    total_loss = nx.array(0, nx.float32)
    total_block_gradient = []
    total_d_table = nx.zeros_like(embedding, dtype=nx.float32)
    for i in range(0, N, chunk_size):
        flat_target_ = flat_target[i:i+chunk_size]
        logit = flatten_input[i:i+chunk_size,...] @ embedding.T #C,V
        rows = nx.arange(logit.shape[0], dtype=nx.int32)
        correct_logits = logit[rows, flat_target_]
        loss = nx.logsumexp(logit, axis=-1)  - correct_logits
        total_loss += nx.sum(loss)
        nx.eval(*[total_loss])

        if is_training:
            probs = nx.softmax(logit)
            probs[rows, flat_target_] -= 1.0
            scale_factor = nx.array(gradient_scale / N, input.dtype)
            scaled_probs = probs * scale_factor
            probs = scaled_probs.astype(input.dtype)
            total_block_gradient.append(probs @ embedding) #dtype
            d_table = probs.reshape(-1, vocab_size).T @ flatten_input[i:i+chunk_size,...]
            d_table = d_table.astype(nx.float32) / gradient_scale
            total_d_table += d_table
            nx.eval(*[total_block_gradient, total_d_table])

    if is_training:
        return total_loss / N, nx.concatenate(total_block_gradient, 0).reshape(B,T,D), total_d_table
    else:
        return total_loss / N