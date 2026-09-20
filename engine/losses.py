from typing import Any

import engine.backend as nx


@nx.compile
def cross_entropy(logits:Any, target_logits:Any):
    '''
    |  |I

    ||   |_
    '''
    logits = logits.astype(nx.float32)
    flat_logits = logits.reshape(-1, logits.shape[-1])
    rows = nx.arange(flat_logits.shape[0], dtype=nx.int32)
    flat_target = target_logits.reshape(-1)
    correct_logits = flat_logits[rows, flat_target]
    loss = nx.logsumexp(flat_logits, axis=-1)  - correct_logits
    return loss.reshape(target_logits.shape)

@nx.compile
def cross_entropy_gradient(logits:Any , target_logits:Any) -> Any:
    '''
    derivative of cross entropy and softmax
    '''
    probs = nx.softmax(logits)
    probs_copy = nx.copy(probs)
    flat_prob = probs_copy.reshape(-1, probs.shape[-1])
    flat_target = target_logits.reshape(-1)
    rows = nx.arange(flat_prob.shape[0], dtype=nx.int32)
    flat_prob[rows, flat_target] -= 1.0
    return flat_prob.reshape(probs.shape)
