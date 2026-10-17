import engine.backend as nx
from engine.tokenizer import Tokenizer
from engine.sessions import Session
from typing import Any
import warnings
from helper.singleton import colorize

DEFAULT_CONFIGS = {
    "seed":12345,
    "n":200,
    "top_k": 50,
    "temp":0.5,
    "top_p":0.9,
    "penalty_mem_size":32,
    "penalty":.25
}

class Inference:
    def __init__(self, session:Session, **kwargs) -> None:
        self.session = session
        self.transformer = session.transformer
        self.session_configs = self.session.configs
        self.tokenizer = self.session.tokenizer

        configs = kwargs | DEFAULT_CONFIGS

        self.seed = configs["seed"]
        self.top_k = configs["top_k"]
        self.n = configs["n"]
        self.temp = configs["temp"]
        self.top_p = configs["top_p"]
        self.penalty_mem_size = configs["penalty_mem_size"]
        self.penalty = configs["penalty"]

        self.all_caches = None
        self.position = 0
        self.penalty_memory = []

    def infer(self, context:Any, temperature=0.8, top_k=3, top_p=0.9, n=100, mem_size=16, penalty:float=.05) -> Any:
        if temperature < 0:
            warnings.warn(colorize("temp < 0 = greedy (like temp = 0)", "yellow"))

        all_caches = self.all_caches
        position = self.position
        memory = self.penalty_memory
        as_symmetric = self.transformer.symmetric_quant
        logits, all_caches = self.transformer.inference(context,self.session_configs["context_size"], all_caches, position, use_symmetric=as_symmetric)
        position = context.shape[1]

        raw_token = self._sample(logits, memory,  temperature, top_k, top_p, penalty)
        memory.append(raw_token)
        token = raw_token.item()
        decoded = self.tokenizer.decode([token])
        print(decoded,end="",flush=True)
        if any(token in decoded for token in ("<EOT>", "<|endofdoc|>")):
            return

        next_token = nx.array([[token]], dtype=nx.int32)

        for _ in range(n-1):
            logits, all_caches = self.transformer.inference(next_token,self.session_configs["context_size"], all_caches, position, use_symmetric=as_symmetric)
            raw_token = self._sample(logits, memory, temperature, top_k, top_p, penalty)

            if len(memory) >= mem_size:
                memory = memory[1:]
            memory.append(raw_token)

            token = raw_token.item()
            decoded = self.tokenizer.decode([token])
            print(decoded,end="",flush=True)
            if any(token in decoded for token in ("<EOT>", "<|endofdoc|>")):
                break
            next_token = nx.array([[token]], dtype=nx.int32)
            position += 1
    
    @staticmethod
    def _sample(logits, memory, temperature=0.8, top_k=3, top_p=0.9, penalty=0.05):
        if memory:
            memory = nx.array(memory, dtype=nx.int32)
            mem_array = nx.unique(memory, return_counts=True)
            mem_unique = mem_array[0]
            mem_count = mem_array[1]
            logits = nx.substract_at(logits, (..., mem_unique), mem_count * penalty)

        if temperature <= 0:
            return nx.argmax(logits[0, -1], -1)
        else:
            probs = nx.softmax(logits[0, -1]/temperature)

        #top k
        top_k = min(top_k, len(probs))
        top_indices = nx.argpartition(probs, -top_k)[-top_k:]

        #top p
        probs = probs[top_indices]
        sorted_order = nx.argsort(probs)[::-1]
        sorted_probs = probs[sorted_order]
        sorted_indices = top_indices[sorted_order]
        cum = nx.cumsum(sorted_probs)
        mask = cum <= top_p

        cutoff = int(nx.sum(mask))
        cutoff = min(cutoff + 1, len(sorted_probs))
        mask[0] = True
        if not bool(nx.all(mask)):
            first_false = nx.argmax(~mask)
            mask[first_false] = True

        keep_indices = sorted_indices[:cutoff]
        keep_probs = sorted_probs[:cutoff]
        keep_probs /= nx.sum(keep_probs)
        return nx.random_choice(keep_indices, p=keep_probs)
