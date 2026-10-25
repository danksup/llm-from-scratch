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
    def __init__(self, session:Session, **configs) -> None:
        self.session = session
        self.transformer = session.transformer
        self.session_configs = self.session.configs
        self.tokenizer = self.session.tokenizer

        configs =  DEFAULT_CONFIGS | configs

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

    def infer(self, context:Any) -> Any:
        if self.temp < 0:
            warnings.warn(colorize("temp < 0 = greedy (like temp = 0)", "yellow"))

        as_symmetric = self.transformer.symmetric_quant
        logits, self.all_caches = self.transformer.inference(context,self.session_configs["context_size"], self.all_caches, self.position, use_symmetric=as_symmetric)
        self.position += context.shape[1]

        penalty_mem = nx.array(self.penalty_memory, nx.uint32)
        raw_token = self._sample(logits, penalty_mem,  self.temp, self.top_k, self.top_p, self.penalty)
        token = raw_token.item()
        self.penalty_memory.append(token)
        decoded = self.tokenizer.decode([token])
        print(decoded,end="",flush=True)
        if any(token in decoded for token in ("<EOT>", "<|endofdoc|>")):
            return

        next_token = nx.array([[token]], dtype=nx.int32)

        for _ in range(self.n-1):
            logits, self.all_caches = self.transformer.inference(next_token,self.session_configs["context_size"], self.all_caches, self.position, use_symmetric=as_symmetric)
            self.position += 1
            penalty_mem = nx.array(self.penalty_memory, nx.uint32)
            raw_token = self._sample(logits, penalty_mem,  self.temp, self.top_k, self.top_p, self.penalty)

            if len(self.penalty_memory) >= self.penalty_mem_size:
                self.penalty_memory =  self.penalty_memory[1:]

            token = raw_token.item()
            self.penalty_memory.append(token)

            decoded = self.tokenizer.decode([token])
            print(decoded,end="",flush=True)
            if any(token in decoded for token in ("<EOT>", "<|endofdoc|>")):
                break
            next_token = nx.array([[token]], dtype=nx.int32)
    
    @staticmethod
    # @nx.compile
    def _sample(logits, memory, temperature=0.8, top_k=3, top_p=0.9, penalty=0.05):
        if memory.size > 0:
            memory = nx.array(memory, dtype=nx.int32)
            logits = nx.substract_at(logits, (..., memory), penalty)

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
        mask = (cum - sorted_probs) < top_p

        keep_probs = nx.where(mask, sorted_probs, 0.0)
        keep_probs /= nx.sum(keep_probs)
        return nx.random_choice(sorted_indices, p=keep_probs)
