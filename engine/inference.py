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

        self.attn_rope_scale = configs["attn_rope_scale"]
        self.transformer.set_rope_scale(self.attn_rope_scale)

        self.trunk_all_caches = None
        self.position = 0
        self.penalty_memory = []

        self.mtp_all_caches = None

        self.mtp_acceptance = 0

    def infer(self, context:Any) -> Any:
        max_cache_len = self.session_configs["context_size"] * self.attn_rope_scale
        T = context.shape[1]
        if self.temp < 0:
            warnings.warn(colorize("temp < 0 = greedy (like temp = 0)", "yellow"))

        logits, self.trunk_all_caches, output = self.transformer.inference(context,max_cache_len, self.trunk_all_caches, self.position)

        penalty_mem = nx.array(self.penalty_memory, nx.uint32)
        raw_token = self._sample(logits, penalty_mem,  self.temp, self.top_k, self.top_p, self.penalty)
        token = raw_token.item()
        decoded = self.tokenizer.decode([token])
        print(decoded,end="",flush=True)
        if any(token in decoded for token in ("<EOT>", "<|endofdoc|>")):
            return

        confirmed = token
        n_generated = 1

        penalty_mem = self.update_penalty_mem(token)
        tokens = [confirmed]
        draft = None
        if self.transformer.mtp_blocks:
            draft = self.make_draft(max_cache_len, output, raw_token, self.position + T - 1, penalty_mem)
            tokens.append(draft)

        self.position += context.shape[1]

        while n_generated < self.n:
            tokens_array = nx.array(tokens, nx.uint32).reshape(1, -1)
            logits, self.trunk_all_caches, output =  self.transformer.inference(tokens_array,max_cache_len, self.trunk_all_caches, self.position)
            s = self._sample(logits[:,:1], penalty_mem, self.temp, self.top_k, self.top_p, self.penalty)
            s_item = s.item()
            penalty_mem = self.update_penalty_mem(s_item)

            if s_item == draft:
                decoded = self.tokenizer.decode([draft]) #type:ignore
                print(decoded,end="",flush=True)
                if any(token in decoded for token in ("<EOT>", "<|endofdoc|>")):
                    return

                next_confirmed = logits[:, 1:]
                next_s = self._sample(next_confirmed, penalty_mem, self.temp, self.top_k, self.top_p, self.penalty)
                confirmed = next_s.item()
                decoded = self.tokenizer.decode([confirmed]) #type:ignore
                print(decoded,end="",flush=True)
                if any(token in decoded for token in ("<EOT>", "<|endofdoc|>")):
                    return

                penalty_mem = self.update_penalty_mem(confirmed)

                draft = self.make_draft(max_cache_len, output[:, 1:, :], confirmed, self.position + 1, penalty_mem)
                tokens = [confirmed, draft]
                self.mtp_acceptance += 1
                self.position += 2
                n_generated += 2


            else:
                decoded = self.tokenizer.decode([s_item])
                print(decoded,end="",flush=True)
                if any(token in decoded for token in ("<EOT>", "<|endofdoc|>")):
                    return

                tokens = [s_item]

                if self.transformer.mtp_blocks:
                    draft = self.make_draft(max_cache_len, output[:, :1, :], s_item, self.position, penalty_mem)
                    tokens.append(draft)

                self.position += 1
                n_generated += 1

            if len(self.penalty_memory) >= self.penalty_mem_size:
                overfill = len(self.penalty_memory) - self.penalty_mem_size
                self.penalty_memory =  self.penalty_memory[1+overfill:]

    def make_draft(self, max_cache_len, output, token, pos, penalty_mem):
        mtp_logits, self.mtp_all_caches = self.transformer.mtp_inference(output, token, max_cache_len, self.mtp_all_caches, pos)
        raw_mtp_token = self._sample(mtp_logits, penalty_mem, self.temp, self.top_k, self.top_p, 0)
        unverified_token = raw_mtp_token.item()
        return unverified_token

    def update_penalty_mem(self, token):
        self.penalty_memory.append(token)
        penalty_mem = nx.array(self.penalty_memory, nx.uint32)
        return penalty_mem

    @staticmethod
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
