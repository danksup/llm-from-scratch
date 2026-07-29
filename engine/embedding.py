import random
import json
import engine.backend as nx
from typing import Any


class Embedding:
    def __init__(self, n:int, embed_dim:int, dtype=nx.float16) -> None:
        self.embed_dim = embed_dim
        self.dtype = dtype
        init = 0.02
        self.lookup_table = nx.uniform(low=-init, high=init, size=(n, self.embed_dim), dtype=dtype)
    
    def __eq__(self, value: object) -> bool:
        if not isinstance(value, Embedding):
            return NotImplemented
        return (self.embed_dim == value.embed_dim) and (self.lookup_table == value.lookup_table)

    def forward(self, token_list:Any):
        ''' loopup and convert to the vector for each token id'''
        return self.lookup_table[token_list]
    
    def to_dict(self) -> dict[str, Any]:
        lookup = {"lookuptable": self.lookup_table.tolist(), "dtype":nx.dtype_to_srt[self.dtype]}
        return lookup

    @classmethod
    def from_dict(cls, thing:dict[str, Any]) -> "Embedding":
        lookuptable = nx.array(thing["lookuptable"], dtype=nx.str_to_dtype[thing["dtype"]])
        embedding = cls(lookuptable.shape[0],lookuptable.shape[1])
        embedding.lookup_table = lookuptable
        return embedding