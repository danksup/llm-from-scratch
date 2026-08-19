from typing import Any
from uuid import NAMESPACE_X500

import engine.backend as nx

class Embedding:
    def __init__(self, n:int, embed_dim:int, dtype=nx.float16, quantized:bool=False) -> None:
        self.embed_dim = embed_dim
        self.dtype = dtype
        init = 0.02
        self.lookup_table = nx.uniform(low=-init, high=init, size=(n, self.embed_dim), dtype=dtype)

        self.table_scale = None
        self.quantized = quantized
        if quantized:
            self.lookup_table, self.table_scale, _ = nx.quantize(self.lookup_table)
        assert nx.isfinite(self.lookup_table).all(), f"non-finite detected when initializing embedding."

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, Embedding):
            return NotImplemented
        return (self.embed_dim == value.embed_dim) and (self.lookup_table == value.lookup_table)

    def forward(self, token_list:Any):
        ''' loopup and convert to the vector for each token id'''
        embed = self.lookup_table[token_list]
        # print(embed)
        qtized = nx.dequantize(embed, scales=self.table_scale[token_list],dtype=self.dtype) if self.table_scale is not None else embed
        # print(qtized)
        return qtized

    def to_dict(self) -> dict[str, Any]:
        lookup = {"lookuptable": self.lookup_table.tolist(), "quantized":self.quantized, "dtype":nx.dtype_to_srt[self.dtype]}
        if self.table_scale is not None:
            lookup["table_scale"] = self.table_scale.tolist()
        else:
            lookup["table_scale"] = self.table_scale
        return lookup

    @classmethod
    def from_dict(cls, thing:dict[str, Any]) -> "Embedding":
        is_quantized = thing["quantized"]
        dtype = nx.str_to_dtype[thing["dtype"]]
        lookup_table = thing["lookuptable"]
        scale = thing["table_scale"]
        embedding = cls(len(lookup_table), len(lookup_table[0]), dtype, is_quantized)

        if is_quantized:
            lookup_table = nx.array(lookup_table, dtype=nx.int8)
            scale = nx.array(scale, dtype=dtype)
            embedding.lookup_table = lookup_table
        else:
            lookup_table = nx.array(lookup_table, dtype=dtype)
            embedding.lookup_table = lookup_table

        return embedding
