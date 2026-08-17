import engine.backend as nx
from engine.quantization import quantize, dequantize
from typing import Any

class Embedding:
    def __init__(self, n:int, embed_dim:int, dtype=nx.float16, quantized:bool=False) -> None:
        self.embed_dim = embed_dim
        self.dtype = dtype
        init = 0.02
        self.lookup_table = nx.uniform(low=-init, high=init, size=(n, self.embed_dim), dtype=dtype)
        
        self.table_scale = None
        if quantized:
            self.lookup_table, self.table_scale, _ = quantize(self.lookup_table, nx.int8, keepdims=True)
        assert nx.isfinite(self.lookup_table).all(), f"non-finite detected when initializing embedding."
    
    def __eq__(self, value: object) -> bool:
        if not isinstance(value, Embedding):
            return NotImplemented
        return (self.embed_dim == value.embed_dim) and (self.lookup_table == value.lookup_table)

    def forward(self, token_list:Any):
        ''' loopup and convert to the vector for each token id'''
        embed = self.lookup_table[token_list]
        # print(embed)
        qtized = dequantize(embed, self.table_scale[token_list], self.dtype) if self.table_scale is not None else embed
        # print(qtized)
        return qtized
    
    def to_dict(self) -> dict[str, Any]:
        lookup = {"lookuptable": self.lookup_table.tolist(), "dtype":nx.dtype_to_srt[self.dtype]}
        return lookup

    @classmethod
    def from_dict(cls, thing:dict[str, Any]) -> "Embedding":
        lookuptable = nx.array(thing["lookuptable"], dtype=nx.str_to_dtype[thing["dtype"]])
        embedding = cls(lookuptable.shape[0],lookuptable.shape[1])
        embedding.lookup_table = lookuptable
        return embedding