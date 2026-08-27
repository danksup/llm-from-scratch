from typing import Any

import engine.backend as nx

class Embedding:
    def __init__(self, n:int, embed_dim:int, dtype=nx.float16, quantized:bool|str=False, *, use_symmetric=False, init=True) -> None:
        self.n = n
        self.embed_dim = embed_dim
        self.dtype = dtype
        self.quantized = quantized

        if init:
            init_ = 0.02
            self.lookup_table = nx.uniform(low=-init_, high=init_, size=(n, self.embed_dim), dtype=dtype)

            self.table_scale = None
            self.use_symmetric = use_symmetric
            self.bias = None
            if quantized:
                self.lookup_table, self.table_scale, self.bias = nx.quantize(self.lookup_table, regular=use_symmetric)

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, Embedding):
            return NotImplemented
        return (self.embed_dim == value.embed_dim) and (self.lookup_table == value.lookup_table)

    def forward(self, token_list:Any):
        ''' loopup and convert to the vector for each token id'''
        embed = self.lookup_table[token_list]
        # print(embed)
        if self.quantized:
            qtized = nx.dequantize(embed, scales=self.table_scale[token_list], biases=self.bias[token_list], dtype=self.dtype, regular=self.use_symmetric) if self.table_scale is not None else embed #type:ignore
            # print(qtized)
            return qtized
        return embed

    def to_dict(self, *, as_symmetric=False) -> dict[str, Any]:
        lookup = {"n":self.n, "embed_dim":self.embed_dim, "lookuptable": self.lookup_table.tolist(), "quantized":self.quantized, "dtype":nx.dtype_to_srt[self.dtype]}
        if self.quantized:
            if as_symmetric:
                lookup_table, table_scale, bias = nx.quantize(nx.dequantize(self.lookup_table, self.table_scale, self.bias, self.dtype), regular=as_symmetric)
                lookup["table_scale"] = table_scale.tolist() #type:ignore
                lookup["bias"] = bias.tolist() #type:ignore
                lookup["lookuptable"] = lookup_table.tolist()
            else:
                lookup["table_scale"] = self.table_scale.tolist() #type:ignore
                lookup["bias"] = self.bias.tolist() #type:ignore
        else:
            lookup["table_scale"] = self.table_scale
            lookup["bias"] = self.bias
        return lookup

    @classmethod
    def from_dict(cls, thing:dict[str, Any],  *, use_symmetric=False) -> "Embedding":
        is_quantized = thing["quantized"]
        dtype = nx.str_to_dtype[thing["dtype"]]
        lookup_table = thing["lookuptable"]
        scale = thing["table_scale"]
        bias = thing["bias"]
        embedding = cls(thing["n"],thing["embed_dim"], dtype, is_quantized, use_symmetric=use_symmetric)

        if is_quantized:
            if nx.backend == "MLX" and not use_symmetric:
                embedding.lookup_table = nx.array(lookup_table, dtype=nx.uint32)
            else:
                embedding.lookup_table  = nx.array(lookup_table, dtype=nx.int8)
            embedding.table_scale = nx.array(scale, dtype=dtype)
            embedding.bias = nx.array(bias, dtype=dtype)
        else:
            embedding.lookup_table =  nx.array(lookup_table, dtype=dtype)

        return embedding

    def get_configs(self):
        return(self.n, self.embed_dim)

    @classmethod
    def from_weights(cls, lookuptable, quants, dtype):
        n = len(lookuptable)
        D = lookuptable.shape[0]
        embedding = cls(n, D, dtype=dtype, init=False)
        embedding.lookup_table = lookuptable

        if quants is not None:
            embedding.table_scale, embedding.bias= quants
        return embedding