from typing import Any

import engine.backend as nx

class Embedding:
    def __init__(self, n:int, embed_dim:int, dtype=nx.float16, quantized:bool|str=False, *, init=True) -> None:
        self.n = n
        self.embed_dim = embed_dim
        self.dtype = dtype
        self.quantized = quantized

        table_shape = n, self.embed_dim
        if init:
            init_ = 0.02
            self.lookup_table = nx.uniform(low=-init_, high=init_, size=table_shape, dtype=dtype)

            self.table_scale = None
            self.bias = None
            if quantized:
                self.lookup_table, self.table_scale, self.bias = nx.quantize(self.lookup_table)

        self.d_lookup_table = nx.zeros(table_shape)

    def zeroes_gradient(self):
        self.d_lookup_table = nx.zeros_like(self.d_lookup_table)

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, Embedding):
            return NotImplemented
        return (self.embed_dim == value.embed_dim) and (self.lookup_table == value.lookup_table)

    def forward(self, token_list:Any):
        ''' loopup and convert to the vector for each token id'''
        embed = self.lookup_table[token_list]
        if self.quantized:
            qtized = nx.dequantize(embed, scales=self.table_scale[token_list], biases=self.bias[token_list], dtype=self.dtype) if self.table_scale is not None else embed #type:ignore
            return qtized
        return embed

    def get_configs(self):
        return(self.n, self.embed_dim)

    @classmethod
    def from_weights(cls, lookuptable, quants, dtype):
        n = len(lookuptable)
        D = lookuptable.shape[0]
        quantized = True if quants is not None else False
        embedding = cls(n, D, dtype=dtype, init=False, quantized=quantized)
        embedding.lookup_table = lookuptable

        if quants is not None:
            embedding.table_scale, embedding.bias= quants
        return embedding

    def copy(self):
        embedding_copy = Embedding(self.n, self.embed_dim, self.dtype, self.quantized, init=False)
        embedding_copy.lookup_table = nx.copy(self.lookup_table)

        if self.quantized:
            embedding_copy.table_scale = nx.copy(self.table_scale)
            embedding_copy.bias = nx.copy(self.bias)

        return embedding_copy