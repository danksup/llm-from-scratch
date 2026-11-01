import engine.backend as nx
from engine.rmsnorm import RMSNorm
from engine.transformer_block import TransformerBlock
import engine.initializers as initializer
class MTPBlock:
    def __init__(self,embed_dim, transformer_block:TransformerBlock, rmsnorm1:RMSNorm, rmsnorm2:RMSNorm, dtype, initializer=initializer.glorot_uniform, quantized=False, *, init=True) -> None:
        self.embed_dim = embed_dim
        self.rmsnorm1 = rmsnorm1
        self.rmsnorm2 = rmsnorm2
        self.transformer_block = transformer_block

        self.quantized = quantized
        self.scale = self.bias = None

        projection_shape = 2*embed_dim, embed_dim
        if init:
            self.Wproj = initializer((projection_shape), dtype=dtype)

            if quantized:
                self.Wproj, proj_scale, proj_bias = nx.quantize(self.Wproj)
                self.scale = proj_scale
                self.bias = proj_bias

        self.dWproj = nx.zeros(projection_shape, dtype=dtype)

    def zeroes_gradient(self):
        self.dWproj = nx.zeros_like(self.dWproj)
        self.transformer_block.zeroes_gradient()
        self.rmsnorm2.zeroes_gradient()
        self.rmsnorm1.zeroes_gradient()

    @staticmethod
    def forward():
        pass

    @staticmethod
    def backward():
        pass