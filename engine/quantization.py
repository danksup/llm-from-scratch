from typing import Any
import engine.backend as nx


def quantize(w:Any, dtype, axis:int=-1, keepdims=True):#, type:Literal['symmetric', 'asymetric']):
    if nx.issubdtype(w.dtype, nx.integer):
        return w

    info = nx.iinfo(dtype)
    q_min = info.min
    q_max = info.max

    max_abs = nx.maximum(nx.max(nx.abs(w), axis=axis,keepdims=True), 1e-9)
    scale = max_abs / q_max

    zero_point = 0

    quantized_w = nx.round(w / scale  + zero_point)
    quantized_w = nx.clip(quantized_w, q_min, q_max).astype(dtype)

    if not keepdims:
        scale = nx._nx.squeeze(scale, axis=axis)

    return quantized_w, scale, zero_point

def dequantize(quantized_w, scale:Any, dtype=nx.float32, zero_point:int=0):
    if nx.issubdtype(quantized_w.dtype, nx.floating) or scale is None:
        return quantized_w

    q_float = quantized_w.astype(dtype)

    return (q_float - zero_point) * scale

def quantized_matmul(a, quantized_w, scale):
    dequant_q = dequantize(quantized_w, scale, a.dtype)
    res = a @ dequant_q
    return res
