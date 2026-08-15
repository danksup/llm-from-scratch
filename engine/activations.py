import engine.backend as nx
from typing import Any

def leaky_relu(x:Any) -> Any:
    """
    f(x) = max(0.01,x)
    """
    return nx.where(x > 0, x, nx.float_32(0.01) * x)  #irl, x is unlikely to be 0 so we can ignore it

def leaky_relu_derivative(x:Any) -> Any:
    """
    derivative of leaky relu
    """
    one = nx.float_32(1.0)
    alpha = nx.float_32(0.01)
    return nx.where(x > 0, one, alpha)

def softmax(x:Any, axis:Any=-1) -> Any:
    """
    turns logit into probability distribution that sums into 1.\n
    f(x) = exp(x - max(x)) / sum(exp(x - max(x)))
    """
    x = x.astype(nx.float32)
    max_x = nx.max(x, axis=axis, keepdims=True)
    exp_x = nx.exp(x - max_x, dtype=nx.float32)
    return exp_x / nx.sum(exp_x, axis=axis, keepdims=True, dtype=nx.float32)

def softmax_derivative(s:Any, grad:Any) -> Any:
    """
    derivative of softmax
    """
    return  s * (grad -nx.sum(grad * s, axis=-1, keepdims=True,dtype=nx.float32))

def sigmoid(x:Any, dtype=None) -> Any:
    """
    turns x into value between 0 and 1 \n
    f(x) = 1 / (1 + e**(-x))
    """
    if dtype is None:
        dtype = x.dtype
    z = nx.exp(-nx.abs(x), dtype=dtype)
    return nx.where(x>=0, 1.0 / (1.0 + z), z / (1.0 + z))

def sigmoid_derivative(x:Any):
    s = sigmoid(x)
    return s * (1.0 - s)

def swish(x:Any, dtype=None) -> Any:
    """
    f(x) = x * sigmoid(Bx)\n
    B, beta, is 1 by default.
    """
    if dtype is None:
        dtype = x.dtype
    return x * sigmoid(x, dtype)

def swish_derivative(x: Any, dtype=None) -> Any:
    """
    derivative of swish
    f(x) = σ(x) + x * σ(x) * (1 - σ)
    """
    if dtype is None:
        dtype = x.dtype
    s = sigmoid(x, dtype=dtype)
    return s + x * s * (1.0 - s)
