# pyright: reportArgumentType=false
# pyright: reportCallIssue=false
# pyright: reportAttributeAccessIssue=false
# pyright: reportFunctionMemberAccess=false

from typing import Any, Union,Literal
import os
backend = os.environ.get("BACKEND", "auto").lower()

if backend == "auto":
    try:
        import mlx.core as _nx
        backend = "MLX"
    except ImportError:
        import numpy as _nx
        backend = "NumPy"
        rng = _nx.random.default_rng()
elif backend == "mlx":
    import mlx.core as _nx
    backend = "MLX"
else:
    import numpy as _nx
    backend = "NumPy"
    rng = _nx.random.default_rng()

ArrayLike = Any
e = _nx.e
inf = _nx.inf
nan = _nx.nan
pi = _nx.pi
float16 = _nx.float16
float32 = _nx.float32
float64 = _nx.float64
int64 = _nx.int64
int32 = _nx.int32
int16 = _nx.int16
int8 = _nx.int8
bool_ = _nx.bool_
bfloat16 = _nx.bfloat16 if backend == "MLX" else _nx.float32

dtype_to_srt = {
    float16: "float16",
    float32: "float32",
    int64: "int64",
    int32: "int32",
    bool_: "bool",
    bfloat16: "bf16",
}

str_to_dtype = {
    "float16":float16,
    "float32":float32,
    "int64":int64,
    "int32":int32,
    "bool":bool_,
    "bf16":bfloat16,
}
    
def array(x:Any, dtype=None) -> ArrayLike:
    if dtype is None:
        dtype = float32
    return _nx.array(x, dtype=dtype)

def sum(a:ArrayLike, axis:Any=None, keepdims:bool=False, dtype=None) -> Any:
    if dtype is None:
        dtype = a.dtype
    if backend == "MLX":
        if dtype is not None:
            a = a.astype(dtype)
        return _nx.sum(a, axis=axis, keepdims=keepdims)
    
    return _nx.sum(a,axis=axis,keepdims=keepdims,dtype=dtype)

def float_32(x:list | ArrayLike | float) -> Any:
    if backend == "NumPy":
        return _nx.float32(x)
    return _nx.array(x, dtype=float32)

def add_at(a:Any, idx:Any, values:Any) -> Any:
    if backend == "MLX":
        return a.at[idx].add(values)
    if backend == "NumPy":
        if isinstance(idx, slice) or (isinstance(idx, tuple) and any(isinstance(i, slice) for i in idx)):
            a[idx] += values
            return a
        _nx.add.at(a, idx, values)
    return a

def unique(a:ArrayLike, return_counts:bool=False ) -> Any:
    """
    note: always flatten
    """
    if backend == "MLX":
        flatten:list = _nx.sort(a.reshape(-1)).tolist() #type:ignore
        unique_arr = []
        counts = []
        
        for i in flatten:
            if not unique_arr or i != unique_arr[-1]:
                unique_arr.append(i)
                counts.append(1)
            else:
                counts[-1] += 1
        
        unique_arr = _nx.array(unique_arr, dtype=a.dtype)
        counts = _nx.array(counts, dtype=int32)
        if return_counts:
            return (unique_arr,counts)
        return unique_arr
    unique_array =  _nx.unique(a, return_counts=return_counts)
    return unique_array

def zeros_like( a:ArrayLike, dtype=None) -> ArrayLike:
    if dtype is not None and dtype != a.dtype:
        a = a.astype(dtype)
    if backend == "MLX":
        return _nx.zeros_like(a)
    return _nx.zeros_like(a, dtype=dtype)

def zeros(size:Any, dtype=None) -> ArrayLike:
    if dtype is None:
        dtype = float32
    return _nx.zeros(size, dtype=dtype)

def ones_like( a:ArrayLike, dtype=None) -> ArrayLike:
    if dtype is not None and dtype != a.dtype:
        a = a.astype(dtype)
    if backend == "MLX":
        return _nx.ones_like(a)
    return _nx.ones_like(a, dtype=dtype)

def ones(size:Any, dtype=None) -> ArrayLike:
    if dtype is None:
        dtype = _nx.float32
    return _nx.ones(size, dtype=dtype)

def where( condition:Any,x:Any,y:Any) -> ArrayLike:
    return _nx.where(condition,x,y)

def triu( x:ArrayLike, k=0) -> ArrayLike:
    return _nx.triu(x, k=k)

def max( x:ArrayLike, axis=None, keepdims:bool=False) -> ArrayLike:
    return _nx.max(x, axis=axis, keepdims=keepdims)

def min( x:ArrayLike, axis=None, keepdims:bool=False) -> ArrayLike:
    return _nx.min(x,axis=axis, keepdims=keepdims)

def exp(x:ArrayLike, dtype=None) -> ArrayLike:
    if dtype is not None and dtype != x.dtype:
        x = x.astype(dtype)
    return _nx.exp(x)

def clip( a:Any, a_min:Any, a_max:Any, dtype:Any=None) -> ArrayLike:
    if dtype is not None and dtype != a.dtype:
        a = a.astype(dtype)
    if backend == "MLX":
        return _nx.clip(a, a_min, a_max)
    return _nx.clip(a,a_min, a_max,dtype=dtype)

def log(x:ArrayLike, *,axis=None, keepdims:bool=False,dtype=None) -> ArrayLike:
    if dtype is not None and dtype != x.dtype:
        x = x.astype(dtype)
    if backend == "MLX":
        return _nx.log(x, axis=axis,keepdims=keepdims)
    return _nx.log(x, axis=axis,keepdims=keepdims, dtype=dtype)

def arange(start:Union[int,float], stop:Union[None,int,float]=None, step:Union[None,int,float]=None, *,dtype=None) -> ArrayLike:
    if stop is None:
        return _nx.arange(start, dtype=dtype)
    else:
        if step is None:
             return _nx.arange(start,stop, dtype=dtype)

    return _nx.arange(start, stop, step, dtype=dtype)

def indices( x:Any) -> Any:
    if backend == "NumPy":
        return _nx.indices(x)
    ndims = len(x)
    result = []
    for axis, size in enumerate(x):
        idx = _nx.arange(size)
        new_shape = [1] * ndims
        new_shape[axis] = size
        idx = idx.reshape(new_shape)
        idx = _nx.broadcast_to(idx, x)
        result.append(idx)
    return _nx.stack(result)

def sqrt(x:Any, dtype=None) -> Any:
    if backend == "MLX":
        if hasattr(x, "dtype"): 
            if dtype is not None and x.dtype != dtype:
                x = x.astype(dtype)
        else: 
            x = _nx.array(x, dtype)
        return _nx.sqrt(x)
    return _nx.sqrt(x,dtype=dtype)   

def uniform( low:float=0, high:float=1, size=None,*,dtype=None) -> Any:
    if dtype is None:
        dtype = _nx.float32
    if backend == "NumPy":
        if size is None:
            return dtype(rng.uniform(low,high))
        else:
            return rng.uniform(low,high, size=size).astype(dtype)

    if size is None:
        size = ()
    return _nx.random.uniform(low, high, shape=size, dtype=dtype)

def permutation(x:Any, axis=0) -> ArrayLike:
    if isinstance(x,int):
        if backend == "NumPy":
            return rng.permutation(x)
        return _nx.random.permutation(x)
    else:
        if backend == "NumPy":
            return rng.permutation(x, axis=axis)
    return _nx.random.permutation(x, axis=axis)

def as_strided(x, shape, strides):
    if backend == 'NumPy':
        x = _nx.asarray(x)
        itemsize = x.dtype.itemsize
        np_stride = tuple(s * itemsize for s in strides)
        return _nx.lib.stride_tricks.as_strided(x, shape=shape, strides=np_stride)
    return _nx.as_strided(x,shape, strides)

def mean(x:ArrayLike, *,axis=None, keepdims:bool=False,dtype=None) -> ArrayLike:
    if dtype is not None and dtype != x.dtype:
        x = x.astype(dtype)
    if backend == "MLX":
        return _nx.mean(x, axis=axis,keepdims=keepdims)
    return _nx.mean(x, axis=axis,keepdims=keepdims, dtype=dtype)

def var( x:ArrayLike,*,axis=None,keepdims=False,dtype=None) -> ArrayLike:
    if dtype is not None and dtype != x.dtype:
        x = x.astype(dtype)
    if backend == "MLX":
        return _nx.var(x, axis=axis,keepdims=keepdims)
    return _nx.var(x, axis=axis,keepdims=keepdims, dtype=dtype)

def dot(u:ArrayLike,v:ArrayLike) -> Any:
    return u @ v

def sin(x:Any,dtype=None) -> Any:
    if dtype is not None and dtype != x.dtype:
        x = x.astype(dtype)
    if backend =="MLX":
        return _nx.sin(x)
    return _nx.sin(x, dtype=dtype)
    
def cos(x:Any,dtype=None) -> Any:
    if dtype is not None and dtype != x.dtype:
        x = x.astype(dtype)
    if backend =="MLX":
        return _nx.cos(x)
    return _nx.cos(x, dtype=dtype)

def power(a,b, dtype=None) -> Any:
    if dtype is None:
        dtype = _nx.float32
    if backend =="MLX":
        a_arr = a if hasattr(a, "dtype") else _nx.array(a, dtype)
        b_arr = b if hasattr(b, "dtype") else _nx.array(b, dtype)
        if a_arr.dtype != dtype: a_arr = a_arr.astype(dtype)
        if b_arr.dtype != dtype: b_arr = b_arr.astype(dtype)
        return _nx.power(a_arr, b_arr)
    return _nx.power(a,b,dtype=dtype)

def argpartition(x:ArrayLike, kth, axis=None) -> ArrayLike:
    return _nx.argpartition(x,kth, axis=axis)

def random_choice(a:ArrayLike, *, p=None) -> Any:
    if backend == "MLX":
        cdf = _nx.cumsum(p)
        r = _nx.random.uniform()
        idx = _nx.argmax(cdf >= r)
        return a[idx]
    return _nx.random.choice(a, p=p)

def copy(x:Any) -> Any:
    if backend == "MLX":
        return _nx.array(x)
    return x.copy()

def eval( *args):
    if backend == "MLX":
        _nx.eval(*args)
    pass

def concatenate(a:list[ArrayLike], axis=None) -> ArrayLike:
    return _nx.concatenate(a,axis=axis)

def cumsum(a:ArrayLike, *,axis=None, dtype=None) -> ArrayLike:
    if dtype is None:
        dtype = _nx.float32
    if backend == "MLX":
        a = _nx.array(a, dtype=dtype)
        return _nx.cumsum(a, axis=axis)
    return _nx.cumsum(a, axis=axis, dtype=dtype)

def argsort( a:ArrayLike, axis:None|int=-1) -> Any:
    return _nx.argsort(a, axis=axis)

def all( a:ArrayLike, *,axis=None, keepdims:bool=False) -> ArrayLike:
    return _nx.all(a, axis=axis, keepdims=keepdims)

def argmax( a:ArrayLike, axis=None, keepdims:bool=False) -> Any:
    return _nx.argmax(a, axis=axis, keepdims=keepdims)

def abs( a:Any) -> Any:
    return _nx.abs(a)

def stack(a:Any, axis=0)-> Any:
    return _nx.stack(a, axis=axis)

def any( a, *,keepdims:bool=False,axis=None):
    return _nx.any(a, keepdims=keepdims, axis=axis)

def isnan( a):
    return _nx.isnan(a)

def compile( fn=None):
    if backend == "MLX":
        return _nx.compile(fn) if fn is not None else _nx.compile
    def no_op_decorator(f):
        return f
    return no_op_decorator(fn) if fn is not None else no_op_decorator

def clear_cache():
    if backend == "MLX":
        _nx.clear_cache()

def repeat(a, repeats:int, axis:int=None):
    return _nx.repeat(a,repeats, axis=axis)

def logsumexp(a:ArrayLike,*,axis=None,keepdims=False) -> ArrayLike:
    if backend == "NumPy":
        m = _nx.max(a, axis=axis, keepdims=True)
        out = m + _nx.log(
            _nx.sum(_nx.exp(a - m), axis=axis, keepdims=True)
        )
        if not keepdims and axis is not None:
            out = _nx.squeeze(out, axis=axis)
        return out
    return _nx.logsumexp(a, axis=axis, keepdims=keepdims)

def norm(a:ArrayLike, ord:Any=None, axis=None, keepdims:bool=False) -> Any:
    return _nx.linalg.norm(a, ord, axis=axis, keepdims=keepdims)

def pad(a:ArrayLike, pad_width:int| tuple[int]| tuple[int,int]|list[tuple[int,int]],mode:Literal['constant', 'edge']='constant', constant_value:Any=0) -> ArrayLike:
    return _nx.pad(a, pad_width=pad_width, mode=mode, constant_values=constant_value)

def split(a:ArrayLike,indices_or_sections, axis=None)-> list[ArrayLike]:
    return _nx.split(a, indices_or_sections, axis=axis) #type:ignore

def ceil(a):
    return _nx.ceil(a)

def floor(a):
    return _nx.floor(a)

def topk(a, k:int, axis:int = -1, return_element:bool=False):
    top_k = _nx.argpartition(a, kth=-k, axis=axis)[..., -k:]
    if return_element:
        if backend == "MLX":
            return _nx.topk(a, k=k, axis=axis)
        return _nx.take_along_axis(a, top_k, axis=axis)
    return top_k

def tile(a, reps:int):
    return _nx.tile(a, reps)

def broadcast_to(a, shape):
    return _nx.broadcast_to(a, shape=shape)

def einsum(subscripts, *operands):
    return _nx.einsum(subscripts, *operands)

def get_active_memory() -> int:
    if backend == "NumPy":
        return 0
    return _nx.get_active_memory()

def get_cache_memory() -> int:
    if backend == "NumPy":
        return 0
    return _nx.get_cache_memory()

def eye(n,m=None,k=0,dtype=None):
    if dtype is None:
        dtype = _nx.float32
    if m is None:
        m = n
    return _nx.eye(n, m, k, dtype=dtype)

def full(shape, vals, dtype=None):
    if dtype is None:
        dtype = _nx.float32
    return _nx.full(shape, vals, dtype=dtype)

def isfinite(a):
    return _nx.isfinite(a)

def isinf(a):
    return _nx.isinf(a)

