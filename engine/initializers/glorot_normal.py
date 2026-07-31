import engine.backend as nx

def glorot_normal(shape:tuple[int,...], dtype=None, gain = 1.0):
    if dtype is None:
        dtype = nx.float32
    fan_in = shape[-2]
    fan_out = shape[-1]
    limit = gain * nx.sqrt(2/(fan_in + fan_out))
    return nx.normal(0.0, limit, size=shape, dtype=dtype)