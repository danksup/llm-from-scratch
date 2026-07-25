import engine.backend as nx

def precompute_freqs(head_dim, max_seq_len=1024):
    positions = nx.arange(max_seq_len, dtype=nx.float16)
    dims = nx.arange(0,head_dim,2, dtype=nx.float16)
    theta = 1.0 / (10000 ** (dims / head_dim))
    angles = positions[:, None] * theta[None,:]
    return angles

def rope_forward(x, angles, position=0):
    T = x.shape[2]    
    current_angles = angles[position:position+ T]

    sin = nx.sin(current_angles, dtype=x.dtype)
    cos = nx.cos(current_angles, dtype=x.dtype)
    x1 = x[..., ::2] 
    x2 = x[..., 1::2]
    
    rotated_x1 = (x1 * cos) - (x2 * sin)
    rotated_x2 = (x1 * sin) + (x2 * cos)
    
    stacked = nx.stack([rotated_x1, rotated_x2], axis=-1)
    return stacked.reshape(x.shape)

def rope_inverse(x,angles):
    T = x.shape[2]
    current_angles = angles[:T,:]

    sin = nx.sin(current_angles, dtype=x.dtype)
    cos = nx.cos(current_angles, dtype=x.dtype)

    x1 = x[..., ::2] 
    x2 = x[..., 1::2]  
    
    rotated_x1 = (x1 * cos) + (x2 * sin)
    rotated_x2 = -(x1 * sin) + (x2 * cos)
    
    stacked = nx.stack([rotated_x1, rotated_x2], axis=-1)
    return stacked.reshape(x.shape)