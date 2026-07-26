import engine.backend as nx
from typing import Any

def PE(context_size:int, embed_dim:int) -> Any:
    positions = nx.arange(context_size,dtype=nx.float32).reshape(-1,1)
    dims = nx.arange(0, embed_dim, 2, dtype=nx.float32)
    div_term = nx.power(nx.float_32(10000.0),nx.float_32(dims / embed_dim), dtype=nx.float32)

    pe = nx.zeros((context_size, embed_dim),dtype=nx.float32)
    pe[:, 0::2] = nx.sin(positions / div_term, dtype=nx.float32)  
    pe[:, 1::2] = nx.cos(positions / div_term,dtype=nx.float32)

    return pe