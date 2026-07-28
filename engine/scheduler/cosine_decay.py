import engine.backend as nx

def cosine_decay(start, end, rate):
    return end + 0.5 * (start - end) * (1 + nx.cos(nx.pi * rate))