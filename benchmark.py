import os
backend = os.environ["BACKEND"] = "auto"
import random
# import mlx.core as mx
EMBED_DIM = 128
CONTEXT_SIZE = 256
BATCH_SIZE = 32
BASE_WIDTH = 1024
N_HEADS = 4
# N_KV_HEADS = max(1, N_HEADS // 4)
N_EXPERTS = 8
CF = 1.25
VAL = .9
TOP_K = 2
LOADER_STRIDE = CONTEXT_SIZE
WINDOWS = CONTEXT_SIZE // 4

from pathlib import Path
import time

import cProfile
import pstats

from engine.transformer import Transformer
from engine.transformer_block import TransformerBlock
from engine.tokenizer import Tokenizer
from engine.embedding import Embedding
from engine.dataloader import DataLoader
from engine.sessions import Session
import engine.backend as nx
from helper.singleton import init_corpus

tokenizer1 = Tokenizer.load("artifacts/tokenizer/tokenizer24000_533726742len.tokenizer")

session_configs = {
    "context_size": CONTEXT_SIZE,
    "batch_size": BATCH_SIZE,
    "optimizer":"adamw",
    "train_split":VAL,
    "optimizer_args":{
        "lr": 1e-3,
        "use_master": False,
        "scheduler": "cosine_decay",
        "min_lr": 1e-5,
    },
    "using":backend,
}

model_configs = {
    "n_blocks":7,
    "embed_dim":EMBED_DIM,
    "dtype": nx.float16,
    "gradient_scale":4096,
    "vocab_size": len(tokenizer1.vocab),
    "moe_lambda":0.05,
    "block_configs":{
        "ff_hidden_width": BASE_WIDTH,
        "ff_n_experts":N_EXPERTS,
        "ff_topk":TOP_K,
        "ff_cf":CF,
        "ff_init":"glorot_uniform",
        "attn_type":"full",
        "attn_variant":"mha",
        "attn_n_heads":N_HEADS,
        # "attn_n_kv_heads":N_KV_HEADS,
        # "attn_windows":WINDOWS,
        "attn_init":"glorot_uniform",
        },
    "block_overrides":{
      }
}

weight_n = CONTEXT_SIZE * EMBED_DIM
real_vocab_size = len(tokenizer1.vocab)
model_configs["vocab_size"] = real_vocab_size

embedding1 = Embedding(real_vocab_size, EMBED_DIM)
transformer = Transformer(model_configs)
start = time.perf_counter()
session_configs["block_size"] = len(transformer.blocks)

print("loading dataloader", end="\r")
dataloader = DataLoader("data", tokenizer1, session_configs["context_size"], LOADER_STRIDE)

session1 = Session(transformer, tokenizer1, True, session_configs)

# profiler = cProfile.Profile()
# profiler.enable()
start = time.perf_counter()
# mx.metal.start_capture("transformer.gputrace")
session1.benchmark(dataloader, 1, 10)
end = time.perf_counter()
# mx.metal.stop_capture()
print(f"benchmarking finished. time: {end - start:.3f}s")

# profiler.disable()
# stats = pstats.Stats(profiler)
# stats.sort_stats("cumtime")
# stats.print_stats(100)




