import os
import math
import random

backend = os.environ["BACKEND"] = "auto"
EPOCHS = 50
EMBED_DIM = 256
CONTEXT_SIZE = 256
BATCH_SIZE = 128
BASE_WIDTH = 512#4 * EMBED_DIM 
N_HEADS = 16
N_KV_HEADS = N_HEADS // 4
N_EXPERTS = 16
CF = 1.25
VAL = .9
TOP_K = 2
LOADER_STRIDE = CONTEXT_SIZE
WINDOWS = CONTEXT_SIZE // 4

#not hooked yet to session
PATIENCE = 20
TRESHOLD = 1e-2

TOKENIZER_PATH = "artifacts/tokenizer/tokenizer8192_33414037len.tokenizer"

from pathlib import Path
import time

import cProfile
import pstats

from engine.transformer import Transformer
from engine.tokenizer import Tokenizer
from engine.dataloader import DataLoader
from engine.sessions import Session
from engine.scheduler.cosine_decay import CosineDecay
from engine.scheduler.linear_schedule import LinearSchedule
import engine.backend as nx

from helper.singleton import init_corpus

session_configs = {
    "epochs":EPOCHS,
    "context_size": CONTEXT_SIZE,
    "batch_size": BATCH_SIZE,
    "dataloader_strides":LOADER_STRIDE,
    "optimizer":"adamw",
    "train_split":VAL,
    "optimizer_args":{
        "lr": 5e-3,
        "use_master": False,
        "scheduler": CosineDecay,
        "min_lr": 5e-4,
    },
    "using":backend,
    "save":True
}

model_configs = {
    "n_blocks":5,
    "embed_dim":EMBED_DIM,
    "dtype": nx.float16,
    "gradient_scale":1024,
    "block_configs":{"ff_hidden_width": BASE_WIDTH,"ff_n_experts":N_EXPERTS,"ff_topk":TOP_K,"ff_cf":CF,"attn_n_heads":N_HEADS,"attn_n_kv_heads":N_KV_HEADS, "attn_windows":WINDOWS},
    "block_overrides":{
        0: {"attn_windows":CONTEXT_SIZE}, 2:{"ff_n_experts": N_EXPERTS//2},3:{"ff_n_experts": N_EXPERTS//2}, 4:{"ff_n_experts": N_EXPERTS//4}
    }
}

corpus, files = init_corpus("data")

tokenizer1 = Tokenizer.load(TOKENIZER_PATH)

session_configs["dataset"] = f"{len(files)} files"
real_vocab_size = len(tokenizer1.vocab)

transformer = Transformer(model_configs)

session_configs["block_size"] = len(transformer.blocks)

print("loading dataloader ", end="\r")
dataloader = DataLoader(corpus, tokenizer1, session_configs["context_size"], stride=LOADER_STRIDE)

corpus_len = len(corpus)
ratio = dataloader.get_compression_rate()
token_size = dataloader.get_token_size()
session_configs["corpus char len"] = f"{corpus_len} -> BPE compression ({len(tokenizer1.vocab)} vocab size): {token_size}. ratio = {ratio:.3f}% "
max_pass = dataloader.get_pass_count(BATCH_SIZE)
session_configs["max step"] = f"{max_pass} ({LOADER_STRIDE} strides)"

session1 = Session(transformer, tokenizer1, True, session_configs)

a = random.randint(1,9999999999999)
a = str(a)
start = time.perf_counter()
session1.train(dataloader, display_message=True)
end = time.perf_counter()
print(f"training finished. time: {end - start:.3f}s")