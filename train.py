import os
backend = os.environ["BACKEND"] = "auto"
import random
EPOCHS = 1
EMBED_DIM = 64
CONTEXT_SIZE = 64
BATCH_SIZE = 128
BASE_WIDTH = 1024#4 * EMBED_DIM 
N_HEADS = EMBED_DIM // 16
N_KV_HEADS = N_HEADS // 4
WINDOWS = CONTEXT_SIZE // 4
N_EXPERTS = 24
CF = 1.25
VAL = .9
TOP_K = 2

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
import engine.backend as nx

from helper.singleton import init_corpus

session_configs = {
    "epochs":1,
    "context_size": CONTEXT_SIZE,
    "batch_size": BATCH_SIZE,
    "optimizer":"adamw",
    "train_split":.9,
    "train_split":VAL,
    "optimizer_args":{
        "min_lr":1e-6,
        "max_lr":1e-3,
        "beta":0.9,
        "beta2":0.999,
        "epsilon":1e-8,
        "weight_decay":0.01
    },
    "using":backend
}

model_configs = {
    "n_blocks":4,
    "embed_dim":EMBED_DIM,
    "dtype": nx.float16,
    "block_configs":{"ff_hidden_width": BASE_WIDTH,"ff_n_experts":N_EXPERTS,"ff_topk":TOP_K,"ff_cf":CF,"attn_n_heads":N_HEADS,"attn_n_kv_heads":N_KV_HEADS,"attn_windows":CONTEXT_SIZE//2},
    "block_overrides":{
        0:{}, 2:{}
    }
}

corpus, files = init_corpus("data")

tokenizer1 = Tokenizer.load(TOKENIZER_PATH)
# print(tokenizer1.vocab)
session_configs["dataset"] = f"{len(files)} files"
real_vocab_size = len(tokenizer1.vocab)

transformer = Transformer(model_configs)

session_configs["block_size"] = len(transformer.blocks)

print("loading dataloader", end="\r")
dataloader = DataLoader(corpus, tokenizer1, session_configs["context_size"])

corpus_len = len(corpus)
token_size = dataloader.tokens.size
ratio = ((corpus_len - token_size ) / corpus_len) * 100
session_configs["corpus char len"] = f"{corpus_len} -> BPE compression ({len(tokenizer1.vocab)} vocab size): {token_size}. ratio = {ratio:.3f}% "

session1 = Session(transformer, tokenizer1, True, session_configs)

a = random.randint(1,9999999999999)
a = str(a)
start = time.perf_counter()
session1.train(dataloader, display_message=True)
end = time.perf_counter()
print(f"training finished. time: {end - start:.3f}s")

session1.save(f"{session1.count_params()}_params_{EPOCHS}_epochs")
