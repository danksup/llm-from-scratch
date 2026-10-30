#btw if u found this repo this is like the manual cus im too lazy to make one.
#feel free to play with any values u see here, especially the filepath.

import os
os.environ["BACKEND"] = "auto"

import time

import engine.backend as nx
from engine.dataloader import DataLoader
from engine.sessions import Session
from engine.tokenizer import Tokenizer
from engine.transformer import Transformer

nx.set_seed(12345)

EPOCHS = 1
EMBED_DIM = 512
CONTEXT_SIZE = 4096
BATCH_SIZE = 1
BASE_WIDTH = 3 * EMBED_DIM
N_HEADS = 8
N_KV_HEADS = max(1, N_HEADS // 2)
N_EXPERTS = 8
CF = 1.25
VAL = 1
TOP_K = 2
MAX_STEP = 50000

CORPUS_PATH = "artifacts/dataloader/2025_26"
TOKENIZER_PATH = "artifacts/tokenizer/tokenizer32768_1624612680len.tokenizer"
tokenizer1 = Tokenizer.load(TOKENIZER_PATH)

session_configs = {
    "epochs":EPOCHS,
    "max_step":MAX_STEP,
    "train_split": VAL,
    "max_val_step":100,
    "eval_every":1,
    "validate_every":10,
    "context_size": CONTEXT_SIZE,
    "batch_size": BATCH_SIZE,
    "microbatch_size":16,
    "optimizer":"adamw",
    "optimizer_args":{
        "lr": 5e-4,
        "use_master": True,
        "scheduler": "cosine_decay",
        "min_lr": 5e-5,
        "t_warmup": int(MAX_STEP * 0.015)
    },
    "save":False,
    "logger":False,
    "error_save":True,
    "create_checkpoint":True,
    "checkpoint_every":100,
    "weights_only": True,
    "backend": {
        "mlx_disable_compile":False,
        "mlx_save_quantized_weights_as_symmetric":True
    }
}

model_configs = {
    "n_blocks":12,
    "embed_dim":EMBED_DIM,
    "dtype": "float16",
    "gradient_scale":32768,
    "vocab_size": len(tokenizer1.vocab),
    "quantized":False,
    "check_non_finite":True,
    "recompute_activation":True,
    "block_configs":{
        "ff_hidden_width": BASE_WIDTH,
        "ff_n_experts":N_EXPERTS,
        "ff_topk":TOP_K,
        "ff_cf":CF,
        "ff_init":"glorot_uniform",
        "ff_dense_hidden_width":BASE_WIDTH, 
        "ff_n_dense":2,
        "ff_moe_lambda":1e-3,
        "ff_jitter_eps":0.02,
        "attn_type":"chunked",
        "attn_chunk_size":CONTEXT_SIZE//8,
        "attn_variant":"gqa",
        "attn_n_heads":N_HEADS,
        "attn_n_kv_heads":N_KV_HEADS,
        "attn_init":"glorot_uniform",
        },
    "block_overrides":{
      }
}

if __name__ == "__main__":
    transformer = Transformer(model_configs)

    session_configs["block_size"] = len(transformer.blocks)

    dataloader = DataLoader(CORPUS_PATH, tokenizer1, session_configs["context_size"], session_configs["batch_size"], session_configs["train_split"])

    session1 = Session(transformer, tokenizer1, True, session_configs)
    start = time.perf_counter()

    session1.train(dataloader, display_message=True)
    end = time.perf_counter()
    print(f"training finished. time: {end - start:.3f}s")
