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
EMBED_DIM = 320
CONTEXT_SIZE = 1200
BATCH_SIZE = 5
BASE_WIDTH = 4 * EMBED_DIM
N_HEADS = 8
N_KV_HEADS = max(1, N_HEADS // 2)
N_EXPERTS = 10
CF = 1.25
VAL = 1
TOP_K = 2

CORPUS_PATH = "artifacts/dataloader"
TOKENIZER_PATH = "artifacts/tokenizer/tokenizer32000_1351277738len.tokenizer"
tokenizer1 = Tokenizer.load(TOKENIZER_PATH)

session_configs = {
    "epochs":EPOCHS,
    "max_step":1000,
    "train_split": VAL,
    "max_val_step":1,
    "eval_every":1,
    "validate_every":0,
    "context_size": CONTEXT_SIZE,
    "batch_size": BATCH_SIZE,
    "microbatch_size":32,
    "optimizer":"adamw",
    "optimizer_args":{
        "lr": 1e-3,
        "use_master": True,
        "scheduler": "none",
        "min_lr": 1e-4,
    },
    "using":os.environ.get("BACKEND"),
    "save":True,
    "create_checkpoint":True,
    "checkpoint_every":1000,
    "weights_only": True,
    "backend": {
        "mlx_disable_compile":False,
        "mlx_save_quantized_weights_as_symmetric":True
    }
}

model_configs = {
    "n_blocks":10,
    "embed_dim":EMBED_DIM,
    "dtype": "float16",
    "gradient_scale":4096,
    "vocab_size": len(tokenizer1.vocab),
    "moe_lambda":0.01,
    "quantized":False, #here can be True, "symmetric", False
    "check_non_finite":False,
    "block_configs":{
        "ff_hidden_width": BASE_WIDTH,
        "ff_n_experts":N_EXPERTS,
        "ff_topk":TOP_K,
        "ff_cf":CF,
        "ff_init":"glorot_uniform",
        "attn_type":"full",
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

    print("loading dataloader ", end="\r")
    dataloader = DataLoader(CORPUS_PATH, tokenizer1, session_configs["context_size"], session_configs["batch_size"], session_configs["train_split"])

    session1 = Session(transformer, tokenizer1, True, session_configs)
    start = time.perf_counter()

    session1.train(dataloader, display_message=True)
    end = time.perf_counter()
    print(f"training finished. time: {end - start:.3f}s")
