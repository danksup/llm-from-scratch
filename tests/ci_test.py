import os
from pathlib import Path
import time

backend = os.environ["BACKEND"] = "auto"
seed = os.environ["SEED"] = "1"

from engine.transformer import Transformer
from engine.tokenizer import Tokenizer
from engine.dataloader import DataLoader
from engine.sessions import Session
import engine.backend as nx
from helper.singleton import init_corpus

#not hooked yet to session
PATIENCE = 20
TRESHOLD = 1e-2

TOKENIZER_PATH = "artifacts/tokenizer/tokenizer24576_1624612680len.tokenizer"
tokenizer1 = Tokenizer.load(TOKENIZER_PATH)

session_configs = {
    "epochs":1,
    "max_step":50,
    "train_split": 1,
    "max_val_step":1,
    "eval_every":1, 
    "validate_every":0,
    "context_size": 4,
    "batch_size": 1,
    "microbatch_size":1,
    "optimizer":"adamw",
    "optimizer_args":{
        "lr": 1e-3,
        "use_master": True,
        "scheduler": "cosine_decay",
        "min_lr": 1e-5,
    },
    "using":os.environ.get("BACKEND"),
    "save":False,
    "create_checkpoint":True,
    "checkpoint_every":1000,
    "weights_only": True
}

model_configs = {
    "n_blocks":10,
    "embed_dim":256,
    "dtype": "float16",
    "gradient_scale":8,
    "vocab_size": len(tokenizer1.vocab),
    "moe_lambda":0.01,
    "block_configs":{
        "ff_hidden_width": 256,
        "ff_n_experts":10,
        "ff_topk":2,
        "ff_cf":1.25,
        "ff_init":"glorot_uniform",
        "attn_type":"full",
        "attn_variant":"gqa",
        "attn_n_heads":8,
        "attn_n_kv_heads":4,
        "attn_init":"glorot_uniform",
        },
    "block_overrides":{
      }
}

if __name__ == "__main__":
    transformer = Transformer(model_configs)

    session_configs["block_size"] = len(transformer.blocks)

    print("loading dataloader ", end="\r")
    dataloader = DataLoader("data/test", tokenizer1, session_configs["context_size"], session_configs["batch_size"], session_configs["train_split"])

    session1 = Session(transformer, tokenizer1, True, session_configs)
    start = time.perf_counter()
    session1.train(dataloader, display_message=True)
    end = time.perf_counter()
    print(f"training finished. time: {end - start:.3f}s")