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

TOKENIZER_PATH = "artifacts/tokenizer/tokenizer32768_1624612680len.tokenizer"
tokenizer1 = Tokenizer.load(TOKENIZER_PATH)

session_configs = {
    "epochs":5,
    "max_step":21,
    "train_split": 1,
    "max_val_step":1,
    "eval_every":1, 
    "validate_every":0,
    "context_size": 32,
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
    "save":True,
    "create_checkpoint":True,
    "checkpoint_every":1000,
    "weights_only": True
}

model_configs = {
    "n_blocks":5,
    "embed_dim":128,
    "dtype": "float16",
    "gradient_scale":32768,
    "vocab_size": len(tokenizer1.vocab),
    "quantized":False, #here can be True, "symmetric", False
    "check_non_finite":True,
    "recompute_activation":True,
    "block_configs":{
        "ff_hidden_width": 512,
        "ff_n_experts":1,
        "ff_topk":1,
        "ff_cf":1.25,
        "ff_init":"glorot_uniform",
        "ff_shared_hidden_width":512, 
        "ff_n_shared_experts":2,
        "ff_moe_lambda":1e-3,
        "ff_jitter_eps":0.02,
        "attn_type":"chunked",
        "attn_chunk_size":16,
        "attn_variant":"gqa",
        "attn_n_heads":2,
        "attn_n_kv_heads":1,
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