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
CONTEXT_SIZE = 2048
BATCH_SIZE = 4
N_HEADS = 8
N_KV_HEADS = max(1, N_HEADS // 4)
N_EXPERTS = 8
CF = 1.25
VAL = .9
TOP_K = 2
MAX_STEP = 31160

CORPUS_PATH = "artifacts/dataloader/2025_26"
TOKENIZER_PATH = "artifacts/tokenizer/tokenizer32768_1624612680len.tokenizer"
tokenizer1 = Tokenizer.load(TOKENIZER_PATH)

session_configs = {
    "epochs":EPOCHS,
    "max_step":MAX_STEP,
    "train_split": VAL,
    "max_val_step":20,
    "eval_every":1,
    "validate_every":10,
    "context_size": CONTEXT_SIZE,
    "batch_size": BATCH_SIZE,
    "microbatch_size":16,
    "optimizer":"adamw",
    "optimizer_args":{
        "lr": 1e-3,
        "use_master": True,
        "scheduler": "cosine_decay",
        "min_lr": 8e-5,
        "t_warmup": max(1, int(MAX_STEP * 0.02)),
        "eight_bit":True,
        "foreach":False
    },
    "save":True,
    "logger":True,
    "error_save":True,
    "create_checkpoint":True,
    "checkpoint_every":100,
    "weight_only": True,
    "backend": {
        "mlx_disable_compile":False,
        "mlx_save_quantized_weights_as_symmetric":True
    }
}

model_configs = {
    "n_blocks":10,
    "n_mtp":1,
    "discard_mtp":False,
    "mtp_lambda":0.5,
    "embed_dim":EMBED_DIM,
    "dtype": "float16",
    "gradient_scale":8192,
    "vocab_size": len(tokenizer1.vocab),
    "quantized":False,
    "doc_id":True,
    "check_non_finite":True,
    "recompute_activation":True,
    "block_configs":{
        "ff_hidden_width": int(2 * EMBED_DIM),
        "ff_n_experts":N_EXPERTS,
        "ff_topk":TOP_K,
        "ff_cf":CF,
        "ff_init":"glorot_uniform",
        "ff_dense_hidden_width":int(4 * EMBED_DIM),
        "ff_n_dense":1,
        "ff_jitter_eps":0.02,
        "ff_bias_gamma": 1e-3,
        "attn_type":"chunked",
        "attn_chunk_size":CONTEXT_SIZE//16,
        "attn_variant":"gqa",
        "attn_n_heads":N_HEADS,
        "attn_n_kv_heads":N_KV_HEADS,
        "attn_init":"glorot_uniform",
        "attn_rope_theta":10000,
        "attn_rope_scale": 1.0,
        "attn_rope_orig_ctx": CONTEXT_SIZE,
        "attn_rope_alpha": 1.0,
        "attn_rope_beta": 32.0
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
