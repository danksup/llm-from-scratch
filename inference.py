#any path here is just leftover from me testing.

import os
backend = os.environ["BACKEND"] = "auto"
import engine.backend as nx
from engine.sessions import Session
from engine.tokenizer import Tokenizer
from engine.inference import Inference
import random

nx.set_seed(random.randrange(0,99999))

tokenizer = Tokenizer.load("artifacts/tokenizer/tokenizer32768_1624612680len.tokenizer")
session = Session.load("artifacts/sessions/session_(165985992, 81014400)_param_1_epochs_weight_only_quantized_9ae41604-0102-4307-839e-efdc41b26d2f.safetensors", tokenizer)

configs = {
    "seed":12345,
    "n":256,
    "top_k": 50,
    "temp":.8,
    "top_p":0.9,
    "penalty_mem_size":128,
    "penalty":.75,
    "attn_rope_scale":2
}

inference = Inference(session, **configs)
context = "i hate you."
print(configs)
print(f"input: {context}")

context = nx.array(tokenizer.encode(context), nx.uint32)
context = context.reshape(-1, context.shape[0])

inference.infer(context)
# print(inference.mtp_acceptance)
