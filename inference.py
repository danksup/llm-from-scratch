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
session = Session.load("artifacts/sessions/session_checkpoint_latest_e6f5ac34-78f0-4e39-b11c-dfd7582c054e.safetensors", tokenizer)

configs = {
    "seed":12345,
    "n":1024,
    "top_k": 50,
    "temp":.7,
    "top_p":0.9,
    "penalty_mem_size":128,
    "penalty":.75
}

inference = Inference(session, **configs)
context = "Chemical"
print(configs)
print(f"input: {context}")

context = nx.array(tokenizer.encode(context), nx.uint32)
context = context.reshape(-1, context.shape[0])

inference.infer(context)
