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
session = Session.load("artifacts/sessions/session_(89025024, 33974784)_param_1_epochs_weights_only_649a52b8-2b43-44dc-bc68-6c5eb8c21b1a.safetensors", tokenizer)

configs = {
    "seed":12345,
    "n":8192,
    "top_k": 50,
    "temp":1.0,
    "top_p":0.9,
    "penalty_mem_size":32,
    "penalty":.25
}

inference = Inference(session, **configs)
context = "You"
print(configs)
print(f"input: {context}")

context = nx.array(tokenizer.encode(context), nx.uint32)
context = context.reshape(-1, context.shape[0])

inference.infer(context)
