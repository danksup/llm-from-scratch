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
session = Session.load("artifacts/sessions/session_(89025024, 33974784)_param_1_epochs_weights_only_ea64308e-c82a-49f1-bab2-d8c221e08eab.safetensors", tokenizer)

configs = {
    "seed":12345,
    "n":200,
    "top_k": 50,
    "temp":0.5,
    "top_p":0.9,
    "penalty_mem_size":32,
    "penalty":.25
}

inference = Inference(session, **configs)
context = "Do I know you?"
print(configs)
print(f"input: {context}")

context = nx.array(tokenizer.encode(context), nx.uint32)
context = context.reshape(-1, context.shape[0])

inference.infer(context)
