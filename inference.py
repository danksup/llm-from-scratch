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
session = Session.load("artifacts/sessions/session_(89025024, 33974784)_param_1_epochs_weights_only_6c8e9f34-a988-4bc5-b83d-9df98cd4bd25.safetensors", tokenizer)

inference = Inference(session)
context = "Do I know you?"
print(f"input: {context}")
context = nx.array(tokenizer.encode(context), nx.uint32)

context = context.reshape(-1, context.shape[0])

TEMPERATURE = 0.7
TOP_K = 15
TOP_P = .9
N = 100
penalty_mem = 128
penalty = 0.25
print(f"n: {N} | temp: {TEMPERATURE} | top_k: {TOP_K} | top_p: {TOP_P} | penalty_mem: {penalty_mem}, | penalty: {penalty}")
inference.infer(context, TEMPERATURE, TOP_K, TOP_P, N, penalty_mem, penalty)
