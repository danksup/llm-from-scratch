#any path here is just leftover from me testing.

import os
backend = os.environ["BACKEND"] = "auto"
import engine.backend as nx
from engine.sessions import Session
from engine.tokenizer import Tokenizer
import random

nx.set_seed(random.randrange(0,99999))

tokenizer = Tokenizer.load("artifacts/tokenizer/tokenizer32768_1624612680len.tokenizer")
session_path = "artifacts/sessions/session_(73296384, 18246144)_param_1_epochs_weights_only_690352f8-9c6a-42aa-8000-16896afe2c78.safetensors"

session = Session.load(session_path, tokenizer)
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
session.inference(context, TEMPERATURE, TOP_K, TOP_P, N, penalty_mem, penalty)
