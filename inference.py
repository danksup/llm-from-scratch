import os
backend = os.environ["BACKEND"] = "numpy"
import engine.backend as nx
from engine.sessions import Session
from engine.tokenizer import Tokenizer
import random

nx.set_seed(random.randrange(0,99999))

tokenizer = Tokenizer.load("artifacts/tokenizer/tokenizer24576_1687527270len.tokenizer")
session_path = "artifacts/sessions/session_133854720_param_1_epochs_weights_only_3e87a240-fef7-4b64-83ce-f2a23d54551e.safetensors"

session = Session.load(session_path, tokenizer)
context = "I."
print(f"input: {context}")
context = nx.array(tokenizer.encode(context), nx.uint32)

context = context.reshape(-1, context.shape[0])

TEMPERATURE = .4
TOP_K = 30
TOP_P = .8
N = 100
penalty_mem = 128
penalty = 1.2
print(f"n: {N} | temp: {TEMPERATURE} | top_k: {TOP_K} | top_p: {TOP_P} | penalty_mem: {penalty_mem}, | penalty: {penalty}")
session.inference(context, TEMPERATURE, TOP_K, TOP_P, N, penalty_mem, penalty)
