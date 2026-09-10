import os
backend = os.environ["BACKEND"] = "auto"
import engine.backend as nx
from engine.sessions import Session
from engine.tokenizer import Tokenizer
import random

nx.set_seed(random.randrange(0,99999))

tokenizer = Tokenizer.load("artifacts/tokenizer/tokenizer32768_1624612680len.tokenizer")
session_path = "artifacts/sessions/session_111894080_param_1_epochs_weights_only_29d87e76-fa44-4045-a57a-88884b6666fe.safetensors"

session = Session.load(session_path, tokenizer)
context = "i jsut wanna keep calling your name"
print(f"input: {context}")
context = nx.array(tokenizer.encode(context), nx.uint32)

context = context.reshape(-1, context.shape[0])

TEMPERATURE = -.10
TOP_K = 30
TOP_P = .8
N = 100
penalty_mem = 128
penalty = 0.25
print(f"n: {N} | temp: {TEMPERATURE} | top_k: {TOP_K} | top_p: {TOP_P} | penalty_mem: {penalty_mem}, | penalty: {penalty}")
session.inference(context, TEMPERATURE, TOP_K, TOP_P, N, penalty_mem, penalty)
