from engine.sessions import Session
from engine.tokenizer import Tokenizer
import engine.backend as nx
import time

tokenizer = Tokenizer.load("artifacts/tokenizer/tokenizer32000_1351277738len.tokenizer")
session = Session.simple_load("artifacts/sessions/session_88832000_param_1_epochs_weights_only_19f47f5e-5e2f-4b8e-a2f1-3a362e0e279e.safetensors", tokenizer)
context = "test123."
print(f"input: {context}")
context = nx.array(tokenizer.encode(context), nx.uint16)

context = context.reshape(-1, context.shape[0])

TEMPERATURE = .4
TOP_K = 30
TOP_P = .8
N = 100
penalty_mem = 128
penalty = 1.2
print(f"n: {N} | temp: {TEMPERATURE} | top_k: {TOP_K} | top_p: {TOP_P} | penalty_mem: {penalty_mem}, | penalty: {penalty}")
session.inference(context, TEMPERATURE, TOP_K, TOP_P, N, penalty_mem, penalty)
