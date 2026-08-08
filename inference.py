from engine.sessions import Session
import time

session = Session.load("artifacts/sessions/session_36435456_param_2_epochs_weights_only.ram2n")
tokenizer = session.tokenizer
context_size = session.configs["context_size"]
context = "are you stupid?"
print(f"input: {context}")
context = tokenizer.encode(context)
context = context.reshape(-1, context.shape[0])

TEMPERATURE = 0.7
TOP_K = 15
TOP_P = .7
N = 50
penalty_mem = 32
penalty = 1.25
print(f"n: {N} | temp: {TEMPERATURE} | top_k: {TOP_K} | top_p: {TOP_P} | penalty_mem: {penalty_mem}, | penalty: {penalty}")
session.inference(context, TEMPERATURE, TOP_K, TOP_P, N, penalty_mem, penalty)