from engine.sessions import Session
import time

session = Session.load("artifacts/sessions/session_51509760_param_1_epochs_weights_only_a.ram2n")
tokenizer = session.tokenizer
context_size = session.configs["context_size"]
context = "<PAD>"
print(f"input: {context}")
context = tokenizer.encode(context)
context = context.reshape(-1, context.shape[0])

TEMPERATURE = 0.7
TOP_K = 50
TOP_P = .9
N = 120
penalty_mem = 128
penalty = 2
print(f"n: {N} | temp: {TEMPERATURE} | top_k: {TOP_K} | top_p: {TOP_P} | penalty_mem: {penalty_mem}, | penalty: {penalty}")
session.inference(context, TEMPERATURE, TOP_K, TOP_P, N, penalty_mem, penalty)