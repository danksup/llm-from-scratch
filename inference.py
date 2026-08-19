from engine.sessions import Session
import engine.backend as nx
import time


session = Session.load("artifacts/sessions/session_136033280_param_1_epochs_weights_only_21d334f4-dea9-4d46-a2bd-3aed81595686.ram2n")
print(session.transformer.quantized)
tokenizer = session.tokenizer
context_size = session.configs["context_size"]
context = "hello."
print(f"input: {context}")
context = nx.array(tokenizer.encode(context), nx.uint16)

context = context.reshape(-1, context.shape[0])

TEMPERATURE = .4
TOP_K = 30
TOP_P = .9
N = 50
penalty_mem = 128
penalty = 0.7
print(f"n: {N} | temp: {TEMPERATURE} | top_k: {TOP_K} | top_p: {TOP_P} | penalty_mem: {penalty_mem}, | penalty: {penalty}")
session.inference(context, TEMPERATURE, TOP_K, TOP_P, N, penalty_mem, penalty)
# print(session)
