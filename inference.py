from engine.sessions import Session
import engine.backend as nx
import time


session = Session.load("artifacts/sessions/session_11886080_param_1_epochs_weights_only_quantized_827cb35e-b746-470e-8753-b7b39abc81b7.ram2n")
print(session.transformer.quantized)
tokenizer = session.tokenizer
context_size = session.configs["context_size"]
context = "you're jealous"
print(f"input: {context}")
context = nx.array(tokenizer.encode(context), nx.uint16)

context = context.reshape(-1, context.shape[0])

TEMPERATURE = .5
TOP_K = 30
TOP_P = .9
N = 100
penalty_mem = 128
penalty = 0.7
print(f"n: {N} | temp: {TEMPERATURE} | top_k: {TOP_K} | top_p: {TOP_P} | penalty_mem: {penalty_mem}, | penalty: {penalty}")
session.inference(context, TEMPERATURE, TOP_K, TOP_P, N, penalty_mem, penalty)
# print(session)
