from engine.sessions import Session
import engine.backend as nx
import time


session = Session.load("artifacts/sessions/session_82688000_param_1_epochs_weights_only_quantized_bca51d20-94e1-40e8-aa8c-fc59eb25a9b4.ram2n")
print(session.transformer.get_all_weights())
# tokenizer = session.tokenizer
# context_size = session.configs["context_size"]
# context = "you are stupid"
# print(f"input: {context}")
# context = nx.array(tokenizer.encode(context), nx.uint16)

# context = context.reshape(-1, context.shape[0])

# TEMPERATURE = .4
# TOP_K = 30
# TOP_P = .8
# N = 100
# penalty_mem = 128
# penalty = 1.2
# print(f"n: {N} | temp: {TEMPERATURE} | top_k: {TOP_K} | top_p: {TOP_P} | penalty_mem: {penalty_mem}, | penalty: {penalty}")
# session.inference(context, TEMPERATURE, TOP_K, TOP_P, N, penalty_mem, penalty)
# # print(session)
