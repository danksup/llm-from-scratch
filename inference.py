from engine.sessions import Session
import engine.backend as nx 
import time


session = Session.load("artifacts/sessions/session_keyboardinterrupt_save_fc2c8683-f566-4bca-aa18-52272d708dbd.ram2n")
tokenizer = session.tokenizer
context_size = session.configs["context_size"]
context = "give me everything."
print(f"input: {context}")
context = nx.array(tokenizer.encode(context), nx.uint16)

context = context.reshape(-1, context.shape[0])

TEMPERATURE = 1
TOP_K = 30
TOP_P = .8
N = 120
penalty_mem = 128
penalty = 1.2
print(f"n: {N} | temp: {TEMPERATURE} | top_k: {TOP_K} | top_p: {TOP_P} | penalty_mem: {penalty_mem}, | penalty: {penalty}")
session.inference(context, TEMPERATURE, TOP_K, TOP_P, N, penalty_mem, penalty)