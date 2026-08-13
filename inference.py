from engine.sessions import Session
import time

session = Session.load("artifacts/sessions/session_keyboardinterrupt_save_e2472e77-70c2-46bf-8c59-6ec48d31af12.ram2n")
tokenizer = session.tokenizer
context_size = session.configs["context_size"]
context = "YOURE REPEATING URSELF"
print(f"input: {context}")
context = tokenizer.encode(context)
context = context.reshape(-1, context.shape[0])

TEMPERATURE = 0.6
TOP_K = 30
TOP_P = .8
N = 120
penalty_mem = 128
penalty = 1.2
print(f"n: {N} | temp: {TEMPERATURE} | top_k: {TOP_K} | top_p: {TOP_P} | penalty_mem: {penalty_mem}, | penalty: {penalty}")
session.inference(context, TEMPERATURE, TOP_K, TOP_P, N, penalty_mem, penalty)