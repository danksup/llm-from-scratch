from engine.sessions import Session
import time

session = Session.load("artifacts/sessions/session_31827456_param_1_epochs_weights_only.ram2n")

tokenizer = session.tokenizer
context_size = session.configs["context_size"]
context = "hello"
print(f"input: {context}")
context = tokenizer.encode(context)
context = context.reshape(-1, context.shape[0])

TEMPERATURE = 0.9
TOP_K = 30
TOP_P = .9
N = 50
print(f"n: {N} | temp: {TEMPERATURE} | top_k: {TOP_K} | top_p: {TOP_P}")
session.inference(context, TEMPERATURE, TOP_K, TOP_P, N, 16)