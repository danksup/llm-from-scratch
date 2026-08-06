from engine.sessions import Session
import time

session = Session.load("artifacts/sessions/session_70743040_param_5_epochs_weights_only.ram2n")

tokenizer = session.tokenizer
context_size = session.configs["context_size"]
context = "lovely"
print(f"input: {context}")
context = tokenizer.encode(context)
context = context.reshape(-1, context.shape[0])

TEMPERATURE = 0.7
TOP_K = 30
TOP_P = .9
N = 100
print(f"n: {N} | temp: {TEMPERATURE} | top_k: {TOP_K} | top_p: {TOP_P}")
session.inference(context, TEMPERATURE, TOP_K, TOP_P, N, 32)