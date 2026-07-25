from engine.sessions import Session

session = Session.load("artifacts/sessions/session_19446272_params_1_epochs.ram2n")
session.configs["epochs"] = 20
tokenizer = session.tokenizer
context_size = session.configs["context_size"]
context = "Great Britain established?"
print(f"input: {context}")
context = tokenizer.encode(context)
context = context.reshape(-1, context.shape[0])

TEMPERATURE = 0.6
TOP_K = 20
TOP_P = .8
N = 200
print(f"n: {N} | temp: {TEMPERATURE} | top_k: {TOP_K} | top_p: {TOP_P}")
session.inference(context, TEMPERATURE, TOP_K, TOP_P, N)