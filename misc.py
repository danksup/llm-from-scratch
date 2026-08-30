from misc.misc_embedding import n_closest, embedding_of
from engine.sessions import Session
from engine.tokenizer import Tokenizer
PATH = "artifacts/sessions/session_82688000_param_1_epochs_weights_only_quantized_59d0ce46-354e-429e-8774-579e6e9b7be9.safetensors"

tokenizer = Tokenizer.load("artifacts/tokenizer/tokenizer32000_1351277738len.tokenizer")
session = Session.load(PATH, tokenizer)
embedding = session.transformer.embedding

closest_to = "hello"
print(f"closest to {closest_to}")
n_closest(closest_to, tokenizer, embedding)
