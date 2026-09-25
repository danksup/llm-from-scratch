from misc.misc_embedding import n_closest, embedding_of
from engine.sessions import Session
from engine.tokenizer import Tokenizer
PATH = "artifacts/sessions/session_73295104_param_1_epochs_weights_only_1dcb2ebe-ba0b-406d-a460-60877860408d.safetensors"

tokenizer = Tokenizer.load("artifacts/tokenizer/tokenizer32768_1624612680len.tokenizer")
session = Session.load(PATH, tokenizer)
embedding = session.transformer.embedding

closest_to = "improving"
print(f"closest to {closest_to}")
n_closest(closest_to, tokenizer, embedding)
