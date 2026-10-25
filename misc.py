from misc.misc_embedding import n_closest, embedding_of
from engine.sessions import Session
from engine.tokenizer import Tokenizer
PATH = "artifacts/sessions/session_(89025024, 33974784)_param_1_epochs_weights_only_26375069-d694-4cf7-b639-8ccec87b15c8.safetensors"

tokenizer = Tokenizer.load("artifacts/tokenizer/tokenizer32768_1624612680len.tokenizer")
session = Session.load(PATH, tokenizer)
embedding = session.transformer.embedding

closest_to = "me"
print(f"closest to {closest_to}")
n_closest(closest_to, tokenizer, embedding)
