from misc.misc_embedding import n_closest, embedding_of
from engine.sessions import Session
from engine.tokenizer import Tokenizer
PATH = "artifacts/sessions/session_136476160_param_1_epochs_weights_only_98175dbc-3833-462e-aba0-083c0e85286f.safetensors"

tokenizer = Tokenizer.load("artifacts/tokenizer/tokenizer32768_1624612680len.tokenizer")
session = Session.load(PATH, tokenizer)
embedding = session.transformer.embedding

closest_to = "hello"
print(f"closest to {closest_to}")
n_closest(closest_to, tokenizer, embedding)
