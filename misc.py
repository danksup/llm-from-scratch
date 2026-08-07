from misc.misc_embedding import n_closest, embedding_of
from engine.sessions import Session
PATH = "artifacts/sessions/session_25559808_param_1_epochs_weights_only.ram2n"

session = Session.load(PATH)
tokenizer = session.tokenizer
embedding = session.transformer.embedding

closest_to = "fuck"
print(f"closest to {closest_to}")
n_closest(closest_to, tokenizer, embedding)
