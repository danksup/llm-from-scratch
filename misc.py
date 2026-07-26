from misc.misc_embedding import n_closest, embedding_of
from engine.sessions import Session
PATH = "artifacts/sessions/session_38974464_params_20_epochs.ram2n"

session = Session.load(PATH)
tokenizer = session.tokenizer
embedding = session.transformer.embedding

# all_4_or_more = []
# for i in tokenizer.vocab:
#     if len(i.replace("</w>","")) > 3:
#         all_4_or_more.append(i)
# print(all_4_or_more)

closest_to = "mother"
print(f"closest to {closest_to}")
n_closest(closest_to, tokenizer, embedding)
