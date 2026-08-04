from helper.singleton import init_corpus
from engine.sessions import Session
from engine.dataloader import DataLoader
PATH_TO_SESSION = "artifacts/sessions/session_23379456_param_1_epochs.ram2n"

corpus, _ = init_corpus("data")
session = Session.load(PATH_TO_SESSION)
dataloader = DataLoader(corpus, session.tokenizer)
session.train(dataloader)