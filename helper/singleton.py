from pathlib import Path

def init_corpus(pathfile:str):
    corpus = ""
    files = []
    folder = Path("data")
    for file in folder.iterdir():
        if file.name != ".gitkeep" and file.name[-1:-5:-1] == "txt." :
            files.append(file)

    for file in files:
        with open(file) as f:
            data = f.read()
            corpus += data + "\n\n\n <|endofdoc|>"
    return corpus, files