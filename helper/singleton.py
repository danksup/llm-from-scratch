from pathlib import Path
from typing import Literal

colors = {
    "RED": '\033[91m',
    "GREEN": '\033[92m',
    "YELLOW": '\033[93m',
    "BLUE": '\033[94m',
    "MAGENTA": '\033[95m',
    "CYAN": '\033[96m'
}

styles = {
    None:"",
    "BOLD": '\033[1m',
    "UNDERLINE": '\033[4m'
}

reset = '\033[0m'

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

def colorize(text:str, color:Literal["red", "green", "yellow", "blue"], style:Literal[None, "bold", "underline"]=None):
    return (f"{colors[color.upper()]}{styles[style.upper()] if style is not None else styles[style]}{text}{reset}")