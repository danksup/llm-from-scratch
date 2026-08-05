from pathlib import Path
import time
from engine.tokenizer import Tokenizer
import cProfile
import pstats
VOCAB_SIZE = [32768, 65536, 131072]

corpus = ""
files = []
folder = Path("data")
for file in folder.iterdir():
    if file.name != ".gitkeep" and file.name[-1:-5:-1] == "txt." :
        files.append(file)

for file in files:
    with open(file) as f:
        data = f.read()
        corpus += data + "\n\n\n"

corpus_len = len(corpus)

for i in VOCAB_SIZE:
    print(f"fitting corpus of length {corpus_len} with vocab size of {i}")
    tokenizer1 = Tokenizer(i)
    t = int(time.time())
    print()
    start = time.perf_counter()
    # profiler = cProfile.Profile()
    # profiler.enable()
    a =tokenizer1.fit(corpus)
    # profiler.disable()
    # stats = pstats.Stats(profiler)
    # stats.sort_stats("cumtime")
    # stats.print_stats(100)
    end = time.perf_counter()
    tokenizer_save_name = f"{i}_{len(corpus)}len"
    tokenizer1.save(tokenizer_save_name)
    print(f"{tokenizer_save_name} saved. fitting finished in {end-start:.3f}")
