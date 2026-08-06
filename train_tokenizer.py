from pathlib import Path
import time
from engine.tokenizer import Tokenizer
from helper.singleton import init_corpus
import cProfile
import pstats
VOCAB_SIZE = [15000, 24000, 48000]

corpus, file = init_corpus("data")

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
