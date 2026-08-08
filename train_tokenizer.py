from pathlib import Path
import time
from engine.tokenizer import Tokenizer
import multiprocessing

def train_tokenizer(vocab_size, filepath:str = "data"):
    print(f"fitting size of {vocab_size}")
    tokenizer1 = Tokenizer(vocab_size)

    start = time.perf_counter()
    tokenizer1.fit(filepath)
    end = time.perf_counter()

    tokenizer_save_name = f"{vocab_size}_{tokenizer1.total_char_raw}len"
    tokenizer1.save(tokenizer_save_name)
    print(f"{tokenizer_save_name} saved. fitting finished in {end-start:.3f}")

if __name__ == "__main__":
    train_tokenizer(48000)

    # VOCAB_SIZE = [24000, 48000]
    # processes = [
    #     multiprocessing.Process(
    #         target=train_tokenizer,
    #         args=(vocab_size,)
    #     )
    #     for vocab_size in VOCAB_SIZE
    # ]
    # for process in processes:
    #     process.start()

    # for process in processes:
    #     process.join()