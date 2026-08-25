import array
import random
from multiprocessing import Process, Queue
from pathlib import Path
from typing import Any, Iterator, Literal

import engine.backend as nx
from engine.tokenizer import Tokenizer


class DataLoader:
    def __init__(self, filepath:str, tokenizer:Tokenizer, context_size:int=1024, batch_size:int=10, train_split:float|Literal['all']=0.9, *, __test_mode:bool = False) -> None:
        '''
        Args:
            filepath: filepath
            tokenizer: tokenizer object
            context_size: how much context is taken into computation at a time
            train_split: split contexts between training and validation
        '''
        self.train_split = train_split
        self.context_size = context_size
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.filepath = filepath

        assert isinstance(train_split, (float,int)) or train_split == "all", f"provide either float or \"all\" for train_split argument. got {train_split} of type {type(train_split)} instead"

        if isinstance(train_split, (float,int)):
            assert 0 < train_split <= 1.0, f"provide a value within (0,1] for train split. got {train_split} instead"

        if train_split == "all":
            train_split = 1.0

        train_files, validation_files = self.split_files(filepath, train_split)

        assert len(train_files) > 0, f"train_split ({train_split}) is too small. 0 files were allocated for training."

        self.train_files = train_files
        self.validation_files = validation_files

        if len(tokenizer.vocab) <=  65_535:
            self.type_code = 'H'
        else:
            self.type_code = 'I'

        self.__test_mode = __test_mode
        if __test_mode:
            self.__cow_factor = 0.1

    @staticmethod
    def get_files(filepath:str="data"):
        path = Path(filepath)
        files = []

        for file in path.iterdir():
            if file.is_file() and file.suffix in [".txt", ".tokenized"]:
                files.append(file)
        return files

    def all_tokenized(self, files:None|list[Path]=None):
        if files is None:
            files = self.train_files + self.validation_files
        return all(file.suffix == ".tokenized" for file in files)

    def get_tokenized_size(self, file:Path):
        if file.suffix != ".tokenized":
            raise ValueError("only for tokenized files (with .tokenized)")

        token_size = 2 if self.type_code == "H" else 4
        header_size = 29
        return  (file.stat().st_size - header_size) // token_size

    def get_token_sizes(self, files:list[Path]|None= None, strict:bool=False):
        total_tokens = 0

        if files is None:
            files = self.train_files + self.validation_files

        if strict:
            if not self.all_tokenized(files):
                raise ValueError("only tokenized files are allowed.")

        for file in files:
            if file.suffix == ".tokenized":
                total_tokens += self.get_tokenized_size(file)
        return total_tokens

    @staticmethod
    def split_files(filepath:str, split_value:float=.9):
        files = DataLoader.get_files(filepath)

        assert len(files) > 0, "no files found in directory."

        if split_value < 1.0:
            assert len(files) >= 2, "at least 2 files are needed if using validation."

        file_sizes = nx.array([i.stat().st_size for i in files], nx.uint64)
        target_train = (nx.sum(file_sizes) * split_value).item()
        sorted_sizes = nx.argsort(file_sizes).tolist()[::-1]

        train_files = []
        validate_files = []

        cum = 0

        for i in sorted_sizes:
            curr_size = file_sizes[i]
            take = cum + curr_size
            distance_take = abs(target_train - take)
            distance_no = abs(target_train - cum)

            if distance_take < distance_no:
                if i == sorted_sizes[-1] and split_value < 1.0:
                    validate_files.append(files[i])
                    break
                train_files.append(files[i])
                cum = take
            else:
                validate_files.append(files[i])

        return train_files, validate_files

    @staticmethod
    def stream_file(files:list[Path], permutation:list[int]) -> Iterator[Path]:
        for idx in permutation:
            yield files[idx]

    @staticmethod
    def stream_chunk(file:Path,type_code:str, chunk_size= 100_240_000, tokenizer_id_bytes:Any|None=None):
        chunk = None
        if file.suffix == ".txt":
            with open(file, "r", encoding="utf-8", errors='ignore') as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break

                    if chunk and not chunk[-1].isspace():
                        while True:
                            a = f.read(1)
                            if a:
                                chunk += a
                                if a.isspace():
                                    break
                            else:
                                break
                    yield chunk

        elif file.suffix == ".tokenized":
            with open(file, "rb") as f:
                magic = f.read(9)
                if magic != b"tokenized":
                    raise ValueError("unknown file")
                version = int.from_bytes(f.read(4), "little")
                this_tokenizer_id = f.read(16)
                if this_tokenizer_id != tokenizer_id_bytes:
                    raise ValueError("please use the same tokenizer you used for tokenizing.")
                while True:
                    tokens = array.array(type_code)
                    try:
                        tokens.fromfile(f, chunk_size)
                        if not tokens:
                            break
                        yield tokens
                    except EOFError:
                        if tokens :
                            yield tokens
                        break

    def stream_token(self, files:list[Path],permutation:list[int], carry_leftover_to_next_file:bool=True, chunk_size= 1002400):
        leftover_temp_context = None
        needed_T = self.context_size + 1

        for file in self.stream_file(files, permutation):
            if not carry_leftover_to_next_file:
                leftover_temp_context = None
            for chungus in self.stream_chunk(file,self.type_code, chunk_size, self.tokenizer.tokenizer_id.bytes): #type:ignore
                context = None

                if leftover_temp_context is not None:
                    context = leftover_temp_context
                    leftover_temp_context = None

                if isinstance(chungus, str):
                    chungus = self.tokenizer.encode(chungus)

                if context is not None and len(chungus) < needed_T:
                    context.extend(chungus)
                    concat_len = len(context)
                    if concat_len == needed_T:
                        yield context
                        continue
                    elif concat_len < needed_T:
                        leftover_temp_context = context
                        continue
                    else:
                        chungus = context
                        context = None

                while len(chungus) >= needed_T:
                    need = needed_T - len(context) if context is not None else needed_T
                    if context is not None:
                        context.extend( chungus[0:need])
                    else:
                        context = chungus[0:need]
                    # if self.__test_mode:
                    #     yield self.__function_that_turns_n_tokens_in_random_sequence_into_the_token_for_the_word_cow_randomly(context)
                    # else:
                    #     yield context
                    yield context
                    context = None
                    chungus = chungus[need:]

                leftover_temp_context = chungus

    def get_pairs(self, files:list[Path],  chunk_size:int= 1024000):
        context_batches = []
        target_batches = []

        permutation = [i for i in range(len(files))]
        random.shuffle(permutation)
        for token in self.stream_token(files, permutation, chunk_size=chunk_size):
            if token is None:
                continue
            # print(type(token))
            context_batches.append(token[:-1])
            target_batches.append(token[1:])

            if len(context_batches) == self.batch_size:
                yield context_batches,target_batches
                context_batches = []
                target_batches = []

    def worker(self, Q:Queue, files:list[Path], chunk_size:int= 1024000):
        for batch in self.get_pairs(files, chunk_size):
            # Q.put(nx.tolist(batch))
            Q.put(batch)
        Q.put(None)

    def prefetch_batch(self, files:list[Path], max_queue_size:int=20, chunk_size:int= 1024000):
        queue = Queue(max_queue_size)
        process  = Process(target=self.worker, args=(queue, files, chunk_size))

        try:
            process.start()
            while True:
                item = queue.get()
                if item is None:
                    break
                yield item
        finally:
            if process.is_alive():
                process.terminate()
        process.join()

    def tokenize(self, file:Path):
        filename = file.stem
        with open(Path(f"artifacts/dataloader/{filename}.tokenized"), "wb") as f:
            f.write(b"tokenized")
            f.write((1).to_bytes(4, "little"))
            f.write(self.tokenizer.tokenizer_id.bytes) #type:ignore
            for x in self.stream_token([file], [0]):
                tokens = array.array(self.type_code, x)
                tokens.tofile(f)

    def pretokenize(self, files:list[Path]|None=None, one_file:bool=True):
        if files is None:
            files = self.train_files + self.validation_files
        indices = [i for i in range(len(files))]

        if len(files) > 1 and one_file:
            filename = f"{len(files)}_files"

            with open(Path(f"artifacts/dataloader/{filename}.tokenized"), "wb") as f:
                f.write(b"tokenized")
                f.write((1).to_bytes(4, "little"))
                f.write(self.tokenizer.tokenizer_id.bytes) #type:ignore

                for token in self.stream_token(files, indices):
                    tokens = array.array(self.type_code, token)
                    tokens.tofile(f)
        else:
            for file in self.stream_file(files, indices):
                self.tokenize(file)

    def estimate_step(self, total_tokens,  microbatch_size:int=1):
        return total_tokens // self.context_size // self.batch_size // microbatch_size

    def __function_that_turns_n_tokens_in_random_sequence_into_the_token_for_the_word_cow_randomly(self, token):
        self.__cow_factor = min(getattr(self, "__cow_factor", 0.1), 0.5)
        if random.random() < self.__cow_factor:
            cow  = self.tokenizer.encode("cow")
            len_token = len(token)
            if len(token) == len(cow):
                return cow
            else:
                n_cow_base =  max(len_token//8, 1)
                considering_cow_factor = n_cow_base + int(n_cow_base * self.__cow_factor)
                how_much = random.randint(0, considering_cow_factor)
                cow_length = len(cow)

                if how_much == 0:
                    self.__cow_factor += 0.123456789

                for _ in range(how_much):
                    random_place = random.randint(0, len_token - 1 - cow_length)

                    if token[random_place:random_place+cow_length] == cow and (token[random_place - cow_length:random_place] != cow if random_place > (cow_length-1) else True):
                        self.__cow_factor += 0.05
                    else:
                        for cow_piece in range(cow_length):
                            token[random_place+cow_piece] = cow[cow_piece]
        else:
            self.__cow_factor += 0.01
        return token
