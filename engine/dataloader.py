from engine.tokenizer import Tokenizer
import engine.backend as nx
from typing import Any, Iterator,Literal
import math
from pathlib import Path


class DataLoader:
    def __init__(self, filepath:str, tokenizer:Tokenizer, context_size:int=16, train_split:float|Literal['all']=0.9) -> None:
        '''
        Args:
            filepath: filepath
            tokenizer: tokenizer object
            context_size: how much context is taken into computation at a time
            train_split: split contexts between training and validation
        '''
        self.train_split = train_split
        self.context_size = context_size
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

    @staticmethod
    def get_files(filepath:str="data"):
        path = Path(filepath)
        files = []
        
        for file in path.iterdir():
            if file.is_file() and file.suffix == ".txt":
                files.append(file)
        return files
        

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
    def stream_chunk(file:Path, chunk_size= 100_240_000):
        chunk = None
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

    def stream_token(self, files:list[Path],permutation:list[int], carry_leftover_to_next_file:bool=True, chunk_size= 1002400):
        leftover_temp_context = None
        needed_T = self.context_size + 1
        for file in self.stream_file(files, permutation):
            if not carry_leftover_to_next_file:
                leftover_temp_context = None
            for chungus in self.stream_chunk(file, chunk_size):
                context = None

                if leftover_temp_context is not None:
                    context = leftover_temp_context
                    leftover_temp_context = None

                # print(chungus)
                temp_context = self.tokenizer.encode(chungus)  

                if context is not None and temp_context.size < needed_T:
                    concatenated = nx.concatenate([context, temp_context])
                    if concatenated.size == needed_T:
                        yield concatenated
                    elif concatenated.size < needed_T:
                        leftover_temp_context = concatenated
                        continue
                    else:
                        temp_context = concatenated
                        context = None

                while temp_context.size >= needed_T:
                    need = needed_T - len(context) if context is not None else needed_T
                    context = nx.concatenate([context, temp_context[0:need]]) if context is not None else temp_context[0:need]
                    yield context
                    context = None
                    temp_context = temp_context[need:]

                leftover_temp_context = temp_context

    def get_total_tokens(self, files:list[Path], batch_size:int, chunk_size:int=  1_024_000):
        total_tokens = 0
        indices = [i for i in range(len(files))]

        for token in self.stream_token(files, indices, chunk_size=chunk_size):
            total_tokens += token.size
        return total_tokens

    def estimate_step(self, total_tokens, batch_size:int):
        return total_tokens // self.context_size // batch_size

    def get_pairs(self, files:list[Path], batch_size:int, chunk_size:int= 1024000):
        context_batches = []
        target_batches = []

        permutation = nx.permutation(len(files)).tolist()
        for token in self.stream_token(files, permutation, chunk_size=chunk_size):
            context_batches.append(token[:-1])
            target_batches.append(token[1:])

            if len(context_batches) == batch_size:
                dtype = token.dtype
                yield nx.array(context_batches, dtype=dtype), nx.array(target_batches,   dtype=dtype)
                context_batches = []
                target_batches = []
