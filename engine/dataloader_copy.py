from engine.tokenizer import Tokenizer
import engine.backend as nx
from typing import Any, Iterator
import math
from pathlib import Path


class DataLoader:
    def __init__(self,filepath:str, tokenizer:Tokenizer, context_size:int=16, train_split=0.9, stride=8) -> None:
        '''
        Args:
            data: corpus
            tokenizer: tokenizer object
            context_size: how much context is taken into computation at a time
            train_split: split contexts between training and validation
        '''
        self.train_split = train_split
        self.context_size = context_size
        self.tokenizer = tokenizer
        self.filepath = filepath

    @staticmethod
    def get_file_permutation(filepath:str) -> tuple[list[int], list[Path]]:
        path = Path(filepath)
        files = []
        
        for file in path.iterdir():
            if file.is_file and file.suffix == ".txt":
                files.append(file)

        n_files = len(files)
        permutation = nx.permutation(n_files)
        permutation = permutation.tolist()

        return permutation, files

    @staticmethod
    def stream_file(filepath:str) -> Iterator[Path]:
        permutation, files = DataLoader.get_file_permutation(filepath)
        for idx in permutation:
            yield files[idx]

    @staticmethod
    def stream_chunk(file:Path, chunk_size= 100_240_000):
        chunk = None
        with open(file, "r") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                if not chunk[-1].isspace():
                    while True:
                        a = f.read(1)
                        if a:
                            chunk += a
                        else:
                            break
                yield chunk

    def stream_token(self, filepath:str="data",carry_leftover_to_next_file:bool=True):
        leftover_temp_context = None
        needed_T = self.context_size + 1
        for file in self.stream_file(filepath):
            if not carry_leftover_to_next_file:
                leftover_temp_context = None
            for chungus in self.stream_chunk(file):
                context = None

                if leftover_temp_context:
                    context = leftover_temp_context
                    leftover_temp_context = None

                temp_context = self.tokenizer.encode(chungus)  

                if context and temp_context.size < needed_T:
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
                    need = needed_T - len(context) if context else needed_T
                    context = nx.concatenate([context, temp_context[0:need]]) if context else temp_context[0:need]
                    yield context
                    context = None
                    temp_context = temp_context[need:]

                leftover_temp_context = temp_context

    def get_pairs(self, batch_size):
        context_batches = []
        target_batches = []

        for token in self.stream_token(self.filepath):
            context_batches.append(token[:-1])
            target_batches.append(token[1:])

            if len(context_batches) == batch_size:
                yield context_batches, target_batches
                context_batches = []
                target_batches = []
