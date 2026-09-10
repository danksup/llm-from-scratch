import array
import random
from multiprocessing import Process, Queue
from queue import Empty
from pathlib import Path
from typing import Any, Iterator, Literal
import mmap

from engine.tokenizer import Tokenizer

from helper.validate_and_raise import validate_match
import uuid


class DataLoader:
    def __init__(self, filepath:str, tokenizer:Tokenizer, context_size:int=1024, batch_size:int=10, train_split:float=0.9) -> None:
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

        assert isinstance(train_split, (float,int)), f"provide either float for train_split argument. got {train_split} of type {type(train_split)} instead"

        if isinstance(train_split, (float,int)):
            assert 0 < train_split <= 1.0, f"provide a value within (0,1] for train split. got {train_split} instead"

        train_files, validation_files = self.split_files(filepath, train_split)

        assert len(train_files) > 0, f"train_split ({train_split}) is too small. 0 files were allocated for training."

        self.train_files = train_files
        self.validation_files = validation_files

        if len(tokenizer.vocab) <=  65_535:
            self.type_code = 'H'
        else:
            self.type_code = 'I'

    @staticmethod
    def get_files(filepath:str|Path="data"):
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
    def split_files(filepath:str|Path, split_value:float=.9):
        path = Path(filepath)
        files = DataLoader.get_files(path)

        assert len(files) > 0, "no files found in directory."

        if split_value < 1.0:
            assert len(files) >= 2, "at least 2 files are needed if using validation."

        file_sizes = [i.stat().st_size for i in files]
        target_train = sum(file_sizes) * split_value
        
        sorted_sizes = [i[0] for i in sorted(enumerate(file_sizes), key=lambda x: x[1], reverse=True)]

        train_files = []
        validate_files = []

        cum = 0
        for i in sorted_sizes:
            if files[i].name.startswith("validation_"):
                validate_files.append(files[i])
            elif files[i].name.startswith("train_"):
                train_files.append(files[i])
                cum += file_sizes[i]
            else:
                curr_size = file_sizes[i]
                take = cum + curr_size
                distance_take = abs(target_train - take)
                distance_no = abs(target_train - cum)

                if distance_take < distance_no:
                    if not validate_files and i == sorted_sizes[-1] and split_value < 1.0:
                        validate_files.append(files[i])
                        break
                    train_files.append(files[i])
                    cum = take
                else:
                    validate_files.append(files[i])

        return train_files, validate_files

    def add_file(self, file_path:str|Path, where:Literal['train', "validation"]):
        path = Path(file_path)
        if path.is_file() and path.suffix in [".txt", ".tokenized"]:
            file_list = getattr(self, f"{where}_files")
            file_list.append(path)

    def add_files(self, file_path, where:Literal['train', 'validation', 'split']):
        path = Path(file_path)
        if path.is_dir():
            match where:
                case 'split':
                    train, val = self.split_files(path, self.train_split)
                    self.train_files.extend(train)
                    self.validation_files.extend(val)
                case 'train' | "validation":
                    for i in path.iterdir():
                        self.add_file(i, where)

    @staticmethod
    def stream_file(files:list[Path], permutation:list[int]) -> Iterator[Path]:
        for idx in permutation:
            yield files[idx]

    @staticmethod
    def stream_chunk(file:Path,type_code:str, chunk_size= 100_240_000, tokenizer_id_bytes:Any|None=None, *, tokenizer_id_str:str|None=None):
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
                str_id = str(uuid.UUID(bytes=this_tokenizer_id))
                validate_match(this_tokenizer_id, tokenizer_id_bytes, f"please use the tokenizer you used to tokenize this file (with id: {tokenizer_id_str}), got with id: {str_id}")
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
            for chungus in self.stream_chunk(file,self.type_code, chunk_size, self.tokenizer.tokenizer_id.bytes, tokenizer_id_str=self.tokenizer.tokenizer_id.__str__()): #type:ignore
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

                    yield context
                    context = None
                    chungus = chungus[need:]

                leftover_temp_context = chungus

    def get_pairs(self, files:list[Path],  chunk_size:int= 1024000):
        context_batches = array.array(self.type_code)
        target_batches =  array.array(self.type_code)

        permutation = [i for i in range(len(files))]
        random.shuffle(permutation)
        for token in self.stream_token(files, permutation, chunk_size=chunk_size):
            if token is None:
                continue
            context_batches.extend(token[:-1])
            target_batches.extend(token[1:])

            if len(context_batches) == self.batch_size * self.context_size:
                yield context_batches,target_batches
                context_batches = array.array(self.type_code)
                target_batches = array.array(self.type_code)

    def worker(self, Q:Queue, files:list[Path], chunk_size:int= 1024000):
        try:
            for batch in self.get_pairs(files, chunk_size):
                Q.put(batch)
            Q.put(None)
        except Exception as e:
            Q.put(e)

    def prefetch_batch(self, files:list[Path], max_queue_size:int=20, chunk_size:int= 1024000):
        queue = Queue(max_queue_size)
        process  = Process(target=self.worker, args=(queue, files, chunk_size), daemon=True)

        try:
            process.start()
            while True:
                try:
                    item = queue.get(timeout=10)
                except Empty as e:
                    if not process.is_alive():
                        raise RuntimeError("idk")
                    else:
                        continue

                if item is None:
                    break
                elif isinstance(item, Exception):
                    raise item
                
                yield item
        finally:
            if process.is_alive():
                process.terminate()
            queue.cancel_join_thread()
            queue.close()
            process.join()

    def tokenize(self, file:Path):
        filename = file.stem
        
        path = Path(f"artifacts/dataloader/{filename}.tokenized")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
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
            path = Path(f"artifacts/dataloader/{filename}.tokenized")
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "wb") as f:
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

