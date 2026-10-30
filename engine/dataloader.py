import mmap
from engine.tokenizer import Tokenizer
from pathlib import Path
from typing import Literal,Any
from helper.validate_and_raise import validate_match
import uuid
import array
import random

class DataLoader:
    def __init__(self, filepath:str|Path, tokenizer:Tokenizer,  context_size:int=1024, batch_size:int=10, train_split:float=0.9) -> None:
        self.train_split = train_split
        self.context_size = context_size
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.filepath = filepath
        if isinstance(filepath, str):
            self.filepath = Path(filepath)

        train_files, validation_files = self.split_files(filepath, train_split)
        self.train_files = train_files
        self.validation_files = validation_files

        if len(tokenizer.vocab) <=  65_535:
            self.type_code:Literal["H", 'I'] = 'H'
        else:
            self.type_code = 'I'
        pass
    
    @staticmethod
    def get_files(filepath:str|Path="data"):
        path = Path(filepath)
        files = []

        for file in path.iterdir():
            if file.is_file() and file.suffix in [".txt", ".tokenized"]:
                files.append(file)
        return files
    
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

    @staticmethod
    def stream_files(files, permutation:list|None=None, shuffle=True):
        if permutation is None:
            permutation = [i for i in range(len(files))]
        if shuffle:
            random.shuffle(permutation)
        for i in permutation:
            yield files[i]

    @staticmethod
    def file_permutation(token_len, token_chunk_size):
        type_code = "I" if token_len <= 4_294_967_295 else "Q"
        a = array.array(type_code,range(0, token_len, token_chunk_size))
        random.shuffle(a)
        return a

    def stream_tokens(self, file:Path, tokenizer_uuid:Any|None=None, carry_leftover:bool=True):
        if file.is_file() and file.suffix == ".tokenized":
            with open(file, 'rb') as f:
                a = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                magic = a[0:9] == b"tokenized"
                if not magic: raise ValueError("unknown file")
                file_tok_id = uuid.UUID(bytes = a[13:29])
                file_str_tokId = str(file_tok_id)
                validate_match(file_tok_id, tokenizer_uuid, f"please use the tokenizer you used to tokenize this file (with id: {str(tokenizer_uuid)}), got with id: {file_str_tokId}")
                tokens = memoryview(a)[29:].cast(self.type_code)

                tsize = self.context_size+1
                for i in self.file_permutation(len(tokens),tsize):
                    y_tokens = tokens[i:i+tsize]
                    if len(y_tokens) == self.context_size + 1:
                        yield y_tokens
                # for i in range(0, len(tokens), self.context_size + 1):
                #     y_tokens = tokens[i:i+self.context_size+1]
                #     if len(y_tokens) == self.context_size + 1:
                #         yield y_tokens
                    
    def stream_batches(self, path:str|Path|list[Path]|None=None):
        if path is not None:
            if isinstance(path, str):
                iter_ = Path(path).iterdir()
            else:
                if isinstance(path, Path):
                    if path.is_dir():
                        iter_ = path.iterdir()
                    else:
                        iter_ = [path]
                elif isinstance(path, list):
                    iter_ = path
                else:
                    raise ValueError("?")
        else:
            iter_ = self.filepath.iterdir() #type:ignore

        context_batches = array.array(self.type_code)
        target_batches = array.array(self.type_code)
        for file in self.stream_files(list(iter_)):
            for tokens in self.stream_tokens(file, self.tokenizer.tokenizer_id):
                context_batches.extend(tokens[0:len(tokens)-1])
                target_batches.extend(tokens[1:len(tokens)])
                if len(context_batches) == self.batch_size * self.context_size:
                    yield context_batches,target_batches
                    context_batches = array.array(self.type_code)
                    target_batches = array.array(self.type_code)

            
                

