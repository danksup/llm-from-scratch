import json
import pickle
import uuid
from collections import Counter
from pathlib import Path
from typing import Any
import ast
import re
import time

#TODO change outdated docstrings
#TODO change outdated typehints

class Tokenizer:
    def __init__(self, target_vocab_size= 1024, tokenizer_id:uuid.UUID|None=None):
        self.target_vocab_size = target_vocab_size
        self.merge_rank = {}
        self.id_to_token = {0:"<PAD>".encode('utf-8'), 1: "<EOT>".encode('utf-8'), 2:"<|endofdoc|>".encode('utf-8')}
        self.vocab = {"<PAD>".encode('utf-8'):0, "<EOT>".encode('utf-8'):1, "<|endofdoc|>".encode('utf-8'):2}

        self.tokenizer_id = tokenizer_id
        if self.tokenizer_id is None:
            self.tokenizer_id = uuid.uuid4()

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, Tokenizer):
            return NotImplemented
        return (self.vocab == value.vocab) and (self.id_to_token == value.id_to_token)

    def init(self):
        for i in range(256):
            byte = bytes([i])
            next_id = len(self.vocab)
            self.vocab[byte] = next_id
            self.id_to_token[next_id] = byte

    def word_to_ids(self, word:str) -> list[int]:
        """
        turn each character in a word into id and add `</w>` last\n

        assume in vocab the id for
            h is 3
            e is 4
            l is 8
            o is 7
            `</w>` is 5
        then "hello" -> [3,4,8,8,7,5]\n
        """
        tokenized = []
        encoded_word = word.encode()
        for b in encoded_word:
            tokenized.append(self.vocab.get(bytes([b])))
        # for ch in word:
        #     ch = ch.encode()
        #     tokenized.append(self.vocab.get(ch))
        # tokenized.append(self.vocab["</w>".encode('utf-8')])
        return tokenized

    @staticmethod
    def get_word_counts(tokenized_words:list[list[int]]) -> dict[tuple[int,...],int]:
        """
        count the frequency of each tokenized word in a list
        """
        counter = {}
        for tokenized_word in tokenized_words:
            tword = tuple(tokenized_word)
            counter[tword] = counter.get(tword, 0) + 1
        return counter

    @staticmethod
    def get_pairs(tokenized_word:tuple[int, ...] | list[int]):
        '''
        yield adjacent pairs of a tokenized word. idk why lazily. maybe because of old implementation, then i didnt bother or forgot to change this.
        '''
        for i in range(len(tokenized_word) - 1):
            yield (tokenized_word[i], tokenized_word[i + 1])

    @staticmethod
    def get_pair_counts( word_counts:dict[tuple[int,...],int]) -> Counter[tuple[int,int]]:
        """
        for each key of tokenized word, count the frequency of djacent token pairss.\n
        ex: "hello hero" then word_counts could be-> {(3,4,8,8,7,5):1,(3,4,9,7,5):1}\n
        therefore, the adjacent token pairs are (3,4),(4,8),(8,8),(8,7),(7,5),(4,9),(9,7) combined, including `</w>`\n
        `+= count` because instead of scanning the entire corpus, we calculate the word's frequency, which is the `word_counts`, then add pair frequency based on the frequency of the word\n
        then, count the frequency of the pairs:
            Counter[(3,4)] = 2,
            Counter[(4,8)] = 1,
            Counter[(8,8)] = 1,
            ...
            Counter[(7,5)] = 2,
        """
        counts = Counter()
        for word, count in word_counts.items():
            for pair in Tokenizer.get_pairs(word):
                counts[pair] += count
        return counts

    @staticmethod
    def build_pair_index(word_counts:dict[tuple[int,...],int]) -> tuple[Counter[tuple[int,int]], dict[tuple[int,int],set[tuple[int,...]]]]:
        '''
        for each key of tokenized word in `word_count`, get the adjacent pairs.\n
        for each adjacent pairs,
            count the frequency of the pair using `Counter()`
            get all words that have the pair

        ex:
            word_counts = {(3,4,8,8,7,5):1,(3,4,9,7,5):1}
            then `get_pairs` will be (3,4),(4,8),(8,8),(8,7),(7,5),(3,4),(4,9),(9,7) combined
            then `counts` will be:
                Counter[(3,4)] = 2,
                Counter[(4,8)] = 1,
                Counter[(8,8)] = 1,
                ...
                Counter[(7,5)] = 2,
            and `pair_to_words` will be:
                {(3,4): set((3,4,8,8,7,5),(3,4,9,7,5)), (4,8): set((3,4,8,8,7,5)),... }

        returns both `counts:Counter` and `pair_to_words:dict`
        '''
        counts = Counter()
        pair_to_words = {}
        for word, count in word_counts.items():
            for pair in Tokenizer.get_pairs(word):
                counts[pair] += count
                if pair not in pair_to_words:
                    pair_to_words[pair] = set()
                pair_to_words[pair].add(word)

        return counts, pair_to_words

    @staticmethod
    def remove_word(word:tuple[int,...], freq:int, pair_counts:Counter[tuple[int,int]], pair_to_words:dict[tuple[int,int],set[tuple[int,...]]]):
        '''
        re
        '''
        word_pairs = list( Tokenizer.get_pairs(word))
        for pair in word_pairs:
            pair_counts[pair] -= freq

        set_pairs = set(word_pairs)
        for pair in set_pairs:
            if pair_counts[pair] <= 0:
                del pair_counts[pair]

        for pair in set_pairs:
            pair_to_words[pair].discard(word)

            if not pair_to_words[pair]:
                pair_to_words.pop(pair)

    @staticmethod
    def add_word(word:tuple[int,...], freq:int, pair_counts:Counter[tuple[int,...]], pair_to_words:dict[tuple[int,...],set[tuple[int,...]]]):
        """
        add a tokenized word into `pair_to_word` dictionary. mutates the `pair_to_words` dict.\n
        because we add a word, we need to update the pair frequencies based on the frequency of the tokenized word. this mutates the `pair_counts` dict.
        """
        word_pairs = list(Tokenizer.get_pairs(word))

        for pair in word_pairs:
            pair_counts[pair] += freq

        for pair in set(word_pairs):
            if pair not in pair_to_words:
                pair_to_words[pair] = set()

            pair_to_words[pair].add(word)

    @staticmethod
    def merge(word:list[int] | tuple[int,...], best_pair, new_id) -> list[int]:
        '''
        replace all occurences of `best_pair` in a tokenized word with `new_id`

        Example:
            word = [3, 4, 8, 8, 7, 5]      # "hello", including `</w`\n
            best_pair = (3,4)\n
            new_id = 12\n
            therefore new_word = [12,8,8,7,5]\n

        Returns:
            new_word: list[int]
        '''
        i = 0
        n = len(word)
        new_word = []
        while i < n - 1:
            pair = word[i], word[i + 1]
            if pair == best_pair:
                new_word.append(new_id)
                i += 2
            else:
                new_word.append(word[i])
                i+=1
        if i == n - 1:
            new_word.append(word[n-1])

        return new_word

    def stream_corpus(self, filepath:str, batch_size:int=10_485_760):
        path = Path(filepath)
        for file in path.iterdir():
            if file.is_file() and file.suffix == ".txt":
                with open(file, "r", encoding="utf-8", errors='ignore') as f:
                    while True:
                        chunk = f.read(batch_size)
                        if not chunk:
                            break

                        if chunk and not chunk[-1].isspace():
                            while True:# or stream[-1] not in  ["", " ", "\r",]:
                                a = f.read(1)
                                if a:
                                    chunk += a
                                    if a.isspace():
                                        break
                                else:
                                    break
                            yield chunk

    def fit(self, filepath:str, batch_size=10_485_760, *, targets:list|None=None):
        '''
        fill vocabs until specified amount (from `self.target_vocab_size`)
        '''
        global_word_count = {}
        total_char = 0
        self.init()
        print("preprocessing")
        for batch in self.stream_corpus(filepath, batch_size):
            encoded_batch  = re.findall(r'\s*\S+', batch)
            total_char += len(batch)
            words = [self.word_to_ids(word) for word in encoded_batch]

            word_counts = self.get_word_counts(words)

            if not global_word_count:
                global_word_count = word_counts
            else:
                for key, val in word_counts.items():
                    global_word_count[key] = global_word_count.get(key, 0) + val

        pair_counts, pair_to_words = self.build_pair_index(global_word_count)

        target = None
        if targets:
            targets.sort()
            targets = [i for i in targets if i < self.target_vocab_size]
            target = targets.pop(0)
        print("finished preprocessing")
        while True:
            if target is not None:
                if len(self.vocab) >= target:
                    self.total_char_raw = total_char
                    yield len(self.vocab)
                    if targets:
                        target = targets.pop(0)
                    else:
                        target = None

            if len(self.vocab) >= self.target_vocab_size:
                break
                
            if not pair_counts:
                break
            best_pair = pair_counts.most_common(1)[0][0]
            affected_words:set[tuple[int,...]] = pair_to_words[best_pair].copy()

            new_id = len(self.vocab)
            for affected_word in affected_words:
                freq = global_word_count.pop(affected_word)
                self.remove_word(affected_word, freq, pair_counts, pair_to_words)
                merged = tuple(self.merge(affected_word, best_pair, new_id))
                global_word_count[merged] = global_word_count.get(merged, 0) + freq
                self.add_word(merged, freq, pair_counts, pair_to_words)

            merged_best = self.id_to_token[best_pair[0]] + self.id_to_token[best_pair[1]]

            len_vocab = len(self.vocab)
            self.merge_rank[best_pair] = (len(self.merge_rank), new_id)
            self.vocab[merged_best] = len_vocab
            self.id_to_token[len_vocab] = merged_best

        self.total_char_raw = total_char
        yield len(self.vocab)


    def encode(self, text: Any) -> list[int]:
        text = re.findall(r'\s*\S+', text)
        words = [self.word_to_ids(word) for word in text]

        for idx, word in enumerate(words):
            while True:
                best_pair = None
                best_rank = float("inf")
                best_new_id = None

                for i in range(len(word) - 1):
                    pair = (word[i], word[i + 1])

                    if pair in self.merge_rank:
                        rank, new_id = self.merge_rank[pair]

                        if rank < best_rank:
                            best_rank = rank
                            best_pair = pair
                            best_new_id = new_id

                if best_pair is None:
                    break

                word = self.merge(word, best_pair, best_new_id)
                words[idx] = word

        tokens = [token for word in words for token in word]

        return tokens

    def decode(self, thing:list[int]) -> str:
        decoded_bytes = bytearray()

        for token_id in thing:
            if token_id == self.vocab["<PAD>".encode('utf-8')]:
                continue
            decoded_bytes.extend(self.id_to_token[token_id])

        # decoded = decoded.replace("</w>", " ")
        return decoded_bytes.decode('utf-8', errors='replace')

    @classmethod
    def train(cls, vocab_size, filepath:str = "data", *, targets:list|None=None):
        tokenizer = cls(vocab_size)
        a = tokenizer.fit(filepath, targets=targets)
        start = time.perf_counter()
        
        for i in a:
            end = time.perf_counter()
            tokenizer_save_name = f"{i}_{tokenizer.total_char_raw}len"
            if i != vocab_size:
                tokenizer.save(tokenizer_save_name, new_id=True)
            else:
                tokenizer.save(tokenizer_save_name)
            print(f"{tokenizer_save_name} saved. fitting finished in {end-start:.3f}")
        
    def to_dict(self,*, include_id:bool=False) -> dict[str,dict[Any,Any]]:
        vocab = {
            "merge_rank":self.merge_rank.copy(),
            "vocab":self.vocab.copy(),
            "id_to_token":self.id_to_token.copy(),
        }
        if include_id:
            vocab["tokenizer_id"] = str(self.tokenizer_id)
        return vocab

    @classmethod
    def from_dict(cls,thing:dict[str,Any], *, tokenizer_id=None) -> "Tokenizer":
        if isinstance(tokenizer_id, bytes):
            tokenizer_id = uuid.UUID(bytes=tokenizer_id)
        elif isinstance(tokenizer_id, str):
            tokenizer_id = uuid.UUID(tokenizer_id)
        elif tokenizer_id is None:
            tokenizer_id = thing.get("tokenizer_id", None)

        vocab_size = len(thing["vocab"])
        tokenizer = cls(vocab_size, tokenizer_id = tokenizer_id)

        tokenizer.vocab = thing["vocab"]
        tokenizer.id_to_token = thing["id_to_token"]
        tokenizer.merge_rank = thing["merge_rank"]

        return tokenizer

    def save(self, filename:str, to_json=False, *, new_id:bool=False):
        tokenizer_id = uuid.uuid4() if new_id else self.tokenizer_id
        tokenizer:Any = self.to_dict()
        filename = f"tokenizer{filename}"
        tokenizer["tokenizer_id"] = str(tokenizer_id)

        if to_json:
            merge_rank = {}
            for key, val in tokenizer["merge_rank"].items():
                merge_rank[str(list(key)).replace(" ","")] = val
            tokenizer["merge_rank"] = merge_rank

            vocab = {}
            for key, val in tokenizer["vocab"].items():
                vocab[key.decode('latin-1')] = val
            tokenizer["vocab"] = vocab

            id_to_token = {}
            for key, val in tokenizer["id_to_token"].items():
                id_to_token[key] = val.decode('latin-1')
            tokenizer["id_to_token"] = id_to_token

            with open(Path(f"artifacts/tokenizer/{filename}.json"), "w") as f:
                json.dump(tokenizer, f, indent=4)
            return

        with open(Path(f"artifacts/tokenizer/{filename}.tokenizer"), "wb") as f:
            f.write(b"tokenizer")
            f.write((1).to_bytes(4, "little"))
            f.write(tokenizer_id.bytes) #type:ignore

            pickle.dump(tokenizer, f)

    @classmethod
    def load(cls, filepath:str) -> "Tokenizer":
        path = Path(filepath)

        if path.suffix == ".tokenizer":
            with open(path, "rb") as f:
                magic = f.read(9)
                if magic != b"tokenizer":
                    raise ValueError("unknown file")
                version = int.from_bytes(f.read(4), "little")
                tokenizer_id = f.read(16)
                loaded = pickle.load(f)

            tokenizer = cls.from_dict(loaded, tokenizer_id=tokenizer_id)
            return tokenizer

        elif path.suffix == ".json":
            with open(path, "r") as f:
                loaded = json.load(f)

                merge_rank = {}
                for key,val in loaded["merge_rank"].items():
                    merge_rank[tuple(ast.literal_eval(key))] = val
                loaded["merge_rank"] = merge_rank

                id_to_token = {}
                for key,val in loaded["id_to_token"].items():
                    id_to_token[int(key)] = val.encode('latin-1')
                loaded["id_to_token"] = id_to_token

                vocab = {}
                for key,val in loaded["vocab"].items():
                    vocab[key.encode('latin-1')] = val
                loaded["vocab"] = vocab

            tokenizer = cls.from_dict(loaded)
            return tokenizer

        else:
            raise ValueError("no")
