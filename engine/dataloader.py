from engine.tokenizer import Tokenizer
import engine.backend as nx
from typing import Any
import math

class DataLoader:
    def __init__(self,data:str, tokenizer:Tokenizer, context_size:int=16, train_split=0.9, stride=8) -> None:
        '''
        Args:
            data: corpus
            tokenizer: tokenizer object
            context_size: how much context is taken into computation at a time
            train_split: split contexts between training and validation
        '''
        self.data_count = len(data)
        self.train_split = train_split
        self.tokens = tokenizer.encode(data)
        self.context_size = context_size
        assert stride <= context_size, "cant have more strides"
        self.stride = stride

        windows = self.dataloader_windows_view(self.tokens, self.context_size + 1, stride)
        self.contexts = windows[:,:-1]
        self.targets = windows[:,1:]
        indices = nx.permutation(len(self.contexts))
        shuffled_contexts = self.contexts[indices] #type:ignore
        shuffled_targets = self.targets[indices]#type:ignore
        split = int(len(self.contexts) * self.train_split)
        self.train_contexts =shuffled_contexts[:split]
        self.train_targets = shuffled_targets[:split]
        self.validate_contexts = shuffled_contexts[split:]
        self.validate_targets = shuffled_targets[split:]

    def get_pairs(self, batch_size:int=32):
        """ 
        slice token per size as inputs
        """
        train_indices = nx.permutation(len(self.train_contexts))
        shuffled_contexts = self.train_contexts[train_indices]
        shuffled_targets = self.train_targets[train_indices]
        for i in range(0, len(shuffled_contexts), batch_size):
            yield(shuffled_contexts[i:i+batch_size], shuffled_targets[i:i+batch_size])
    
    def get_validation_pairs(self, batch_size:int=32):
        """ 
        slice token per size as inputs
        """
        for i in range(0, len(self.validate_targets), batch_size):
            yield(self.validate_contexts[i:i+batch_size], self.validate_targets[i:i+batch_size])

    @staticmethod
    def dataloader_windows_view( x:Any, window_shape:int,  stride=1) -> nx.ArrayLike:
        n = len(x)
        num_windows = ((n - window_shape) // stride) + 1
        shape = (num_windows,window_shape)
        strides = (stride,1)
        return nx.as_strided(x, shape, strides)
    
    def get_token_size(self):
        return self.tokens.size

    def get_pass_count(self, batch_size):
        return math.ceil(((((self.tokens.size - (self.context_size + 1)) / self.stride) + 1) * self.train_split) / batch_size)
    
    def get_compression_rate(self):
        corpus_len = self.data_count
        token_size = self.tokens.size
        ratio = ((corpus_len - token_size ) / corpus_len) * 100
        return ratio