from engine.tokenizer import Tokenizer

Tokenizer.train(50432, "data", targets=[4096, 8192, 16384, 24576, 32768, 40960])