from engine.transformer import Transformer
from engine.tokenizer import Tokenizer
from engine.dataloader import DataLoader
from engine.activations import softmax
from engine.optimizer.adamw import AdamW
from engine.optimizer.adam import Adam
from engine.optimizer.sgd import SGD
import engine.backend as nx
from typing import Any,Union
import time
import pickle
import numpy as np
import gc

DEFAULT_CONFIGS = {
    "epochs": 100,
    "context_size": 64,
    "batch_size": 32,
    "optimizer":"adamw",
    "train_split":.9,
}

OPTIMIZERS = {
    "sgd": SGD,
    # "momentum": Momentum,
    "adamw": AdamW,
    "adam": Adam,
}

DEFAULT_OPTIMIZER_ARGS = {
    "sgd": {
        "min_lr":0.05,
        "scheduler":None,
        "use_master":True,
        "scheduler":None,
        "min_lr":0.05,
        },
    "adam": {
        "lr":0.05,
        "beta1":0.9,
        "beta2":0.999,
        "epsilon":1e-8,
        "use_master":True,
        "scheduler":None,
        "min_lr":None,
        },
    "adamw":{
        "lr":0.03,
        "beta1":0.9,
        "beta2":0.999,
        "epsilon":1e-8,
        "weight_decay":0.01,
        "use_master":True,
        "scheduler":None,
        "min_lr":None,
        },
}

class Session:
    def __init__(self, transformer:Transformer, tokenizer:Tokenizer, init_optimizer:bool | AdamW = True, configs:dict | None = None):
        self.transformer = transformer
        self.tokenizer = tokenizer

        if configs is None:
            configs = {}

        self.configs = DEFAULT_CONFIGS | configs
        config_optimizer = self.configs["optimizer"].lower()
        assert config_optimizer in OPTIMIZERS, f"invalid optimizers \"{config_optimizer}\". choose between {", ".join(OPTIMIZERS.keys())}"
        self.configs["optimizer_args"] = DEFAULT_OPTIMIZER_ARGS[config_optimizer] | configs.get("optimizer_args", {})

        if isinstance(init_optimizer, bool) and init_optimizer: 
            optimizer_class = OPTIMIZERS[self.configs["optimizer"]]
            if self.configs["optimizer_args"]["scheduler"] and not self.configs["optimizer_args"]["min_lr"]:
                self.configs["optimizer_args"]["min_lr"] = self.configs["optimizer_args"]["lr"] / 1e2
            self.optimizer = optimizer_class(**self.configs["optimizer_args"])
        elif isinstance(init_optimizer, (AdamW, Adam, SGD)):
            self.optimizer = init_optimizer
    
    def __str__(self) -> str:
        t_mess = f"param: {self.count_params()} \n"
        for key,val in self.configs.items():
            t_mess += f"{key}: {val}\n"
        t_mess += self.transformer.get_configs_str()
        return t_mess

    @classmethod
    def build_from_files(cls):
        '''
        build from save file using filepath.
        '''
        raise NotImplementedError("not yet")
    
    def count_params(self) -> int:
        """
        whole architecture number of params
        """
        total = 0
        for i in self.transformer.blocks:
            total += i.ff.Wcombined.size
            total += i.ff.Wout.size
            total += i.ff.router.size
            total += i.attention.Wqkv.size
            total += i.attention.Wo.size
            total += i.rmsnorm1.gamma.size
            total += i.rmsnorm2.gamma.size
        total += self.transformer.embedding.lookup_table.size
        return total

    def validation(self, dataloader:DataLoader) -> Any:
        return self.transformer.validate(dataloader, self.configs["batch_size"], train_split=self.configs["train_split"])
    
    def benchmark(self, dataloader:DataLoader, _pass=1, epoch=10):
        t_mess = f"[BENCHMARKING]param: {self.count_params()} "
        t_mess += f"passes: {_pass}\n"
        for key,val in self.configs.items():
            if key != "epochs":
                t_mess += f"{key}: {val}\n"
        t_mess += self.transformer.get_configs_str()
        print(t_mess)     
        for i in range(epoch):
            start_per = time.perf_counter()
            bench_loss, loss_times, backward_times, network_optimizer_times, total_histograms, histogram_loss = self.transformer.benchmark(dataloader, self.optimizer, batch_size=self.configs["batch_size"], pass_=_pass)
            end_per = time.perf_counter()
            print("loss ",bench_loss)
            print("weighted router loss ",histogram_loss)
            print(f"loss time mean: {np.mean(loss_times)} | max times: {np.max(loss_times)} | min times: {np.min(loss_times)}")
            print(f"backward time mean: {np.mean(backward_times)} | max times: {np.max(backward_times)} | min times: {np.min(backward_times)}")
            print(f"netowk optimizer time mean: {np.mean(network_optimizer_times)} | max times: {np.max(network_optimizer_times)} | min times: {np.min(network_optimizer_times)}")
            print(f"benchmarking {i}. time: {end_per - start_per:.3f}s")
            print("active mem gb", nx.get_active_memory() / 1_000_000_000)
            print("cache mem gb",nx.get_cache_memory() / 1_000_000_000)

            if total_histograms:
                for idx, histogram in enumerate(total_histograms):
                    hmin = nx.min(histogram).item()
                    hmax = nx.max(histogram).item()
                    print(f"block{idx}: ideal: {1/histogram.shape[0]} | spread: {hmax-hmin} | min: {hmin} | max: {hmax}")

    def train(self,dataloader:DataLoader,patience:int=10, display_message:bool=True):
        """
        Args:
            patience: max bad epochs
            display_message: obvious
        train using set configs
        """
        if display_message:
            print("[training]")
            print(self)       
        epoch = 0
        checkpoint = None
        try:
            # prev_error = None
            bad_epoch = 0
            THRESHOLD = 1e-4
            best_val_loss = float('inf')
            for i in range(self.configs["epochs"]):
                epoch = i
                start = time.perf_counter()
                assert self.optimizer, "optimizer doesnt exist"
                error, total_histograms, batch_count = self.transformer.train(dataloader, self.optimizer, self.configs["epochs"], batch_size=self.configs["batch_size"])
                val_loss = self.validation(dataloader)
                end = time.perf_counter()   
                time_ = end-start

                improvement = best_val_loss - val_loss
                small_improvement = improvement > 0 and improvement < THRESHOLD
                regression = improvement < 0

                if val_loss < best_val_loss:
                    checkpoint = self.create_checkpoint(self)  
                    best_val_loss = val_loss

                if small_improvement or regression:
                    bad_epoch += 1
                    if display_message:
                        bad_epoch_reason = "improvement too small" if small_improvement else "getting worse"
                        print(f"epoch {i}: bad epoch: {bad_epoch_reason} | improvement: {improvement:.5f} | best val loss: {best_val_loss} | ({bad_epoch}/{patience}) | time: {time_}")
                else:
                    if bad_epoch > 0:
                        if display_message:
                            print(f"epoch: {i}: bad epoch count reset")
                    bad_epoch = 0

                if bad_epoch >= patience:
                    if display_message:
                        print(f"[STOPPED] epoch {i} | not getting better | avg val loss: {val_loss} | best val loss: {best_val_loss}")
                    break
                display_every = max(1, self.configs["epochs"] // 10)

                if display_message and( i % display_every == 0 or i == self.configs["epochs"] - 1):
                    print(f"epoch {epoch} | avg loss: {error} | val: {val_loss} | best val loss: {best_val_loss} | lr: {self.optimizer.lr:.6f} | time: {time_}")
                    # if total_histograms:
                    #     for idx, histogram in enumerate(total_histograms):
                    #         hmin = nx.min(histogram).item()
                    #         hmax = nx.max(histogram).item()
                    #         print(f"block{idx}: ideal: {1/histogram.shape[0]} | spread: {hmax-hmin} | min: {hmin} | max: {hmax}")

            if self.configs["save"]:
                self.save("lol")
            if checkpoint is not None:
                checkpoint.save("checkpoint_save")
        except ValueError as e:
            print(f"epoch {epoch}: {e}")
            if checkpoint is not None:
                checkpoint.save("valueerror_save")
            raise
        except OverflowError as e:
            print(f"epoch {epoch}: {e}")
            if checkpoint is not None:
                checkpoint.save("overflow_save")
            raise
        except KeyboardInterrupt as e:
            print(f"epoch {epoch}: {e}")
            if checkpoint is not None:
                checkpoint.save("keyboardinterrupt_save")
    
    def inference(self, context:Any, temperature=0.8, top_k=3, top_p=0.9, n=100, mem_size=16, penalty:float=.05) -> Any:
        all_caches = None
        position = 0
        memory = []
        logits, all_caches = self.transformer.inference(context,self.configs["context_size"], all_caches, position)
        position = context.shape[1]

        raw_token = self._sample(logits, memory,  temperature, top_k, top_p)
        memory.append(raw_token)
        token = raw_token.item()
        decoded = self.tokenizer.decode([token])
        print(decoded,end="",flush=True)
        if "<EOT>" in decoded:
            return

        next_token = nx.array([[token]], dtype=nx.int32)

        for i in range(n-1):
            logits, all_caches = self.transformer.inference(next_token,self.configs["context_size"], all_caches, position)
            raw_token = self._sample(logits, memory, temperature, top_k, top_p, penalty)

            if len(memory) >= mem_size:
                memory = memory[1:]
            memory.append(raw_token)

            token = raw_token.item()
            decoded = self.tokenizer.decode([token])
            print(decoded,end="",flush=True)
            if "<EOT>" in decoded:
                break
            next_token = nx.array([[token]], dtype=nx.int32)
            position += 1

        # print(memory)

    def _sample(self, logits, memory, temperature=0.8, top_k=3, top_p=0.9, penalty=0.05):
        if memory:
            memory = nx.array(memory, dtype=nx.int32)
            mem_array = nx.unique(memory, return_counts=True)
            mem_unique = mem_array[0]
            mem_count = mem_array[1]
            # print(memory)
            logits = logits.at[..., mem_unique].subtract(mem_count * penalty)
            # logits[...,mem_unique] -= mem_count * penalty

        probs = softmax(logits[0, -1]/temperature) 
        # print(logits.shape)

        #top k
        top_k = min(top_k, len(probs))
        top_indices = nx.argpartition(probs, -top_k)[-top_k:]

        #top p
        probs = probs[top_indices]
        sorted_order = nx.argsort(probs)[::-1]
        sorted_probs = probs[sorted_order]
        sorted_indices = top_indices[sorted_order]
        cum = nx.cumsum(sorted_probs)
        mask = cum <= top_p
    
        cutoff = int(nx.sum(mask))
        cutoff = min(cutoff + 1, len(sorted_probs))
        mask[0] = True
        if not bool(nx.all(mask)):
            first_false = nx.argmax(~mask)
            mask[first_false] = True

        keep_indices = sorted_indices[:cutoff]
        keep_probs = sorted_probs[:cutoff]
        keep_probs /= nx.sum(keep_probs)
        return nx.random_choice(keep_indices, p=keep_probs)

    def save(self, filename:str, save_artifacts:bool=False):
        '''
        Args:
            filename: the name the save file will have. session_{filename}.json
            save_artifacts: also save artifacts seperately (not implemented yet)
        '''
        session = {
            "configs":self.configs,
            "transformer":self.transformer.to_dict(),
            "tokenizer":self.tokenizer.to_dict(),
            "optimizer_states": self.optimizer.to_dict()
        }

        filename = f"session_{filename}.ram2n"
        with open(f"artifacts/sessions/{filename}", "wb") as f:
           f.write(b"RAM2N")
           f.write((1).to_bytes(4, "little"))
           pickle.dump(session, f)
    
    @classmethod
    def load(cls, filepath:str) -> "Session":
        """
        load the saved session file and build
        """
        with open(filepath, "rb") as f:
            magic = f.read(5)
            if magic != b"RAM2N":
                raise ValueError("unknown file")
            version = int.from_bytes(f.read(4), "little")
            session = pickle.load(f)

        transformer = Transformer.from_dict(session["transformer"])
        tokenizer = Tokenizer.from_dict(session["tokenizer"])
        configs = session["configs"]
        optimizer_class = OPTIMIZERS[configs["optimizer"]]
        optimizer = optimizer_class.from_dict(session["optimizer_states"])
        
        return  cls(transformer, tokenizer, optimizer, configs=configs)
    
    @classmethod
    def create_checkpoint(cls, to_checkpoint:"Session") -> "Session":
        transformer_checkpoint = Transformer.from_dict(to_checkpoint.transformer.to_dict())
        tokenizer_checkpoint = Tokenizer.from_dict(to_checkpoint.tokenizer.to_dict())
        optimizer = to_checkpoint.optimizer.from_dict(to_checkpoint.optimizer.to_dict())        
        checkpoint = cls(transformer=transformer_checkpoint, tokenizer = tokenizer_checkpoint, init_optimizer=optimizer)
        return checkpoint