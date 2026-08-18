from engine.transformer import Transformer
from engine.tokenizer import Tokenizer
from engine.dataloader import DataLoader
from engine.activations import softmax
from helper.singleton import colorize, sleep
import engine.optimizer as optim
import engine.optimizer.scheduler as scheduler
import engine.backend as nx
from typing import Any,Union
from pathlib import Path 
import time
import pickle
import copy
import warnings
import uuid

optimizers = Union[optim.Adam, optim.AdamW, optim.SGD]
OPTIMIZER_TYPES = (optim.Adam,optim.AdamW,optim.SGD,)

DEFAULT_CONFIGS = {
    "epochs": 100,
    "max_step":50000,
    "max_val_step":"all",
    "eval_every":5,
    "validate_every":5000,
    "context_size": 64,
    "batch_size": 32,
    "optimizer":"adamw",
    "train_split":.9,
    "save":True,
    "create_checkpoint":False,
    "weights_only": True,
    "disable_compile":False
}

OPTIMIZERS = {
    "sgd": optim.SGD,
    "adamw": optim.AdamW,
    "adam": optim.Adam
}

SCHEDULER = {
    "none": None,
    "cosine_decay": scheduler.CosineDecay,
    "linear_schedule":scheduler.LinearSchedule
}

DEFAULT_OPTIMIZER_ARGS = {
    "sgd": {
        "lr":1e-2,
        "momentum":.9,
        "weight_decay":1e-4,
        "dampening":0.0,
        "use_master":False,
        "scheduler":None,
        "min_lr":None,
        },
    "adam": {
        "lr":1e-3,
        "beta1":0.9,
        "beta2":0.999,
        "epsilon":1e-8,
        "use_master":False,
        "scheduler":None,
        "min_lr":None,
        },
    "adamw":{
        "lr":1e-3,
        "beta1":0.9,
        "beta2":0.999,
        "epsilon":1e-8,
        "weight_decay":1e-2,
        "use_master":False,
        "scheduler":None,
        "min_lr":None,
        },
}

class Session:
    def __init__(self, transformer:Transformer, tokenizer:Tokenizer, init_optimizer:bool | optimizers = True, configs:dict | None = None, *, session_id=None):
        self.session_id = session_id
        if self.session_id is None:
            self.session_id = uuid.uuid4()
        self.tokenizer = tokenizer
        self.transformer = transformer
        assert len(tokenizer.vocab) == transformer.vocab_size, f"vocab size mismatch of {transformer.vocab_size} in transformer and {len(tokenizer.vocab)} in tokenizer."

        if configs is None:
            configs = {}
        configs["session_id"] = self.session_id
        self.configs = DEFAULT_CONFIGS | configs
        config_optimizer = self.configs["optimizer"].lower()
        assert config_optimizer in OPTIMIZERS, f"invalid optimizer \"{config_optimizer}\". valid optimizers: {", ".join(OPTIMIZERS.keys())}"
        self.configs["optimizer_args"] = DEFAULT_OPTIMIZER_ARGS[config_optimizer] | configs.get("optimizer_args", {})
        if transformer.dtype == nx.float32 and self.configs["optimizer_args"]["use_master"]:
            warnings.warn("master is disabled when using full precision", Warning)
            self.configs["optimizer_args"]["use_master"] = False

        if self.configs["train_split"] == 1:
            self.configs["validate_every"] = 0

        self.configs_str = copy.deepcopy(self.configs)
        if self.configs["train_split"] == 1 or self.configs["validate_every"] == 0:
            self.configs_str["validate_every"] = f"validation is disabled"
            self.configs_str["val_max_step"] = f"validation is disabled"

        if not self.configs['save']:
            self.configs_str["save"]= colorize("False", "red")
            self.configs_str["create_checkpoint"] = "disabled because save is false"

        if "disable_compile" in self.configs_str:
            if nx.backend == "MLX": 
                if self.configs_str["disable_compile"] :
                    self.configs_str["disable_compile"] = colorize("True", "red")
                    nx._nx.disable_compile() #type:ignore
            else:
                self.configs_str["disable_compile"] = "not available for current backend"
    
        if isinstance(init_optimizer, bool) and init_optimizer: 
            optimizer_class = OPTIMIZERS[self.configs["optimizer"].lower()]
            schedule = self.configs["optimizer_args"]["scheduler"]
            if schedule is not None:
                schedule = self.configs["optimizer_args"]["scheduler"].lower()
                if isinstance(schedule, str):
                    if schedule not in SCHEDULER:
                        raise ValueError(f"invalid scheduler {schedule}. valid schedulers: {", ".join(SCHEDULER.keys())}")
                    self.configs["optimizer_args"]["scheduler"] = SCHEDULER[schedule]
                
                if self.configs["optimizer_args"]["min_lr"] is None:
                    self.configs["optimizer_args"]["min_lr"] = self.configs["optimizer_args"]["lr"] / 1e2
                    self.configs_str["optimizer_args"]["min_lr"] = self.configs["optimizer_args"]["lr"] / 1e2
            else: 
                self.configs["optimizer_args"].pop("min_lr")
            self.optimizer = optimizer_class(**self.configs["optimizer_args"])
        elif isinstance(init_optimizer, OPTIMIZER_TYPES):
            self.optimizer = init_optimizer

    def __str__(self) -> str:
        t_mess = f"param: {self.transformer.count_params()} \n"
        for key,val in self.configs_str.items():
            t_mess += f"{key}: {val}\n"
        t_mess += self.transformer.get_configs_str()
        return t_mess

    @classmethod
    def build_from_files(cls):
        '''
        build from save file using filepath.
        '''
        raise NotImplementedError("not yet")

    def train(self,dataloader:DataLoader,patience:int=10, display_message:bool=True, savefile_name:str=""):
        if display_message:
            print("[training]              ")
            print(self)   
        epoch = 0
        best_val_loss = float('inf')
        validate_every = self.configs["validate_every"]
        checkpoint_every = self.configs["checkpoint_every"]
        start = time.perf_counter()
        try:
            for i in range(self.configs["epochs"]):
                epoch = i
                assert self.optimizer, "optimizer doesnt exist"
                train = self.transformer.train(dataloader, self.optimizer, self.configs["epochs"], max_step=self.configs["max_step"], eval_every=self.configs["eval_every"], microbatch_size=self.configs["microbatch_size"])
                
                final_loss = 0.0
                total_histograms = None
                total_steps = 0
                val_loss = None
                next_validate_step = validate_every
                next_checkpoint = checkpoint_every

                for loss, count, histograms, step_counter in train:
                    final_loss = loss / count
                    total_steps = step_counter
                    total_histograms = histograms
                    flag_to_check_if_validate_checkpoint_crash_with_regular_checkpoint = False

                    if dataloader.validation_files and validate_every > 0 and step_counter >= next_validate_step:
                        next_validate_step += validate_every
                        print(f"step: {step_counter} validating", end="\r")
                        val_loss = self.transformer.validate(dataloader, self.configs["max_val_step"])

                        if val_loss is not None and val_loss < best_val_loss:
                            best_val_loss = val_loss
                            if self.configs["create_checkpoint"]:
                                flag_to_check_if_validate_checkpoint_crash_with_regular_checkpoint = True
                                self.save(f"checkpoint_best_{self.session_id}")
                        else:
                            print(f"step: {step_counter}: validation becomes worse: best: {best_val_loss} | val:{val_loss}")

                    if self.configs["create_checkpoint"] and checkpoint_every > 0 and step_counter >= next_checkpoint:
                        next_checkpoint += checkpoint_every
                        if flag_to_check_if_validate_checkpoint_crash_with_regular_checkpoint:
                            flag_to_check_if_validate_checkpoint_crash_with_regular_checkpoint = False
                            continue
                        self.save(f"checkpoint_latest_{self.session_id}")

                    print(f"step: {step_counter}                                            ",end="\r" )

                if total_histograms is not None:
                    for i in range(len(total_histograms)):
                        total_histograms[i] /= total_steps
                end = time.perf_counter()   
                time_ = end-start

                display_every = max(1, self.configs["epochs"] // 10)

                if display_message and( i % display_every == 0 or i == self.configs["epochs"] - 1):
                    if val_loss is None:
                        if not dataloader.validation_files:
                            val_loss = 'no validation' 
                        else:
                            val_loss = 'validation is skipped because something is wrong' 

                    if hasattr(dataloader, "_DataLoader__cow_factor"):
                        cow_factor = dataloader._DataLoader__cow_factor #type:ignore
                        print(f"epoch {epoch} | step_counter: {total_steps}:  | avg loss: {final_loss} | avg val: {val_loss} | lr: {self.optimizer.lr:.6f} | {colorize("cow_factor:",'red', 'bold')} {colorize(str(cow_factor), 'red', 'bold')} | time: {time_}")
                    else:
                        print(f"epoch {epoch} | step_counter: {total_steps}:  | avg loss: {final_loss} | avg val: {val_loss} | lr: {self.optimizer.lr:.6f} | time: {time_}")
                    if total_histograms:
                        for idx, histogram in enumerate(total_histograms):
                            hmin = nx.min(histogram).item()
                            hmax = nx.max(histogram).item()
                            print(f"block{idx}: ideal: {1/histogram.shape[0]} | spread: {hmax-hmin} | min: {hmin} | max: {hmax}")

            if self.configs["save"]:
                infer_only = "_weights_only" if self.configs["weights_only"] else ""
                filename = f"{self.transformer.count_params()}_param_{epoch+1}_epochs{infer_only}_{self.session_id}"
                if savefile_name == "":
                    savefile_name = filename
                self.save(savefile_name)
        except ValueError as e:
            end = time.perf_counter()
            print(f"epoch {epoch}: {e}. Time elapsed: {end-start:.5f}")
            if self.configs["save"] and self.configs["create_checkpoint"]:
                self.save(f"valueerror_save_{self.session_id}")
            raise
        except OverflowError as e:
            end = time.perf_counter()
            print(f"epoch {epoch}: {e}. Time elapsed: {end-start:.5f}")
            if self.configs["save"] and self.configs["create_checkpoint"]:
                self.save(f"overflow_save_{self.session_id}")
            raise
        except KeyboardInterrupt as e:
            end = time.perf_counter()
            print(f"epoch {epoch}: {e}. Time elapsed: {end-start:.5f}")
            if self.configs["save"] and self.configs["create_checkpoint"]:
                self.save(f"keyboardinterrupt_save_{self.session_id}")
            raise
        except RuntimeError as e:
            end = time.perf_counter()
            print(f"epoch {epoch}: {e}. Time elapsed: {end-start:.5f}")
            if self.configs["save"] and self.configs["create_checkpoint"]:
                self.save(f"RuntimeError_save_{self.session_id}")
            raise
    
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
        if any(token in decoded for token in ("<EOT>", "<|endofdoc|>")):
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
            if any(token in decoded for token in ("<EOT>", "<|endofdoc|>")):
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
            "optimizer":self.optimizer.to_dict(config_only=self.configs["weights_only"]),
            "session_id":self.session_id
        }

        filename = f"session_{filename}.ram2n"
        with open(Path(f"artifacts/sessions/{filename}"), "wb") as f:
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
        optimizer = optimizer_class.from_dict(session["optimizer"])
        session_id = session["session_id"]
        
        return  cls(transformer, tokenizer, optimizer, configs=configs, session_id=session_id)
    
    @classmethod
    def create_checkpoint(cls, to_checkpoint:"Session",) -> "Session":
        transformer_checkpoint = Transformer.create_checkpoint(to_checkpoint.transformer)
        tokenizer_checkpoint = Tokenizer.from_dict(to_checkpoint.tokenizer.to_dict())
        optimizer = to_checkpoint.optimizer.from_dict(to_checkpoint.optimizer.to_dict(to_checkpoint.configs["weights_only"]))        
        checkpoint = cls(transformer=transformer_checkpoint, tokenizer = tokenizer_checkpoint, init_optimizer=optimizer)
        return checkpoint
