import ast
import copy
import pickle
import time
import uuid
import warnings
from pathlib import Path
from typing import Any, Union
import safetensors as safe

import engine.backend as nx
import engine.optimizer as optim
from engine.activations import softmax
from engine.dataloader import DataLoader
from engine.optimizer import scheduler
from engine.tokenizer import Tokenizer
from engine.transformer import Transformer
from helper.singleton import colorize, sleep
from engine.transformer_block import TransformerBlock
from engine.moe import MoE
from engine.embedding import Embedding

from helper.singleton import sleep


optimizers = Union[optim.Adam, optim.AdamW, optim.SGD]
OPTIMIZER_TYPES = (optim.Adam,optim.AdamW,optim.SGD,)

DEFAULT_CONFIGS = {
    "epochs": 1,
    "max_step":5000,
    "max_val_step":None,
    "eval_every":5,
    "validate_every":1000,
    "context_size": 256,
    "batch_size": 5,
    "microbatch_size":32,
    "optimizer":"adamw",
    "train_split":.9,
    "save":True,
    "error_save":False,
    "create_checkpoint":False,
    "checkpoint_every":1000,
    "weights_only": True,
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

BACKEND_SPECIFICS = {
    "MLX":{
        "mlx_disable_compile":False,
        "mlx_save_quantized_weights_as_symmetric":True
    },
    "CuPy": {},
    "NumPy": {}
}

class Session:
    def __init__(self, transformer:Transformer, tokenizer:Tokenizer, init_optimizer:bool | optimizers | None = True, configs:dict | None = None, *, session_id=None):
        self.session_id = session_id
        if self.session_id is None:
            self.session_id = str(uuid.uuid4())
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
        self.configs["backend"] = BACKEND_SPECIFICS[nx.backend] | configs.get("backend", {})

        delete_keys = []
        for k,v in self.configs["backend"].items():
            if not k.startswith(nx.backend.lower()):
                delete_keys.append(k)
        for key in delete_keys:
            del self.configs["backend"][key]

        if not nx.backend == "MLX" and self.transformer.quantized:
            self.transformer.symmetric_quant = True

        if transformer.dtype == nx.float32:
            if self.configs["optimizer_args"]["use_master"]:
                self.configs["optimizer_args"]["use_master"] = False
                warnings.warn(colorize("master is disabled when using full precision", "yellow"), UserWarning)
            if self.transformer.gradient_scale != 1:
                self.transformer.gradient_scale = 1
                warnings.warn(colorize("gradient scale is reset to 1 when using full precision", "yellow"), UserWarning)

        if self.configs["train_split"] == 1:
            self.configs["validate_every"] = 0

        if nx.backend == "MLX":
            if self.configs["backend"]["mlx_disable_compile"]:
                nx._nx.disable_compile() #type:ignore

        self.optimizer = None
        optimizer_args = self.configs["optimizer_args"].copy()
        if isinstance(init_optimizer, bool):
            if init_optimizer:
                optimizer_class = OPTIMIZERS[self.configs["optimizer"].lower()]
                schedule = self.configs["optimizer_args"]["scheduler"]
                if schedule is not None:
                    schedule = self.configs["optimizer_args"]["scheduler"].lower()
                    if isinstance(schedule, str):
                        if schedule not in SCHEDULER:
                            raise ValueError(f"invalid scheduler {schedule}. valid schedulers: {", ".join(SCHEDULER.keys())}")
                        optimizer_args["scheduler"] = SCHEDULER[schedule]

                    if self.configs["optimizer_args"]["min_lr"] is None:
                        self.configs["optimizer_args"]["min_lr"] = optimizer_args["min_lr"] = self.configs["optimizer_args"]["lr"] / 1e2
                else:
                    self.configs["optimizer_args"].pop("min_lr")
                self.optimizer = optimizer_class(**optimizer_args)
        elif isinstance(init_optimizer, OPTIMIZER_TYPES):
            self.optimizer = init_optimizer

    def __str__(self) -> str:
        t_mess = f"param: {self.transformer.count_params()} \n"
        configs_str = copy.deepcopy(self.configs)
        if self.configs["train_split"] == 1 or self.configs["validate_every"] == 0:
            configs_str["validate_every"] = f"validation is disabled"
            configs_str["val_max_step"] = f"validation is disabled"

        if not self.configs['save']:
            configs_str["save"]= colorize("False", "red")
            configs_str["create_checkpoint"] = "disabled because save is false"

        if nx.backend == "MLX":
            if self.configs["backend"]["mlx_disable_compile"]:
                configs_str["backend"]["mlx_disable_compile"] = colorize("True", "red")

        for key,val in configs_str.items():
            if isinstance(val, dict):
                a = f"{key}: "
                b = ""
                for v_key, v_val in val.items():
                    b += f"{v_key}: {str(v_val)} | "
                a += b
                t_mess += a + "\n"
            else:
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
                        total_histograms[i] /= total_steps * self.configs["microbatch_size"]
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
                quantized = "_quantized" if self.transformer.quantized else ""
                filename = f"{self.transformer.count_params()}_param_{epoch+1}_epochs{infer_only}{quantized}_{self.session_id}"
                if savefile_name == "":
                    savefile_name = filename
                self.save(savefile_name)
                self.simple_save(savefile_name)
        except ValueError as e:
            end = time.perf_counter()
            print(f"epoch {epoch}: {e}. Time elapsed: {end-start:.5f}")
            if self.configs["save"] and self.configs["error_save"]:
                self.save(f"valueerror_save_{self.session_id}")
            raise
        except OverflowError as e:
            end = time.perf_counter()
            print(f"epoch {epoch}: {e}. Time elapsed: {end-start:.5f}")
            if self.configs["save"] and self.configs["error_save"]:
                self.save(f"overflow_save_{self.session_id}")
            raise
        except KeyboardInterrupt as e:
            end = time.perf_counter()
            print(f"epoch {epoch}: {e}. Time elapsed: {end-start:.5f}")
            if self.configs["save"] and self.configs["error_save"]:
                self.save(f"keyboardinterrupt_save_{self.session_id}")
            raise
        except RuntimeError as e:
            end = time.perf_counter()
            print(f"epoch {epoch}: {e}. Time elapsed: {end-start:.5f}")
            if self.configs["save"] and self.configs["error_save"]:
                self.save(f"RuntimeError_save_{self.session_id}")
            raise

    def inference(self, context:Any, temperature=0.8, top_k=3, top_p=0.9, n=100, mem_size=16, penalty:float=.05) -> Any:
        all_caches = None
        position = 0
        memory = []
        as_symmetric = self.configs["backend"].get("mlx_save_quantized_weights_as_symmetric", False if nx.backend == "MLX" else True)
        logits, all_caches = self.transformer.inference(context,self.configs["context_size"], all_caches, position, use_symmetric=as_symmetric)
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
            logits, all_caches = self.transformer.inference(next_token,self.configs["context_size"], all_caches, position, use_symmetric=as_symmetric)
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

    def to_dict(self) -> dict:
        convert_to_symmetric = nx.backend == "MLX" and self.configs["backend"]["mlx_save_quantized_weights_as_symmetric"] and not self.transformer.symmetric_quant

        session = {
            "configs":self.configs,
            "transformer":self.transformer.to_dict(as_symmetric=convert_to_symmetric),
            "tokenizer":self.tokenizer.to_dict(),
            "optimizer":self.optimizer.to_dict(config_only=self.configs["weights_only"]) if self.optimizer is not None else None,
        }

        return session

    def save(self, filename:str, save_artifacts:bool=False):
        '''
        Args:
            filename: the name the save file will have. session_{filename}.json
            save_artifacts: also save artifacts seperately (not implemented yet)
        '''
        session = self.to_dict()

        filename = f"session_{filename}.ram2n"
        with open(Path(f"artifacts/sessions/{filename}"), "wb") as f:
           f.write(b"RAM2N")
           f.write((1).to_bytes(4, "little"))
           pickle.dump(session, f)

    def simple_save(self, filename:str="test123", *, save_tokenizer:bool=False):
        weights = self.transformer.get_all_weights("dict") 
        quants = self.transformer.get_quant_params()

        convert_to_symmetric = nx.backend == "MLX" and self.transformer.quantized and self.configs["backend"]["mlx_save_quantized_weights_as_symmetric"] and not self.transformer.symmetric_quant
        if convert_to_symmetric:
            for k,v in weights.items():
                if f"{k}.scales" in quants:
                    scale = quants[f"{k}.scales"]
                    bias = quants[f"{k}.biases"]
                    weights[k], new_scale, new_bias = nx.quantize(nx.dequantize(v, scale, bias))
                    quants[f"{k}.scales"] = new_scale
                    quants[f"{k}.biases"] = new_bias

        if self.transformer.quantized:
            embedding = {"embedding":self.transformer.embedding.lookup_table, "embedding_scale":self.transformer.embedding.table_scale,  "embedding_bias":self.transformer.embedding.bias}
            tensors = weights | quants | embedding
        else:
            embedding = {"embedding":self.transformer.embedding.lookup_table}
            tensors = weights | embedding

        metadata = {
            "tokenizer_id": str(self.tokenizer.tokenizer_id),
            "session_configs": str(self.configs),
            "transformer_configs": str(self.transformer.configs),
            "block_configs": str(self.transformer.get_block_configs())
        }
        nx.save_safetensors(Path(f"artifacts/sessions/session_{filename}.safetensors"), tensors, metadata)

    @classmethod
    def simple_load(cls, filepath:str|Path, tokenizer:Tokenizer):
        if isinstance(filepath, str):
            filepath = Path(filepath)

        session, metadata = nx.load(filepath, format='safetensors', return_metadata=True) #type:ignore
        session_configs = ast.literal_eval(metadata["session_configs"]) #type:ignore
        transformer_configs =  ast.literal_eval(metadata["transformer_configs"]) #type:ignore
        block_configs =  ast.literal_eval(metadata["block_configs"]) #type:ignore
        tokenizer_id = metadata["tokenizer_id"] #type:ignore

        if tokenizer_id != str(tokenizer.tokenizer_id):
            raise ValueError(f"input tokenizer of id {tokenizer.tokenizer_id} doesnt match the session's tokenizer of id {tokenizer_id}")

        blocks = []
        n_block = transformer_configs["n_blocks"]
        quantized = transformer_configs["quantized"]
        dtype = transformer_configs["dtype"]

        embedding_lookuptable = session["embedding"] #type:ignore
        embedding_quants = None
    
        if quantized:
            embedding_quants = session["embedding_scale"],session["embedding_bias"] #type:ignore

        for i in range(n_block):
            configs = block_configs[i]

            attn_type = configs["attn_type"]
            attn_configs = configs["attention"]
            attn_params = (session[f"{i}.attention.Wqkv"], session[f"{i}.attention.Wo"]) #type:ignore
            attn_quants = None #type:ignore

            ff_configs = configs["ff"]
            ff_params = (session[f"{i}.ff.router"],session[f"{i}.ff.Wcombined"], session[f"{i}.ff.Wout"]) #type:ignore
            ff_quants = None #type:ignore

            rmsnorm1_configs = configs["rmsnorm1"]
            rmsnorm1_gamma = session[f"{i}.rmsnorm1.gamma"] #type:ignore
            rmsnorm2_configs = configs["rmsnorm2"]
            rmsnorm2_gamma = session[f"{i}.rmsnorm2.gamma"] #type:ignore

            if quantized:
                attn_quants = (session[f"{i}.attention.Wqkv.scales"], session[f"{i}.attention.Wo.biases"]) #type:ignore
                ff_quants = (session[f"{i}.attention.Wqkv.scales"], session[f"{i}.attention.Wo.biases"]) #type:ignore

            block = TransformerBlock.from_weights(attn_type=attn_type, attn_configs=attn_configs, attn_weights=attn_params,attn_quants=attn_quants, ff_configs=ff_configs, ff_weights=ff_params,ff_quants=ff_quants, rmsnorm1_configs=rmsnorm1_configs, rmsnorm2_configs=rmsnorm2_configs, gamma1=rmsnorm1_gamma, gamma2=rmsnorm2_gamma, dtype=dtype)
            blocks.append(block)

        embedding = Embedding.from_weights(lookuptable=embedding_lookuptable, quants=embedding_quants, dtype=dtype)
        transformer = Transformer(transformer_configs, blocks, embedding=embedding)

        session_id = session_configs["session_id"]
        session = cls(transformer=transformer, tokenizer=tokenizer, init_optimizer=False, session_id=session_id)
        return session
        
    @classmethod
    def load(cls, filepath:str|Path, *, inference_only:bool=True) -> "Session":
        """
        load the saved session file and build
        """
        with open(filepath, "rb") as f:
            magic = f.read(5)
            if magic != b"RAM2N":
                raise ValueError("unknown file")
            version = int.from_bytes(f.read(4), "little")
            session = pickle.load(f)

        configs = session["configs"]
        use_symmetric = configs["backend"].get("mlx_save_quantized_weights_as_symmetric", False if nx.backend=="MLX" else True)
        transformer = Transformer.from_dict(session["transformer"], saved_as_symmetric=use_symmetric)
        tokenizer = Tokenizer.from_dict(session["tokenizer"])
        optimizer_class = OPTIMIZERS[configs["optimizer"]]
        optimizer = optimizer_class.from_dict(session["optimizer"])
        session_id = configs["session_id"]

        return  cls(transformer, tokenizer, optimizer, configs=configs, session_id=session_id)

    @classmethod
    def create_checkpoint(cls, to_checkpoint:"Session",) -> "Session":
        transformer_checkpoint = Transformer.create_checkpoint(to_checkpoint.transformer)
        tokenizer_checkpoint = Tokenizer.from_dict(to_checkpoint.tokenizer.to_dict())
        optimizer = None if to_checkpoint.optimizer is None else to_checkpoint.optimizer.from_dict(to_checkpoint.optimizer.to_dict(to_checkpoint.configs["weights_only"]))
        checkpoint = cls(transformer=transformer_checkpoint, tokenizer = tokenizer_checkpoint, init_optimizer=optimizer)
        return checkpoint
