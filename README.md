# nn-from-scratch
just a fun little project i do on my days off. a small llm project built for (my) learning purposes. you can read  [here](engine/README.md) to see the flow or [here](docs/) to see the explanation of each individual piece.

logs:
- did not track
  - older stuff
  - save/load 
  - layernorm (replaced by rmsnorm)
  - attention 
  - transformer 
- apple silicon acceleration (MLX) (jun 23 2026)
- jun 24 2026
  - rmsnorm
  - multi head attention 
  - top p 
  - weight tying 
- jun 25 2026
  - dropout (jun 25 2026)
  - rope (jun 25 2026)
  - train/valid split (jun 25 2026)
  - mixed precision(jun 25 2026)
- swiglu(jun 26 2026)
- purely optimizing in june 27 2026
- purely optimizing in june 28 2026
- purely optimizing in june 29 2026
- jun 30 2026
  - KV caching 
  - grouped query attention 
- BPE tokenizer (jul 1 2026)
- jul 2 2026
  - rewrote some BPE functions in C 
  - use hashing for counting on C BPE 
  - revert back to pythonic BPE 
- optimizing BPE in jul 3 2026
- jul 4 2026
  - incremental BPE 
  - logsumexp crossentropy 
- misc stuff jul 5 2026
- lr scheduling (cosine decay) in jul 6 2026
- jul 7 2026
  - checkpointing
  - validation-loss based logging
- shuffled dataloader contexts jul 9 2026
- sliding frequency penalty jul 10 2026
- jul 13 2026
  - MoE (top-1)
  - MoE load balancing
- MoE top k jul 15 2026
- sliding windows attention jul 17 2026
- jul 18 - jul 25: optimizing, fixed mix precision
- exposing more apis,bug fix, stability jul 27 - aug 7 2026
- lazy BPE aug 8 2026
- fully lazy dataloader aug 10 2026
- gradient accumulation aug 12 2026
- prefetching encoded chunk aug 13 2026

## Ongoing:
- byte level bpe
- optimizing/cleanup/docs

## TODO (not in order):
- moe noise
- moe router z loss
- inference optimization 
- inference quantization
- conversation memory
  
#### Maybe:
- autograd
- other gpus acceleration (maybe not)
## Bugs:
### Fixed
- inference degraded after a certain number of tokens 
  - cause: position
  - fix: sliding kv position (fixed jul 1 2026) 
- AGX: exceeded compiled variants footprint limit (MLX); time increases per epoch; ram usage shoots up (jul 7 2026)
  - cause: it likely doesnt like non mlx object changing, in this case the value `t` inside optimizer being python int that increments. 
  - ~~fix: removed @nx.compile decorator from AdamW._step (jul 7 2026)~~
  - fix: initializing `t` as an mlx array.
### Open
- inference breaks when token length is too large -> generate freqs on demand when needed (jun 29 2026)

## Known issue:
### Open
- if num of last batch or microbatch not the multiplier, it will be dropped (aug 13 2026, non severe) 
  - cause: cus `if len(context_batches) == batch_size:` in dataloader and `step_counter % microbatch_size == 0:` in train
  - fix: like in stream_token, handle leftover
  
# Performance Logs
Apple M1 Pro \
param: 72512 | epochs: 1 | context_size: 64 | batch_size: 256 | embed_dim: 64 \
ff_width: 256 | optimizer: adamw | train_split: 0.9 | n_heads: 8 \
optimizer_args: {'lr': 0.001, 'beta': 0.9, 'beta2': 0.999, 'epsilon': 1e-08, 'weight_decay': 0.01} \
dataset: 5 files | using: MLX | block_size: 1 | corpus char len: 3417355 

- Date: 2026-06-28 | 1092171 function calls in 239.626 seconds | ram peaked at ~ 800MB | compiled each layer backward an forward locally
- Date: 2026-06-29 | 865328 function calls in 214.715 seconds | ram peaked at ~ 900MB | block level compilation
- Date: 2026-06-29 | 504964 function calls in 211.726 seconds | ram peaked at ~ 800MB | compiled optimizers
#### slightly different configs, the rest is the same unless otherwise stated
- Date 2026-06-30 | 623763 function calls in 204.493 seconds | ram peaked at ~ 800MB | grouped-query attention (8Q/4KV)
- Date 2026-07-04 | 131387 function calls in 153.710 seconds | ram peaked at ~ 1200MB | logsumexp cross entropy, compiled cross entropy; BPE Tokenizer, corpus char len: 1106747 param: 323712 (larger vocabulary because of BPE)
- Date 2026-07-13  | 122553 function calls in 297.455 seconds | ram peaked at ~4.6GB (stable) | MoE; param: 5256832, MoE: {'cf': 1.25, 'n_experts': 24, 'ff_width': 1024}; cosine decay LR;


#### Date 2026-08-02: 
Apple M1 Pro (MLX) \
param: 213_165_568 \
context_size: 256 | batch_size: 64 | optimizer: adamw | train_split: 0.9 | dataloader_strides: 256 \
optimizer_args: {'lr': 0.001, 'beta1': 0.9, 'beta2': 0.999, 'epsilon': 1e-08, 'weight_decay': 0.01, 'use_master': False, 'scheduler': None,'min_lr': None} \
dataset: 6 files block_size: 10 corpus char len: 4609144 -> BPE compression (16384 vocab size): 1102771. ratio = 76.074% \
max step: 61 (256 strides) | embed_dim: 512 | gradient_scale: 2048 | precision: mixed precision (mlx.core.float16) \
block configs: {'ff_hidden_width': 2048, 'ff_n_experts': 12, 'ff_topk': 2, 'ff_cf': 1.25, 'ff_init': 'glorot_uniform', 'attn_type': 'full','attn_variant': 'gqa', 'attn_n_heads': 16, 'attn_init': 'glorot_uniform', 'attn_n_kv_heads': 4} \ 
individual block configs (only difference is shown): block 2: ff_n_experts: 6 | block 3: ff_n_experts: 6 | block 4: ff_n_experts: 6 | block 5: ff_n_experts: 6 | block 6: ff_n_experts: 6 | block 7: ff_n_experts: 3 | block 8: ff_n_experts: 3 | block 9: ff_n_experts: 3 
- avg loss: 7.732100963592529 | val: 6.184670448303223 | best val loss: 6.184670448303223 | lr: 0.001 | time: 1461.225281s
  
#### Date: 2026-08-04:
Apple M1 Pro (MLX) \
param: 35_438_848 \
context_size: 256 | batch_size: 64 | optimizer: adamw | train_split: 0.9 | dataloader_strides: 256 \
optimizer_args: {'lr': 0.001, 'beta1': 0.9, 'beta2': 0.999, 'epsilon': 1e-08, 'weight_decay': 0.01, 'use_master': False, 'scheduler': None, 'min_lr': None} \
dataset: 6 files | block_size: 7 | corpus char len: 4609144 -> BPE compression (16384 vocab size): 1102771. ratio = 76.074% \
max step: 61 (256 strides) | embed_dim: 256 | gradient_scale: 2048 | precision: mixed precision (mlx.core.float16) \
block configs: {'ff_hidden_width': 768, 'ff_n_experts': 12, 'ff_topk': 2, 'ff_cf': 1.25, 'ff_init': 'glorot_uniform', 'attn_type': 'full', 'attn_variant': 'gqa', 'attn_n_heads': 16, 'attn_init': 'glorot_uniform', 'attn_n_kv_heads': 4} \
individual block configs (only difference is shown): block 2: ff_n_experts: 6 | block 3: ff_n_experts: 6 | block 4: ff_n_experts: 6 | block 5: ff_n_experts: 6 | block 6: ff_n_experts: 3 |
- avg loss: 7.254411220550537 | val: 6.278104305267334 | best val loss: 6.278104305267334 | lr: 0.001 | time: 73.973663s

### Tokenizer
- 11296590 corpus len | 2048 | fitting finished in 1614.249 py 
- 11296590 corpus len | 2048 | fitting finished in 579.541 C
- 11296590 corpus len | 2048 | fitting finished in 629.932 post optimized py (3 jul 2026)
- 11296590 corpus len | 2048 | fitting finished in 17.050 incremental BPE py (4 jul 2026)
- 195_605_563 corpus len | 8192 |fitting finished in 124.606 (4 jul 2026)
- 195_605_563 corpus len | 16384 |fitting finished in 283.527 (5 jul 2026)
- 1_351_277_738 corpus len | 24000 | fitting finished in 1437.236 (6 aug 2026)
- 1_351_277_738 corpus len | 48000 | fitting finished in 3883.186 (6 aug 2026)