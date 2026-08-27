# qwen0_6b_scratch

A from-scratch PyTorch reimplementation of the **Qwen3-0.6B** architecture, pretrained
on a custom web/code/math token mixture with a phased sampling schedule, WSD
(warmup-stable-decay) learning rate, and single-GPU or multi-GPU (DDP) training.

Nothing here depends on `transformers`' modeling code — the model is built layer by
layer in [`model_qwen0_6b.py`](model_qwen0_6b.py); `transformers` is only used to load
the Qwen tokenizer.

## Architecture

[`model_qwen0_6b.py`](model_qwen0_6b.py) implements the Qwen3 decoder-only transformer:

- Grouped-query attention (16 query heads / 8 KV heads, head_dim 128)
- QK-norm — per-head RMSNorm on queries/keys before RoPE (Qwen3's distinguishing
  feature vs. Qwen2/Llama)
- Rotary position embeddings (theta = 1,000,000, context length 32,768)
- SwiGLU feed-forward
- RMSNorm pre-normalization, tied input/output embeddings
- Residual-stream projections initialized with a depth-scaled std (`0.02 / sqrt(2*n_layers)`)

| | |
|---|---|
| Layers | 28 |
| Embedding dim | 1024 |
| Hidden (FFN) dim | 3072 |
| Attention heads | 16 (query) / 8 (KV) |
| Head dim | 128 |
| Vocab size | 151,936 (Qwen3 tokenizer) |
| Context length | 32,768 |

Run `python model_qwen0_6b.py` for a quick shape/param-count sanity check.

## Pipeline

### 1. Pretokenize data

[`pretokenize.py`](pretokenize.py) streams a HuggingFace dataset, tokenizes it, and
writes a flat token array to disk (`.bin` + a `.json` meta file) so training never
tokenizes on the hot path:

```bash
python pretokenize.py \
    --repo HuggingFaceFW/fineweb-edu --config sample-10BT \
    --tokenizer Qwen/Qwen3-0.6B-Base \
    --name web --out-dir ./tokens --max-tokens 5_000_000_000
```

Repeat per source (e.g. `web`, `code`, `math` — these are the three sources
[`mixture_loader.py`](mixture_loader.py) expects by default). Large sources can be
sharded with `--shard i --num-shards N` across parallel processes and concatenated
afterward, since the format is a flat array.

### 2. Mixture dataloader

[`mixture_loader.py`](mixture_loader.py) samples training sequences from the
pretokenized `.bin` files according to a **phased mixture schedule** — source weights
ramp between phases (e.g. more web early, more code/math later, annealing to the
cleanest data as LR decays to zero), interpolated smoothly to avoid loss spikes at
phase boundaries. Key properties:

- Sampling is a pure function of `(seed, global_sample_index)`, so resuming a run
  just means setting `start_step` — no dataloader state to checkpoint.
- Each source's `.bin` is read via `np.memmap`, keeping RAM flat regardless of dataset
  size.
- A held-out tail (`val_fraction`) of each source is reserved for validation and is
  never sampled during training.
- `audit_budget()` estimates how many epochs each source will see under the schedule
  before you burn any GPU time on it.

Run `python mixture_loader.py` for a schedule/weight-trajectory printout and a demo
batch (expects tokenized data under `./tokens`).

### 3. Train

Two training entry points, sharing the same model, data pipeline, and WSD LR schedule:

- [`train-qwen.py`](train-qwen.py) — single GPU (or CPU)
- [`qwen-train-dist.py`](qwen-train-dist.py) — multi-GPU via `DistributedDataParallel`
  + `ZeroRedundancyOptimizer`

```bash
# single GPU
python train-qwen.py [--resume]

# multi-GPU (DDP)
torchrun --nproc_per_node=<N> qwen-train-dist.py [--resume]
```

Both scripts:

- Use a **warmup-stable-decay (WSD)** LR schedule: linear warmup, hold at peak
  through the web/mixed-mixture phases, then cosine-decay to a floor only during the
  final anneal phase (so the decay lines up with the schedule's cleanest-data phase,
  not the whole run).
- Checkpoint periodically to `./checkpoints-qwen/step_<N>.pt` (model, optimizer,
  step, tokens seen, W&B/MLflow run IDs) and resume exactly from the latest one with
  `--resume`.
- Log training/validation loss (overall and per-source), LR, grad norm, and GPU
  memory to both **Weights & Biases** and **MLflow**.
- Evaluate periodically on a fixed, deterministic validation set drawn from each
  source's held-out tail.

Training expects pretokenized sources under `../../tokens` relative to the script
(adjust `token_dir` in the script, or place your `tokens/` directory accordingly).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file (not committed) with:

```
HF_TOKEN=<your huggingface token>
WANDB_API_KEY=<your wandb api key>
```

## Repo layout

```
model_qwen0_6b.py    Qwen3-style transformer implementation
pretokenize.py        HF dataset -> flat token .bin + meta .json
mixture_loader.py      Phased-mixture dataloader, schedule, validation set, budget audit
train-qwen.py          Single-GPU training loop
qwen-train-dist.py      Multi-GPU (DDP) training loop
```
