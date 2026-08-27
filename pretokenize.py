"""
Pre-tokenize a HuggingFace dataset into a flat token array (.bin) + meta (.json).

Streaming JSON/parquet at train time is a throughput killer for a 1B run. Do this
once per source, then the training loader is just np.memmap + random offsets.

Example:
    python pretokenize.py \
        --repo HuggingFaceFW/fineweb-edu --config sample-100BT \
        --tokenizer meta-llama/Llama-3.2-1B \
        --name web --out-dir ./tokens --max-tokens 60_000_000_000

Parallelize by launching one process per data shard and passing --shard i --num-shards N,
then `cat` the parts together (the format is a flat array, so concatenation is valid).
python qwen/pretokenize.py --repo HuggingFaceFW/fineweb-edu --config sample-10BT --tokenizer Qwen/Qwen3-0.6B-Base --name web --out-dir ./tokens --max-tokens 5_000_000_000

"""

import argparse
import json
import os
import time

import numpy as np
from datasets import load_dataset
from dotenv import load_dotenv
from transformers import AutoTokenizer

load_dotenv()

BUF_DOCS_DEFAULT = 2000  # docs per tokenizer batch - fine for web-page-sized
                         # documents, but each "document" in a books/long-form
                         # dataset can be hundreds of thousands of tokens, so
                         # buffering 2000 of them for one batched tokenizer
                         # call can spike memory into the tens of GB and get
                         # OOM-killed. Override with --buf-docs for those sources.


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True, help="HF dataset repo id")
    p.add_argument("--config", default=None, help="dataset config / subset name")
    p.add_argument("--split", default="train")
    p.add_argument("--text-key", default="text")
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--name", required=True, help="short source name, used for filenames")
    p.add_argument("--out-dir", default="./tokens")
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--data-files", default=None, help="glob for a file subset, e.g. 'global-shard_01*/*.zst'")
    p.add_argument("--buf-docs", type=int, default=BUF_DOCS_DEFAULT,
                    help="docs per tokenizer batch - lower this a lot (e.g. 20-50) for "
                         "sources where each document is very long (full books, etc.)")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    eos = tok.eos_token_id
    assert eos is not None, "tokenizer has no eos_token_id; set one explicitly"

    # uint16 only works if the vocab fits. Llama-3 (128k) needs uint32 -> 2x disk.
    dtype = np.uint16 if tok.vocab_size < 65536 else np.uint32
    print(f"vocab={tok.vocab_size} -> dtype={np.dtype(dtype).name}")

    kwargs = {"split": args.split, "streaming": True}
    if args.config:
        kwargs["name"] = args.config
    if args.data_files:
        kwargs["data_files"] = args.data_files
    ds = load_dataset(args.repo, **kwargs)

    suffix = "" if args.num_shards == 1 else f".part{args.shard:03d}"
    bin_path = os.path.join(args.out_dir, f"{args.name}{suffix}.bin")

    written = 0
    n_docs = 0
    t0 = time.time()
    buf = []

    def flush(fh, buf):
        nonlocal written
        if not buf:
            return
        # verbose=False: suppresses the "sequence length longer than the
        # specified maximum" warning - harmless here since we never feed
        # these ids into the model, just pack them into a flat token array
        # (truncation defaults to False, so nothing is actually being cut).
        ids = tok(buf, add_special_tokens=False, verbose=False)["input_ids"]
        flat = []
        for seq in ids:
            flat.extend(seq)
            flat.append(eos)  # document boundary
        arr = np.asarray(flat, dtype=dtype)
        if args.max_tokens is not None and written + len(arr) > args.max_tokens:
            arr = arr[: args.max_tokens - written]
        arr.tofile(fh)
        written += len(arr)

    with open(bin_path, "wb") as fh:
        for i, ex in enumerate(ds):
            if i % args.num_shards != args.shard:
                continue
            text = ex.get(args.text_key)
            if not text:
                continue
            buf.append(text)
            n_docs += 1
            if len(buf) >= args.buf_docs:
                flush(fh, buf)
                buf = []
                if written % (50_000_000) < 5_000_000:
                    rate = written / max(time.time() - t0, 1e-6)
                    print(f"  {written/1e9:.2f}B tokens  {rate/1e6:.1f}M tok/s")
                if args.max_tokens is not None and written >= args.max_tokens:
                    break
        flush(fh, buf)

    meta = {
        "name": args.name,
        "repo": args.repo,
        "config": args.config,
        "tokenizer": args.tokenizer,
        "dtype": np.dtype(dtype).name,
        "n_tokens": int(written),
        "n_docs": n_docs,
        "eos_token_id": eos,
    }
    with open(os.path.join(args.out_dir, f"{args.name}{suffix}.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"done: {written/1e9:.3f}B tokens -> {bin_path}")


if __name__ == "__main__":
    main()