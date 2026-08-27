"""
Phased-mixture dataloader for LLM pretraining.

Key properties:
  * Mixture weights change over training (warmup -> domain ramp -> anneal).
  * Sampling is a pure function of (seed, global_sample_index), so resume is exact:
    just set start_step. No dataloader state to checkpoint.
  * Sources are flat token .bin files read via np.memmap -> constant RAM, no
    tokenization in the hot path.
  * Every batch carries source_ids so you can log per-source loss, which is the
    single most useful diagnostic for catching a bad mix early.

Usage:
    sources, schedule = build_default_mix("./tokens")
    audit_budget(sources, schedule, total_tokens=200e9)   # do this BEFORE launching
    loader = make_loader(sources, schedule, seq_len=4096, total_steps=48_000,
                         global_batch_size=1024, micro_batch_size=8)
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset


# --------------------------------------------------------------------------------------
# Sources
# --------------------------------------------------------------------------------------

@dataclass
class Source:
    name: str
    path: str                       # flat .bin of token ids
    dtype: np.dtype = np.uint16
    n_tokens: int = 0               # filled from meta or file size
    val_fraction: float = 0.005     # last slice of the array, held out from MixtureDataset training samples
    _mm: np.memmap | None = field(default=None, repr=False, compare=False)

    @classmethod
    def from_meta(cls, meta_path: str, val_fraction: float = 0.005) -> "Source":
        with open(meta_path) as f:
            m = json.load(f)
        bin_path = meta_path.replace(".json", ".bin")
        return cls(
            name=m["name"],
            path=bin_path,
            dtype=np.dtype(m["dtype"]),
            n_tokens=m["n_tokens"],
            val_fraction=val_fraction,
        )

    def open(self) -> np.memmap:
        # Opened lazily inside each worker; memmaps do not survive fork cleanly.
        if self._mm is None:
            self._mm = np.memmap(self.path, dtype=self.dtype, mode="r")
            if self.n_tokens == 0:
                self.n_tokens = len(self._mm)
        return self._mm

    @property
    def train_end(self) -> int:
        # Everything from here to n_tokens is validation-only. Computed off
        # n_tokens (from meta), not len(memmap), so it's available before
        # open() has ever been called.
        return int(self.n_tokens * (1.0 - self.val_fraction))


# --------------------------------------------------------------------------------------
# Schedule
# --------------------------------------------------------------------------------------

class MixtureSchedule:
    """
    phases: list of {"until": <fraction of training, cumulative>, "weights": {name: w}}

    With interpolate=True (recommended) each phase's weights are anchored at that
    phase's midpoint and linearly interpolated between anchors. Hard switches at
    phase boundaries produce visible loss spikes; ramps do not.
    """

    def __init__(self, phases: list[dict], names: list[str], interpolate: bool = True):
        assert phases, "need at least one phase"
        assert abs(phases[-1]["until"] - 1.0) < 1e-6, "last phase must end at 1.0"
        self.names = list(names)
        self.interpolate = interpolate
        self.phases = phases

        edges = [0.0] + [p["until"] for p in phases]
        self.anchors = []       # (position, weight_vector)
        for i, p in enumerate(phases):
            mid = 0.5 * (edges[i] + edges[i + 1])
            w = np.array([p["weights"].get(n, 0.0) for n in self.names], dtype=np.float64)
            s = w.sum()
            assert s > 0, f"phase {i} has all-zero weights"
            self.anchors.append((mid, w / s))
        self.edges = edges

    def weights_at(self, progress: float) -> np.ndarray:
        progress = float(np.clip(progress, 0.0, 1.0))
        if not self.interpolate:
            for i, p in enumerate(self.phases):
                if progress <= p["until"] or i == len(self.phases) - 1:
                    return self.anchors[i][1]
        pos = [a[0] for a in self.anchors]
        if progress <= pos[0]:
            return self.anchors[0][1]
        if progress >= pos[-1]:
            return self.anchors[-1][1]
        j = int(np.searchsorted(pos, progress))
        lo, hi = self.anchors[j - 1], self.anchors[j]
        t = (progress - lo[0]) / (hi[0] - lo[0])
        w = (1 - t) * lo[1] + t * hi[1]
        return w / w.sum()


# --------------------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------------------

class MixtureDataset(IterableDataset):
    def __init__(
        self,
        sources: list[Source],
        schedule: MixtureSchedule,
        seq_len: int,
        total_steps: int,
        global_batch_size: int,
        seed: int = 1234,
        start_step: int = 0,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.sources = sources
        self.schedule = schedule
        self.seq_len = seq_len
        self.total_samples = total_steps * global_batch_size
        self.global_batch_size = global_batch_size
        self.seed = seed
        self.start_sample = start_step * global_batch_size
        self.rank = rank
        self.world_size = world_size
        assert [s.name for s in sources] == schedule.names, "source order must match schedule"

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        n_workers = info.num_workers if info else 1
        worker_id = info.id if info else 0

        # Interleave across (rank, worker) so every sample index is emitted once.
        stride = self.world_size * n_workers
        offset = self.rank * n_workers + worker_id

        arrs = [s.open() for s in self.sources]
        lens = np.array([len(a) for a in arrs])

        idx = self.start_sample + offset
        while idx < self.total_samples:
            progress = idx / self.total_samples
            w = self.schedule.weights_at(progress)

            rng = np.random.default_rng([self.seed, idx])
            src = int(rng.choice(len(arrs), p=w))
            # Cap at train_end, not the full array length - the tail
            # (val_fraction of each source) is reserved for build_val_set()
            # and must never be seen during training.
            hi = min(lens[src], self.sources[src].train_end) - self.seq_len - 1
            if hi <= 0:
                raise ValueError(f"source '{self.sources[src].name}' shorter than seq_len")
            start = int(rng.integers(0, hi))

            chunk = np.asarray(arrs[src][start : start + self.seq_len + 1], dtype=np.int64)
            yield {
                "input_ids": torch.from_numpy(chunk[:-1]),
                "labels": torch.from_numpy(chunk[1:]),
                "source_id": src,
            }
            idx += stride


def collate(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
        "source_ids": torch.tensor([b["source_id"] for b in batch], dtype=torch.long),
    }


def make_loader(
    sources, schedule, seq_len, total_steps, global_batch_size,
    micro_batch_size, num_workers=4, seed=1234, start_step=0, rank=0, world_size=1,
):
    ds = MixtureDataset(
        sources, schedule, seq_len, total_steps, global_batch_size,
        seed=seed, start_step=start_step, rank=rank, world_size=world_size,
    )
    return DataLoader(
        ds,
        batch_size=micro_batch_size,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=4 if num_workers else None,
        persistent_workers=bool(num_workers),
    )


# --------------------------------------------------------------------------------------
# Validation -- fixed, deterministic windows from each source's held-out tail
# --------------------------------------------------------------------------------------

def build_val_set(sources: list[Source], seq_len: int, n_examples_per_source: int = 64, seed: int = 999):
    """Unlike MixtureDataset (fresh random samples every pass, weighted by the
    live schedule), this draws a FIXED set of windows once, only from each
    source's [train_end, n_tokens) tail - the region MixtureDataset's train
    sampling is capped away from. Same set every call, so val loss is
    comparable across evaluate() calls over the course of training, and it
    never overlaps what training actually saw.

    Returns one combined batch dict (collate()-shaped) covering all sources;
    the caller chunks it into micro-batches for evaluation.
    """
    all_inputs, all_labels, all_source_ids = [], [], []
    for i, s in enumerate(sources):
        arr = s.open()
        n = s.n_tokens or len(arr)
        val_start = s.train_end
        hi = n - seq_len - 1
        if hi <= val_start:
            raise ValueError(
                f"source '{s.name}' has no room for {n_examples_per_source} "
                f"val windows of seq_len={seq_len} after train_end={val_start} (n_tokens={n})"
            )
        rng = np.random.default_rng([seed, i])
        starts = rng.integers(val_start, hi, size=n_examples_per_source)
        for start in starts:
            chunk = np.asarray(arr[start : start + seq_len + 1], dtype=np.int64)
            all_inputs.append(chunk[:-1])
            all_labels.append(chunk[1:])
            all_source_ids.append(i)

    return {
        "input_ids": torch.from_numpy(np.stack(all_inputs)),
        "labels": torch.from_numpy(np.stack(all_labels)),
        "source_ids": torch.tensor(all_source_ids, dtype=torch.long),
    }


# --------------------------------------------------------------------------------------
# Budget audit -- run this before you burn GPU hours
# --------------------------------------------------------------------------------------

def audit_budget(sources, schedule, total_tokens, epoch_cap=4.0, n_grid=1000):
    """Integrate the schedule to get expected tokens drawn per source, and report
    how many epochs that implies. >4 epochs on any source is the danger zone."""
    grid = (np.arange(n_grid) + 0.5) / n_grid
    W = np.stack([schedule.weights_at(p) for p in grid])
    mean_w = W.mean(axis=0)

    print(f"{'source':<16} {'avg %':>7} {'tokens':>10} {'avail':>10} {'epochs':>8}")
    print("-" * 56)
    rows = []
    for i, s in enumerate(sources):
        drawn = mean_w[i] * total_tokens
        avail = s.n_tokens or os.path.getsize(s.path) // np.dtype(s.dtype).itemsize
        ep = drawn / max(avail, 1)
        flag = "  <-- OVER CAP" if ep > epoch_cap else ""
        print(f"{s.name:<16} {mean_w[i]*100:>6.1f}% {drawn/1e9:>9.1f}B {avail/1e9:>9.1f}B {ep:>7.2f}x{flag}")
        rows.append((s.name, mean_w[i], drawn, avail, ep))
    print("-" * 56)
    over = [r[0] for r in rows if r[4] > epoch_cap]
    if over:
        print(f"WARNING: {', '.join(over)} exceed {epoch_cap} epochs. "
              f"Either collect more data, generate synthetic rephrasings, or lower the weight.")
    return rows


# --------------------------------------------------------------------------------------
# The default recipe
# --------------------------------------------------------------------------------------

# DEFAULT_PHASES = [
#     {"until": 0.65, "weights": {
#         "web": 0.55, "code": 0.15, "math": 0.10, "domain": 0.15,
#         "books": 0.05, "domain_synth": 0.00}},
#     {"until": 0.90, "weights": {
#         "web": 0.30, "code": 0.10, "math": 0.10, "domain": 0.40,
#         "books": 0.05, "domain_synth": 0.05}},
#     {"until": 1.00, "weights": {   # anneal: LR -> 0 here, cleanest data only
#         "web": 0.20, "code": 0.05, "math": 0.05, "domain": 0.45,
#         "books": 0.05, "domain_synth": 0.20}},
# ]
#
# ORDER = ["web", "code", "math", "domain", "books", "domain_synth"]

DEFAULT_PHASES = [
    {"until": 0.65, "weights": {
        "web": 0.65, "code": 0.25, "math": 0.10}},
    {"until": 0.90, "weights": {
        "web": 0.40, "code": 0.30, "math": 0.30}},
    {"until": 1.00, "weights": {   # anneal: LR -> 0 here, cleanest data only
        "web": 0.20, "code": 0.40, "math": 0.40}},
]

ORDER = ["web", "code", "math"]

def build_default_mix(token_dir: str, interpolate: bool = True):
    sources = []
    for name in ORDER:
        meta = os.path.join(token_dir, f"{name}.json")
        if not os.path.exists(meta):
            print(f"skipping missing source: {name}")
            continue
        sources.append(Source.from_meta(meta))

    present = [s.name for s in sources]
    phases = [
        {"until": p["until"],
         "weights": {k: v for k, v in p["weights"].items() if k in present}}
        for p in DEFAULT_PHASES
    ]
    return sources, MixtureSchedule(phases, present, interpolate=interpolate)


if __name__ == "__main__":
    # Resolved relative to this file, not the process cwd - "./tokens" only
    # works if you happen to launch from the repo root, and silently resolves
    # to a different (nonexistent) directory otherwise, e.g. when a run
    # config's working directory defaults to the script's own folder.
    token_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tokens")

    sources, sched = build_default_mix(token_dir)
    audit_budget(sources, sched, total_tokens=9e9)

    print("\nweight trajectory:")
    print(f"{'progress':>9} " + " ".join(f"{n:>7}" for n in sched.names))
    for p in np.linspace(0, 1, 11):
        w = sched.weights_at(p)
        print(f"{p:>9.2f} " + " ".join(f"{x*100:>6.1f}%" for x in w))

    loader = make_loader(sources, sched, seq_len=1024, total_steps=100,
                         global_batch_size=64, micro_batch_size=4, num_workers=0)
    b = next(iter(loader))
    print(f"\nbatch: input_ids={tuple(b['input_ids'].shape)} "
          f"labels={tuple(b['labels'].shape)} sources={b['source_ids'].tolist()}")
