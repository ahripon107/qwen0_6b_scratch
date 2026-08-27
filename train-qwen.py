import argparse
import logging
import math
import os

from dotenv import load_dotenv

from model_qwen0_6b import QwenModel, QWEN_CONFIG
from mixture_loader import build_default_mix, make_loader, build_val_set

load_dotenv()

CHECKPOINT_DIR = "./checkpoints-qwen"
CHECKPOINT_EVERY = 250
VALIDATE_EVERY = 100
VAL_EXAMPLES_PER_SOURCE = 64  # fixed, deterministic windows drawn from each
                              # source's held-out tail (see Source.val_fraction) -
                              # never seen by training, same set every evaluate() call
SEQ_LEN = 2048
BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 2
MAX_STEPS = 60000
WARMUP_STEPS = 1000
PEAK_LR = 3e-4
MIN_LR_RATIO = 0.1  # cosine decay floor, as a fraction of PEAK_LR
NUM_WORKERS = 4  # MixtureDataset shards by (rank, worker_id) stride, not by
                 # underlying dataset shard count, so this isn't capped the way
                 # the fineweb streaming loaders in gpt-2/ are.
WANDB_PROJECT = "pretrain-qwen"
MLFLOW_EXPERIMENT_NAME = "pretrain-qwen"

LOG_FILE = "./train-qwen.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),  # default mode="a" -> appends across --resume runs
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def find_latest_checkpoint():
    if not os.path.isdir(CHECKPOINT_DIR):
        return None
    steps = []
    for name in os.listdir(CHECKPOINT_DIR):
        if name.startswith("step_") and name.endswith(".pt"):
            steps.append(int(name[len("step_"):-len(".pt")]))
    if not steps:
        return None
    return os.path.join(CHECKPOINT_DIR, f"step_{max(steps)}.pt")


def make_wsd_lr_lambda(warmup_steps, decay_start_step, max_steps, min_lr_ratio):
    """Warmup-Stable-Decay: ramp up, hold at peak through the web/mixed mixture
    phases, then cosine-decay to the floor only during the anneal phase
    (decay_start_step -> max_steps). Plain warmup+cosine over the whole run
    doesn't work here: with the DEFAULT_PHASES boundaries (0.65/0.90/1.00) it
    front-loads ~98% of its decay before the anneal phase even starts, so the
    "LR -> 0 here, cleanest data only" anneal phase would train at an LR
    that's already flat and bottomed out instead of actively decaying."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        if step < decay_start_step:
            return 1.0
        if step >= max_steps:
            return min_lr_ratio
        decay_ratio = (step - decay_start_step) / max(1, max_steps - decay_start_step)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr_ratio + coeff * (1.0 - min_lr_ratio)
    return lr_lambda


def train(resume: bool = False):
    import torch
    import mlflow
    import wandb
    import torch.nn.functional as F

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_hparams = {
        "peak_lr": PEAK_LR,
        "warmup_steps": WARMUP_STEPS,
        "max_steps": MAX_STEPS,
        "min_lr_ratio": MIN_LR_RATIO,
        "batch_size": BATCH_SIZE,
        "grad_accum_steps": GRAD_ACCUM_STEPS,
        "effective_batch_size": BATCH_SIZE * GRAD_ACCUM_STEPS,
        "seq_len": SEQ_LEN,
        **QWEN_CONFIG,
    }

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    model = QwenModel(QWEN_CONFIG).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=PEAK_LR)

    start_step = 0
    tokens_seen = None
    wandb_run_id = None
    mlflow_run_id = None

    if resume:
        checkpoint_path = find_latest_checkpoint()
        if checkpoint_path is not None:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_step = checkpoint["step"]
            tokens_seen = checkpoint.get("tokens_seen")
            wandb_run_id = checkpoint.get("wandb_run_id")
            mlflow_run_id = checkpoint.get("mlflow_run_id")
            logger.info(f"Resumed from {checkpoint_path} at step {start_step}")
        else:
            logger.info("No checkpoint found, starting from scratch")

    token_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tokens")
    sources, sched = build_default_mix(token_dir)
    source_names = [s.name for s in sources]

    decay_start_step = int(sched.edges[-2] * MAX_STEPS)
    run_hparams["decay_start_step"] = decay_start_step
    lr_lambda = make_wsd_lr_lambda(WARMUP_STEPS, decay_start_step, MAX_STEPS, MIN_LR_RATIO)

    # LambdaLR needs 'initial_lr' present on each param group whenever it's
    # constructed with last_epoch != -1 (i.e. on resume).
    for group in optimizer.param_groups:
        group.setdefault("initial_lr", PEAK_LR)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda, last_epoch=start_step - 1 if start_step > 0 else -1
    )

    wandb.init(project=WANDB_PROJECT, config=run_hparams, id=wandb_run_id, resume="must" if wandb_run_id else None)
    wandb_run_id = wandb.run.id

    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    mlflow_run = mlflow.start_run(run_id=mlflow_run_id)
    mlflow_run_id = mlflow_run.info.run_id
    if start_step == 0:
        mlflow.log_params(run_hparams)

    # global_batch_size (not micro_batch_size) is what MixtureDataset uses to
    # compute progress/total_samples - it needs to be the *effective* batch
    # size so that total_steps below lines up with optimizer steps, not
    # DataLoader micro-batch iterations.
    train_loader = make_loader(
        sources, sched, seq_len=SEQ_LEN, total_steps=MAX_STEPS,
        global_batch_size=BATCH_SIZE * GRAD_ACCUM_STEPS, micro_batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS, start_step=start_step,
    )

    val_batch = build_val_set(sources, seq_len=SEQ_LEN, n_examples_per_source=VAL_EXAMPLES_PER_SOURCE)
    val_input = val_batch["input_ids"].to(device)
    val_target = val_batch["labels"].to(device)
    val_source_ids = val_batch["source_ids"]

    def evaluate():
        model.eval()
        losses = []
        with torch.no_grad():
            for i in range(0, val_input.size(0), BATCH_SIZE):
                vi = val_input[i : i + BATCH_SIZE]
                vt = val_target[i : i + BATCH_SIZE]
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    v_logits = model(vi)
                    v_loss = F.cross_entropy(v_logits.transpose(1, 2), vt, reduction="none").mean(dim=1)
                losses.append(v_loss.float())
        model.train()
        all_losses = torch.cat(losses)
        metrics = {"val/loss": float(all_losses.mean().item())}
        for i, name in enumerate(source_names):
            mask = val_source_ids == i
            if mask.any():
                metrics[f"val/loss_{name}"] = float(all_losses[mask].mean().item())
        return metrics

    model.train()
    optimizer.zero_grad()
    step = start_step
    tokens_per_step = BATCH_SIZE * GRAD_ACCUM_STEPS * SEQ_LEN
    if tokens_seen is None:
        # Older checkpoint predating this field, or a fresh run - fall back to
        # deriving it from step count under the current hyperparameters.
        tokens_seen = start_step * tokens_per_step

    step_losses = []
    step_source_ids = []

    for micro_step, batch in enumerate(train_loader):
        input_batch = batch["input_ids"].to(device, non_blocking=True)
        target_batch = batch["labels"].to(device, non_blocking=True)
        source_ids = batch["source_ids"]

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(input_batch)
            # Per-sequence loss (not just the batch mean) so it can be grouped
            # by source below - per mixture_loader.py's own docstring, per-source
            # loss is "the single most useful diagnostic for catching a bad mix early".
            per_token_loss = F.cross_entropy(
                logits.transpose(1, 2), target_batch, reduction="none"
            )
            per_seq_loss = per_token_loss.mean(dim=1)
            loss = per_seq_loss.mean()

        (loss / GRAD_ACCUM_STEPS).backward()
        step_losses.append(per_seq_loss.detach())
        step_source_ids.append(source_ids)

        if (micro_step + 1) % GRAD_ACCUM_STEPS != 0:
            continue

        step += 1
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        # Average over all GRAD_ACCUM_STEPS * BATCH_SIZE sequences that made up
        # this optimizer step, not just the last micro-batch.
        all_step_losses = torch.cat(step_losses)
        all_step_source_ids = torch.cat(step_source_ids)
        train_loss = float(all_step_losses.mean().item())
        tokens_seen += tokens_per_step
        step_losses = []
        step_source_ids = []

        if step % 10 == 0:
            peak_gb = float(torch.cuda.max_memory_allocated() / 1e9)
            reserved_gb = float(torch.cuda.memory_reserved() / 1e9)
            current_lr = float(scheduler.get_last_lr()[0])
            logger.info(f"Step {step} | loss {train_loss:.4f} | lr {current_lr:.2e} | grad_norm {float(grad_norm):.4f} | tokens_seen {tokens_seen} | GPU mem peak {peak_gb:.2f}GB / reserved {reserved_gb:.2f}GB")
            train_metrics = {
                "train/loss": train_loss,
                "train/lr": current_lr,
                "train/grad_norm": float(grad_norm),
                "train/tokens_seen": tokens_seen,
                "train/gpu_mem_peak_gb": peak_gb,
                "train/gpu_mem_reserved_gb": reserved_gb,
            }
            with torch.no_grad():
                for i, name in enumerate(source_names):
                    mask = all_step_source_ids == i
                    if mask.any():
                        train_metrics[f"train/loss_{name}"] = float(all_step_losses[mask].mean().item())
            wandb.log(train_metrics, step=step)
            mlflow.log_metrics(train_metrics, step=step)

        if step % VALIDATE_EVERY == 0:
            val_metrics = evaluate()
            logger.info(f"Step {step} | " + " | ".join(f"{k} {v:.4f}" for k, v in val_metrics.items()))
            wandb.log(val_metrics, step=step)
            mlflow.log_metrics(val_metrics, step=step)

        if step % CHECKPOINT_EVERY == 0 and step > 0:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"step_{step}.pt")
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss.item(),
                "tokens_seen": tokens_seen,
                "wandb_run_id": wandb_run_id,
                "mlflow_run_id": mlflow_run_id,
            }, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")

        if step > MAX_STEPS:
            break

    wandb.finish()
    mlflow.end_run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    train(resume=args.resume)
