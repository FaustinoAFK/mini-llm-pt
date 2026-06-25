import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time

import torch

from src.data_loader import read_text_file
from src.models.transformer import MiniTransformerLanguageModel
from src.tokenizer.bpe_tokenizer import BPETokenizer


TRAIN_PATH = "data/splits/train.txt"
VAL_PATH = "data/splits/val.txt"
TOKENIZER_PATH = "artifacts/tokenizers/bpe.json"
CHECKPOINT_PATH = "checkpoints/transformer_bpe.pt"
METRICS_PATH = "artifacts/runs/transformer_bpe_metrics.jsonl"
BLOCK_SIZE = 128
BATCH_SIZE = 12
EVAL_BATCH_SIZE = 12
EVAL_NUM_BATCHES = 20
MAX_ITERS = 15000
LEARNING_RATE = 8e-4
MIN_LEARNING_RATE = 8e-5
WARMUP_ITERS = 200
N_EMBD = 256
N_HEAD = 8
N_LAYER = 4
DROPOUT = 0.1
EVAL_INTERVAL = 500
PATIENCE = 6
GRAD_CLIP = 1.0
GRADIENT_ACCUMULATION_STEPS = 2
DEVICE = "auto"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Treina o Mini Transformer com tokenizer BPE."
    )
    parser.add_argument("--train-path", default=TRAIN_PATH)
    parser.add_argument("--val-path", default=VAL_PATH)
    parser.add_argument("--tokenizer-path", default=TOKENIZER_PATH)
    parser.add_argument("--checkpoint-path", default=CHECKPOINT_PATH)
    parser.add_argument("--metrics-path", default=METRICS_PATH)
    parser.add_argument("--block-size", type=int, default=BLOCK_SIZE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--eval-batch-size", type=int, default=EVAL_BATCH_SIZE)
    parser.add_argument("--eval-num-batches", type=int, default=EVAL_NUM_BATCHES)
    parser.add_argument("--max-iters", type=int, default=MAX_ITERS)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--min-learning-rate", type=float, default=MIN_LEARNING_RATE)
    parser.add_argument("--warmup-iters", type=int, default=WARMUP_ITERS)
    parser.add_argument("--n-embd", type=int, default=N_EMBD)
    parser.add_argument("--n-head", type=int, default=N_HEAD)
    parser.add_argument("--n-layer", type=int, default=N_LAYER)
    parser.add_argument("--dropout", type=float, default=DROPOUT)
    parser.add_argument("--eval-interval", type=int, default=EVAL_INTERVAL)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--grad-clip", type=float, default=GRAD_CLIP)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=GRADIENT_ACCUMULATION_STEPS,
    )
    parser.add_argument("--disable-lr-schedule", action="store_true")
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=DEVICE)
    return parser.parse_args()


def validate_args(args):
    positive_int_fields = [
        "block_size",
        "batch_size",
        "eval_batch_size",
        "eval_num_batches",
        "max_iters",
        "n_embd",
        "n_head",
        "n_layer",
        "eval_interval",
        "patience",
        "gradient_accumulation_steps",
    ]

    for field in positive_int_fields:
        if getattr(args, field) <= 0:
            raise ValueError(f"{field} precisa ser maior que zero.")

    if args.learning_rate <= 0:
        raise ValueError("learning_rate precisa ser maior que zero.")

    if args.min_learning_rate <= 0:
        raise ValueError("min_learning_rate precisa ser maior que zero.")

    if args.min_learning_rate > args.learning_rate:
        raise ValueError("min_learning_rate nao pode ser maior que learning_rate.")

    if args.warmup_iters < 0:
        raise ValueError("warmup_iters precisa ser maior ou igual a zero.")

    if args.grad_clip <= 0:
        raise ValueError("grad_clip precisa ser maior que zero.")

    if not 0 <= args.dropout < 1:
        raise ValueError("dropout precisa ser maior ou igual a 0 e menor que 1.")


def append_jsonl(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def print_text_safely(text):
    try:
        print(text)
    except UnicodeEncodeError:
        encoding = sys.stdout.encoding or "utf-8"
        safe_text = text.encode(encoding, errors="replace").decode(
            encoding,
            errors="replace",
        )
        print(safe_text)


def resolve_device(device_name):
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device_name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA foi solicitado, mas torch.cuda.is_available() é False.")

    return torch.device(device_name)


def build_token_tensor(ids, block_size, device):
    if len(ids) < block_size + 1:
        raise ValueError("Texto tokenizado pequeno demais para o BLOCK_SIZE.")

    return torch.tensor(ids, dtype=torch.long, device=device)


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters())


def effective_batch_tokens(batch_size, block_size, gradient_accumulation_steps):
    return batch_size * block_size * gradient_accumulation_steps


def compute_learning_rate(
    step,
    max_iters,
    learning_rate,
    min_learning_rate,
    warmup_iters,
    disable_lr_schedule=False,
):
    if disable_lr_schedule:
        return learning_rate

    if warmup_iters > 0 and step < warmup_iters:
        return learning_rate * (step + 1) / warmup_iters

    decay_iters = max(1, max_iters - warmup_iters)
    decay_step = min(max(step - warmup_iters, 0), decay_iters)
    cosine = 0.5 * (1.0 + math.cos(math.pi * decay_step / decay_iters))
    return min_learning_rate + cosine * (learning_rate - min_learning_rate)


def set_optimizer_learning_rate(optimizer, learning_rate):
    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate


def get_optimizer_learning_rate(optimizer):
    return optimizer.param_groups[0]["lr"]


def get_scheduler_state(args, completed_steps):
    return {
        "completed_steps": completed_steps,
        "disable_lr_schedule": args.disable_lr_schedule,
        "learning_rate": args.learning_rate,
        "min_learning_rate": args.min_learning_rate,
        "warmup_iters": args.warmup_iters,
        "max_iters": args.max_iters,
    }


def get_rng_state(device):
    state = {"torch_rng_state": torch.get_rng_state()}
    if device.type == "cuda":
        state["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    else:
        state["cuda_rng_state_all"] = None
    return state


def restore_rng_state(checkpoint, device):
    if checkpoint.get("torch_rng_state") is not None:
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())

    cuda_rng_state_all = checkpoint.get("cuda_rng_state_all")
    if device.type == "cuda" and cuda_rng_state_all is not None:
        torch.cuda.set_rng_state_all(cuda_rng_state_all)


def load_checkpoint(path, device):
    return torch.load(path, map_location=device, weights_only=True)


def get_sequence_batch(ids, block_size, batch_size):
    if batch_size <= 0:
        raise ValueError("batch_size precisa ser maior que zero.")

    max_start = ids.numel() - block_size
    if max_start <= 0:
        raise ValueError("Texto tokenizado pequeno demais para o BLOCK_SIZE.")

    starts = torch.randint(0, max_start, (batch_size,), device=ids.device)
    offsets = torch.arange(block_size, device=ids.device)
    positions = starts[:, None] + offsets
    x = ids[positions]
    y = ids[positions + 1]
    return x, y


def estimate_loss_over_token_batches(model, ids, block_size, batch_size, num_batches):
    if num_batches <= 0:
        raise ValueError("num_batches precisa ser maior que zero.")

    was_training = model.training
    model.eval()

    losses = []
    with torch.no_grad():
        for _ in range(num_batches):
            batch_x, batch_y = get_sequence_batch(ids, block_size, batch_size)
            _, loss = model(batch_x, batch_y)
            losses.append(loss.item())

    if was_training:
        model.train()

    return sum(losses) / len(losses)


def main():
    args = parse_args()
    validate_args(args)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    tokenizer = BPETokenizer.load(args.tokenizer_path)
    train_text = read_text_file(args.train_path)
    val_text = read_text_file(args.val_path)

    train_ids = tokenizer.encode(train_text, add_bos=True, add_eos=True)
    val_ids = tokenizer.encode(val_text, add_bos=True, add_eos=True)

    print(f"Tokenizer BPE: {args.tokenizer_path}")
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Train chars: {len(train_text)} | train tokens: {len(train_ids)}")
    print(f"Val chars: {len(val_text)} | val tokens: {len(val_ids)}")
    print(f"Compressao train chars/tokens: {len(train_text) / len(train_ids):.2f}x")
    print(
        f"Config: block_size={args.block_size}, batch_size={args.batch_size}, "
        f"n_embd={args.n_embd}, n_head={args.n_head}, n_layer={args.n_layer}, "
        f"dropout={args.dropout}, lr={args.learning_rate}, "
        f"min_lr={args.min_learning_rate}, warmup_iters={args.warmup_iters}, "
        f"grad_clip={args.grad_clip}, "
        f"gradient_accumulation_steps={args.gradient_accumulation_steps}, "
        f"lr_schedule={'off' if args.disable_lr_schedule else 'on'}, "
        f"patience={args.patience}, seed={args.seed}, device={device}"
    )

    train_tensor = build_token_tensor(train_ids, args.block_size, device)
    val_tensor = build_token_tensor(val_ids, args.block_size, device)

    model = MiniTransformerLanguageModel(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_layer=args.n_layer,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    param_count = count_parameters(model)
    effective_tokens = effective_batch_tokens(
        args.batch_size,
        args.block_size,
        args.gradient_accumulation_steps,
    )

    best_val_loss = float("inf")
    best_step = None
    best_model_state_dict = None
    evaluations_without_improvement = 0
    start_step = 0

    if args.resume_from is not None:
        checkpoint = load_checkpoint(args.resume_from, device)
        model.load_state_dict(
            checkpoint.get("training_model_state_dict", checkpoint["model_state_dict"])
        )

        if checkpoint.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        best_val_loss = checkpoint.get("best_val_loss", best_val_loss)
        best_step = checkpoint.get("best_step", best_step)
        best_model_state_dict = checkpoint.get("model_state_dict")
        evaluations_without_improvement = checkpoint.get(
            "evaluations_without_improvement",
            evaluations_without_improvement,
        )
        scheduler_state = checkpoint.get("scheduler_state_dict", {})
        start_step = checkpoint.get(
            "completed_steps",
            scheduler_state.get("completed_steps", 0),
        )
        restore_rng_state(checkpoint, device)
        print(f"Retomando treino de: {args.resume_from} | start_step={start_step}")

    append_jsonl(
        args.metrics_path,
        {
            "event": "start",
            "timestamp": utc_now_iso(),
            "train_path": args.train_path,
            "val_path": args.val_path,
            "tokenizer_path": args.tokenizer_path,
            "checkpoint_path": args.checkpoint_path,
            "metrics_path": args.metrics_path,
            "resume_from": args.resume_from,
            "start_step": start_step,
            "vocab_size": tokenizer.vocab_size,
            "param_count": param_count,
            "train_chars": len(train_text),
            "train_tokens": len(train_ids),
            "val_chars": len(val_text),
            "val_tokens": len(val_ids),
            "block_size": args.block_size,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "effective_batch_tokens": effective_tokens,
            "tokens_per_step": effective_tokens,
            "eval_batch_size": args.eval_batch_size,
            "eval_num_batches": args.eval_num_batches,
            "max_iters": args.max_iters,
            "eval_interval": args.eval_interval,
            "patience": args.patience,
            "learning_rate": args.learning_rate,
            "min_learning_rate": args.min_learning_rate,
            "warmup_iters": args.warmup_iters,
            "disable_lr_schedule": args.disable_lr_schedule,
            "grad_clip": args.grad_clip,
            "n_embd": args.n_embd,
            "n_head": args.n_head,
            "n_layer": args.n_layer,
            "dropout": args.dropout,
            "seed": args.seed,
            "device": str(device),
            "requested_device": args.device,
        },
    )

    model.train()
    training_start_time = time.perf_counter()
    completed_steps = start_step
    last_batch_loss = None
    last_grad_norm = None

    for step in range(start_step, args.max_iters):
        current_lr = compute_learning_rate(
            step,
            max_iters=args.max_iters,
            learning_rate=args.learning_rate,
            min_learning_rate=args.min_learning_rate,
            warmup_iters=args.warmup_iters,
            disable_lr_schedule=args.disable_lr_schedule,
        )
        set_optimizer_learning_rate(optimizer, current_lr)

        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0

        for _ in range(args.gradient_accumulation_steps):
            batch_x, batch_y = get_sequence_batch(
                train_tensor,
                block_size=args.block_size,
                batch_size=args.batch_size,
            )
            logits, loss = model(batch_x, batch_y)
            accumulated_loss += loss.item()
            (loss / args.gradient_accumulation_steps).backward()

        last_batch_loss = accumulated_loss / args.gradient_accumulation_steps
        last_grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=args.grad_clip,
        ).item()
        optimizer.step()
        completed_steps = step + 1

        if step % args.eval_interval == 0 or step == args.max_iters - 1:
            elapsed_seconds = time.perf_counter() - training_start_time
            trained_tokens = max(completed_steps - start_step, 0) * effective_tokens
            tokens_per_second = (
                trained_tokens / elapsed_seconds if elapsed_seconds > 0 else 0.0
            )
            train_loss = estimate_loss_over_token_batches(
                model,
                train_tensor,
                block_size=args.block_size,
                batch_size=args.eval_batch_size,
                num_batches=args.eval_num_batches,
            )
            val_loss = estimate_loss_over_token_batches(
                model,
                val_tensor,
                block_size=args.block_size,
                batch_size=args.eval_batch_size,
                num_batches=args.eval_num_batches,
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_step = step
                best_model_state_dict = deepcopy(model.state_dict())
                evaluations_without_improvement = 0
            else:
                evaluations_without_improvement += 1

            print(
                f"step {step}: "
                f"batch loss {last_batch_loss:.4f} | "
                f"train loss {train_loss:.4f} | "
                f"val loss {val_loss:.4f} | "
                f"lr {get_optimizer_learning_rate(optimizer):.6f} | "
                f"grad norm {last_grad_norm:.4f} | "
                f"tok/s {tokens_per_second:.1f} | "
                f"best val {best_val_loss:.4f} at step {best_step} | "
                f"sem melhora {evaluations_without_improvement}/{args.patience}"
            )
            append_jsonl(
                args.metrics_path,
                {
                    "event": "eval",
                    "timestamp": utc_now_iso(),
                    "step": step,
                    "completed_steps": completed_steps,
                    "batch_loss": last_batch_loss,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "best_val_loss": best_val_loss,
                    "best_step": best_step,
                    "evaluations_without_improvement": evaluations_without_improvement,
                    "learning_rate": get_optimizer_learning_rate(optimizer),
                    "grad_norm": last_grad_norm,
                    "param_count": param_count,
                    "tokens_per_step": effective_tokens,
                    "effective_batch_tokens": effective_tokens,
                    "elapsed_seconds": elapsed_seconds,
                    "tokens_per_second": tokens_per_second,
                },
            )

            if evaluations_without_improvement >= args.patience:
                print(
                    f"\nEarly stopping ativado no step {step}. "
                    f"Melhor step: {best_step} | best val loss: {best_val_loss:.4f}"
                )
                break

    checkpoint_path = Path(args.checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    if best_model_state_dict is None:
        best_model_state_dict = deepcopy(model.state_dict())
        best_step = args.max_iters - 1
        best_val_loss = estimate_loss_over_token_batches(
            model,
            val_tensor,
            block_size=args.block_size,
            batch_size=args.eval_batch_size,
            num_batches=args.eval_num_batches,
        )

    torch.save(
        {
            "model_state_dict": best_model_state_dict,
            "tokenizer_type": "bpe",
            "tokenizer_path": args.tokenizer_path,
            "train_path": args.train_path,
            "val_path": args.val_path,
            "metrics_path": args.metrics_path,
            "vocab_size": tokenizer.vocab_size,
            "block_size": args.block_size,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "eval_num_batches": args.eval_num_batches,
            "max_iters": args.max_iters,
            "eval_interval": args.eval_interval,
            "patience": args.patience,
            "learning_rate": args.learning_rate,
            "min_learning_rate": args.min_learning_rate,
            "warmup_iters": args.warmup_iters,
            "disable_lr_schedule": args.disable_lr_schedule,
            "grad_clip": args.grad_clip,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "effective_batch_tokens": effective_tokens,
            "tokens_per_step": effective_tokens,
            "param_count": param_count,
            "n_embd": args.n_embd,
            "n_head": args.n_head,
            "n_layer": args.n_layer,
            "dropout": args.dropout,
            "seed": args.seed,
            "device": str(device),
            "requested_device": args.device,
            "resume_from": args.resume_from,
            "completed_steps": completed_steps,
            "best_step": best_step,
            "best_val_loss": best_val_loss,
            "evaluations_without_improvement": evaluations_without_improvement,
            "training_model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": get_scheduler_state(args, completed_steps),
            **get_rng_state(device),
        },
        checkpoint_path,
    )

    print(
        f"\nMelhor checkpoint BPE salvo em: {checkpoint_path} "
        f"| step {best_step} | val loss {best_val_loss:.4f}"
    )
    append_jsonl(
        args.metrics_path,
        {
            "event": "end",
            "timestamp": utc_now_iso(),
            "completed_steps": completed_steps,
            "best_step": best_step,
            "best_val_loss": best_val_loss,
            "learning_rate": get_optimizer_learning_rate(optimizer),
            "param_count": param_count,
            "tokens_per_step": effective_tokens,
            "effective_batch_tokens": effective_tokens,
            "checkpoint_path": str(checkpoint_path),
        },
    )

    model.load_state_dict(best_model_state_dict)
    model.eval()
    start = torch.tensor(
        [[tokenizer.stoi[tokenizer.bos_token]]],
        dtype=torch.long,
        device=device,
    )
    with torch.no_grad():
        generated_ids = model.generate(start, max_new_tokens=80)[0].cpu().tolist()
    generated_text = tokenizer.decode_for_display(generated_ids)

    print("\n--- Texto gerado pelo melhor checkpoint BPE ---")
    print_text_safely(generated_text)


if __name__ == "__main__":
    main()
