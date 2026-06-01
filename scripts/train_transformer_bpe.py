import argparse
from datetime import datetime, timezone
from copy import deepcopy
import json
from pathlib import Path

import torch

from src.batching import get_batch
from src.data_loader import read_text_file
from src.evaluation import estimate_loss_over_batches
from src.models.transformer import MiniTransformerLanguageModel
from src.tokenizer.bpe_tokenizer import BPETokenizer
from src.training_data import create_training_examples


TRAIN_PATH = "data/splits/train.txt"
VAL_PATH = "data/splits/val.txt"
TOKENIZER_PATH = "artifacts/tokenizers/bpe.json"
CHECKPOINT_PATH = "checkpoints/transformer_bpe.pt"
METRICS_PATH = "artifacts/runs/transformer_bpe_metrics.jsonl"
BLOCK_SIZE = 64
BATCH_SIZE = 16
EVAL_BATCH_SIZE = 16
EVAL_NUM_BATCHES = 20
MAX_ITERS = 10000
LEARNING_RATE = 1e-3
N_EMBD = 128
N_HEAD = 4
N_LAYER = 3
DROPOUT = 0.1
EVAL_INTERVAL = 1000
PATIENCE = 2
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
    parser.add_argument("--n-embd", type=int, default=N_EMBD)
    parser.add_argument("--n-head", type=int, default=N_HEAD)
    parser.add_argument("--n-layer", type=int, default=N_LAYER)
    parser.add_argument("--dropout", type=float, default=DROPOUT)
    parser.add_argument("--eval-interval", type=int, default=EVAL_INTERVAL)
    parser.add_argument("--patience", type=int, default=PATIENCE)
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
    ]

    for field in positive_int_fields:
        if getattr(args, field) <= 0:
            raise ValueError(f"{field} precisa ser maior que zero.")

    if args.learning_rate <= 0:
        raise ValueError("learning_rate precisa ser maior que zero.")

    if not 0 <= args.dropout < 1:
        raise ValueError("dropout precisa ser maior ou igual a 0 e menor que 1.")


def append_jsonl(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def resolve_device(device_name):
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device_name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA foi solicitado, mas torch.cuda.is_available() é False.")

    return torch.device(device_name)


def build_tensors(ids, block_size):
    if len(ids) < block_size + 1:
        raise ValueError("Texto tokenizado pequeno demais para o BLOCK_SIZE.")

    examples = create_training_examples(ids, block_size)
    x = torch.tensor([example_x for example_x, _ in examples], dtype=torch.long)
    y = torch.tensor([example_y for _, example_y in examples], dtype=torch.long)
    return x, y


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

    train_ids = tokenizer.encode(train_text)
    val_ids = tokenizer.encode(val_text)

    print(f"Tokenizer BPE: {args.tokenizer_path}")
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Train chars: {len(train_text)} | train tokens: {len(train_ids)}")
    print(f"Val chars: {len(val_text)} | val tokens: {len(val_ids)}")
    print(f"Compressao train chars/tokens: {len(train_text) / len(train_ids):.2f}x")
    print(
        f"Config: block_size={args.block_size}, batch_size={args.batch_size}, "
        f"n_embd={args.n_embd}, n_head={args.n_head}, n_layer={args.n_layer}, "
        f"dropout={args.dropout}, lr={args.learning_rate}, "
        f"patience={args.patience}, seed={args.seed}, device={device}"
    )

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
            "vocab_size": tokenizer.vocab_size,
            "train_chars": len(train_text),
            "train_tokens": len(train_ids),
            "val_chars": len(val_text),
            "val_tokens": len(val_ids),
            "block_size": args.block_size,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "eval_num_batches": args.eval_num_batches,
            "max_iters": args.max_iters,
            "eval_interval": args.eval_interval,
            "patience": args.patience,
            "learning_rate": args.learning_rate,
            "n_embd": args.n_embd,
            "n_head": args.n_head,
            "n_layer": args.n_layer,
            "dropout": args.dropout,
            "seed": args.seed,
            "device": str(device),
            "requested_device": args.device,
        },
    )

    x_train, y_train = build_tensors(train_ids, args.block_size)
    x_val, y_val = build_tensors(val_ids, args.block_size)
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_val = x_val.to(device)
    y_val = y_val.to(device)

    model = MiniTransformerLanguageModel(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_layer=args.n_layer,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    best_val_loss = float("inf")
    best_step = None
    best_model_state_dict = None
    evaluations_without_improvement = 0

    model.train()
    for step in range(args.max_iters):
        batch_x, batch_y = get_batch(x_train, y_train, args.batch_size)
        logits, loss = model(batch_x, batch_y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % args.eval_interval == 0 or step == args.max_iters - 1:
            train_loss = estimate_loss_over_batches(
                model,
                x_train,
                y_train,
                batch_size=args.eval_batch_size,
                num_batches=args.eval_num_batches,
            )
            val_loss = estimate_loss_over_batches(
                model,
                x_val,
                y_val,
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
                f"batch loss {loss.item():.4f} | "
                f"train loss {train_loss:.4f} | "
                f"val loss {val_loss:.4f} | "
                f"best val {best_val_loss:.4f} at step {best_step} | "
                f"sem melhora {evaluations_without_improvement}/{args.patience}"
            )
            append_jsonl(
                args.metrics_path,
                {
                    "event": "eval",
                    "timestamp": utc_now_iso(),
                    "step": step,
                    "batch_loss": loss.item(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "best_val_loss": best_val_loss,
                    "best_step": best_step,
                    "evaluations_without_improvement": evaluations_without_improvement,
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
        best_model_state_dict = model.state_dict()
        best_step = args.max_iters - 1
        best_val_loss = estimate_loss_over_batches(
            model,
            x_val,
            y_val,
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
            "n_embd": args.n_embd,
            "n_head": args.n_head,
            "n_layer": args.n_layer,
            "dropout": args.dropout,
            "seed": args.seed,
            "device": str(device),
            "requested_device": args.device,
            "best_step": best_step,
            "best_val_loss": best_val_loss,
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
            "best_step": best_step,
            "best_val_loss": best_val_loss,
            "checkpoint_path": str(checkpoint_path),
        },
    )

    model.load_state_dict(best_model_state_dict)
    model.eval()
    start = torch.tensor([[train_ids[0]]], dtype=torch.long, device=device)
    with torch.no_grad():
        generated_ids = model.generate(start, max_new_tokens=80)[0].cpu().tolist()
    generated_text = tokenizer.decode(generated_ids)

    print("\n--- Texto gerado pelo melhor checkpoint BPE ---")
    print(generated_text)


if __name__ == "__main__":
    main()
