from copy import deepcopy
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
BLOCK_SIZE = 64
BATCH_SIZE = 16
EVAL_BATCH_SIZE = 16
EVAL_NUM_BATCHES = 20
MAX_ITERS = 10000
LEARNING_RATE = 5e-4
N_EMBD = 128
N_HEAD = 4
N_LAYER = 3
DROPOUT = 0.2
EVAL_INTERVAL = 1000
PATIENCE = 2


def build_tensors(ids, block_size):
    if len(ids) < block_size + 1:
        raise ValueError("Texto tokenizado pequeno demais para o BLOCK_SIZE.")

    examples = create_training_examples(ids, block_size)
    x = torch.tensor([example_x for example_x, _ in examples], dtype=torch.long)
    y = torch.tensor([example_y for _, example_y in examples], dtype=torch.long)
    return x, y


def main():
    torch.manual_seed(42)

    tokenizer = BPETokenizer.load(TOKENIZER_PATH)
    train_text = read_text_file(TRAIN_PATH)
    val_text = read_text_file(VAL_PATH)

    train_ids = tokenizer.encode(train_text)
    val_ids = tokenizer.encode(val_text)

    print(f"Tokenizer BPE: {TOKENIZER_PATH}")
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Train chars: {len(train_text)} | train tokens: {len(train_ids)}")
    print(f"Val chars: {len(val_text)} | val tokens: {len(val_ids)}")
    print(f"Compressao train chars/tokens: {len(train_text) / len(train_ids):.2f}x")
    print(
        f"Config: block_size={BLOCK_SIZE}, batch_size={BATCH_SIZE}, "
        f"n_embd={N_EMBD}, n_head={N_HEAD}, n_layer={N_LAYER}, "
        f"dropout={DROPOUT}, lr={LEARNING_RATE}, patience={PATIENCE}"
    )

    x_train, y_train = build_tensors(train_ids, BLOCK_SIZE)
    x_val, y_val = build_tensors(val_ids, BLOCK_SIZE)

    model = MiniTransformerLanguageModel(
        vocab_size=tokenizer.vocab_size,
        block_size=BLOCK_SIZE,
        n_embd=N_EMBD,
        n_head=N_HEAD,
        n_layer=N_LAYER,
        dropout=DROPOUT,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float("inf")
    best_step = None
    best_model_state_dict = None
    evaluations_without_improvement = 0

    model.train()
    for step in range(MAX_ITERS):
        batch_x, batch_y = get_batch(x_train, y_train, BATCH_SIZE)
        logits, loss = model(batch_x, batch_y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % EVAL_INTERVAL == 0 or step == MAX_ITERS - 1:
            train_loss = estimate_loss_over_batches(
                model,
                x_train,
                y_train,
                batch_size=EVAL_BATCH_SIZE,
                num_batches=EVAL_NUM_BATCHES,
            )
            val_loss = estimate_loss_over_batches(
                model,
                x_val,
                y_val,
                batch_size=EVAL_BATCH_SIZE,
                num_batches=EVAL_NUM_BATCHES,
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
                f"sem melhora {evaluations_without_improvement}/{PATIENCE}"
            )

            if evaluations_without_improvement >= PATIENCE:
                print(
                    f"\nEarly stopping ativado no step {step}. "
                    f"Melhor step: {best_step} | best val loss: {best_val_loss:.4f}"
                )
                break

    checkpoint_path = Path(CHECKPOINT_PATH)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    if best_model_state_dict is None:
        best_model_state_dict = model.state_dict()
        best_step = MAX_ITERS - 1
        best_val_loss = estimate_loss_over_batches(
            model,
            x_val,
            y_val,
            batch_size=EVAL_BATCH_SIZE,
            num_batches=EVAL_NUM_BATCHES,
        )

    torch.save(
        {
            "model_state_dict": best_model_state_dict,
            "tokenizer_type": "bpe",
            "tokenizer_path": TOKENIZER_PATH,
            "vocab_size": tokenizer.vocab_size,
            "block_size": BLOCK_SIZE,
            "batch_size": BATCH_SIZE,
            "eval_batch_size": EVAL_BATCH_SIZE,
            "eval_num_batches": EVAL_NUM_BATCHES,
            "max_iters": MAX_ITERS,
            "eval_interval": EVAL_INTERVAL,
            "patience": PATIENCE,
            "learning_rate": LEARNING_RATE,
            "n_embd": N_EMBD,
            "n_head": N_HEAD,
            "n_layer": N_LAYER,
            "dropout": DROPOUT,
            "best_step": best_step,
            "best_val_loss": best_val_loss,
        },
        checkpoint_path,
    )

    print(
        f"\nMelhor checkpoint BPE salvo em: {checkpoint_path} "
        f"| step {best_step} | val loss {best_val_loss:.4f}"
    )

    model.load_state_dict(best_model_state_dict)
    model.eval()
    start = torch.tensor([[train_ids[0]]], dtype=torch.long)
    with torch.no_grad():
        generated_ids = model.generate(start, max_new_tokens=80)[0].tolist()
    generated_text = tokenizer.decode(generated_ids)

    print("\n--- Texto gerado pelo melhor checkpoint BPE ---")
    print(generated_text)


if __name__ == "__main__":
    main()
