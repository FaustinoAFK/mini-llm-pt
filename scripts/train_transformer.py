from pathlib import Path

import torch

from src.data_loader import read_text_file
from src.evaluation import estimate_loss
from src.models.transformer import MiniTransformerLanguageModel
from src.tokenizer.char_tokenizer import CharTokenizer
from src.training_data import create_training_examples


TRAIN_PATH = "data/splits/train.txt"
VAL_PATH = "data/splits/val.txt"
CHECKPOINT_PATH = "checkpoints/transformer.pt"
BLOCK_SIZE = 32
MAX_ITERS = 300
LEARNING_RATE = 1e-3
N_EMBD = 64
N_HEAD = 4
N_LAYER = 2
DROPOUT = 0.1
EVAL_INTERVAL = 30


def build_tensors(ids, block_size):
    if len(ids) < block_size + 1:
        raise ValueError(
            "O texto é pequeno demais para o BLOCK_SIZE escolhido."
        )

    examples = create_training_examples(ids, block_size)
    x = torch.tensor([example_x for example_x, _ in examples], dtype=torch.long)
    y = torch.tensor([example_y for _, example_y in examples], dtype=torch.long)
    return x, y


def main():
    torch.manual_seed(42)

    train_text = read_text_file(TRAIN_PATH)
    val_text = read_text_file(VAL_PATH)

    tokenizer = CharTokenizer(train_text)
    train_ids = tokenizer.encode(train_text)
    val_ids = tokenizer.encode(val_text)

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

    model.train()
    for step in range(MAX_ITERS):
        logits, loss = model(x_train, y_train)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % EVAL_INTERVAL == 0 or step == MAX_ITERS - 1:
            train_loss = estimate_loss(model, x_train, y_train)
            val_loss = estimate_loss(model, x_val, y_val)
            print(
                f"step {step}: "
                f"train loss {train_loss:.4f} | "
                f"val loss {val_loss:.4f}"
            )

    checkpoint_path = Path(CHECKPOINT_PATH)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab_size": tokenizer.vocab_size,
            "stoi": tokenizer.stoi,
            "itos": tokenizer.itos,
            "unk_token": tokenizer.unk_token,
            "start_id": train_ids[0],
            "block_size": BLOCK_SIZE,
            "n_embd": N_EMBD,
            "n_head": N_HEAD,
            "n_layer": N_LAYER,
            "dropout": DROPOUT,
        },
        checkpoint_path,
    )

    print(f"\nCheckpoint salvo em: {checkpoint_path}")

    model.eval()
    start = torch.tensor([[train_ids[0]]], dtype=torch.long)
    with torch.no_grad():
        generated_ids = model.generate(start, max_new_tokens=200)[0].tolist()
    generated_text = "".join(tokenizer.itos[i] for i in generated_ids)

    print("\n--- Texto gerado ---")
    print(generated_text)


if __name__ == "__main__":
    main()
