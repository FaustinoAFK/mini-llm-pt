import torch

from src.data_loader import read_text_file
from src.models.bigram import BigramLanguageModel
from src.tokenizer.char_tokenizer import CharTokenizer
from src.training_data import create_training_examples


TRAIN_PATH = "data/splits/train.txt"
BLOCK_SIZE = 16
MAX_ITERS = 200
LEARNING_RATE = 1e-2


def main():
    torch.manual_seed(42)

    text = read_text_file(TRAIN_PATH)
    tokenizer = CharTokenizer(text)
    ids = tokenizer.encode(text)

    if len(ids) < BLOCK_SIZE + 1:
        raise ValueError(
            "O texto de treino é pequeno demais para o BLOCK_SIZE escolhido."
        )

    examples = create_training_examples(ids, BLOCK_SIZE)

    x = torch.tensor([example_x for example_x, _ in examples], dtype=torch.long)
    y = torch.tensor([example_y for _, example_y in examples], dtype=torch.long)

    model = BigramLanguageModel(vocab_size=tokenizer.vocab_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    for step in range(MAX_ITERS):
        logits, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % 20 == 0 or step == MAX_ITERS - 1:
            print(f"step {step}: loss {loss.item():.4f}")

    start = torch.tensor([[ids[0]]], dtype=torch.long)
    generated_ids = model.generate(start, max_new_tokens=120)[0].tolist()
    generated_text = tokenizer.decode(generated_ids)

    print("\n--- Texto gerado ---")
    print(generated_text)


if __name__ == "__main__":
    main()
