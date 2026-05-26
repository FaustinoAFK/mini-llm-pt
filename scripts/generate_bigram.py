import torch

from src.models.bigram import BigramLanguageModel


CHECKPOINT_PATH = "checkpoints/bigram.pt"
MAX_NEW_TOKENS = 200


def main():
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")

    model = BigramLanguageModel(vocab_size=checkpoint["vocab_size"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    itos = checkpoint["itos"]
    start_id = checkpoint["start_id"]

    context = torch.tensor([[start_id]], dtype=torch.long)

    with torch.no_grad():
        generated_ids = model.generate(context, max_new_tokens=MAX_NEW_TOKENS)[0].tolist()

    generated_text = "".join(itos[i] for i in generated_ids)

    print("--- Texto gerado ---")
    print(generated_text)


if __name__ == "__main__":
    main()
