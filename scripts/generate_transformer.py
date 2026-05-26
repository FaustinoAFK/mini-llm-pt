import torch

from src.models.transformer import MiniTransformerLanguageModel


CHECKPOINT_PATH = "checkpoints/transformer.pt"
MAX_NEW_TOKENS = 300


def main():
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")

    model = MiniTransformerLanguageModel(
        vocab_size=checkpoint["vocab_size"],
        block_size=checkpoint["block_size"],
        n_embd=checkpoint["n_embd"],
        n_head=checkpoint["n_head"],
        n_layer=checkpoint["n_layer"],
        dropout=checkpoint["dropout"],
    )
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
