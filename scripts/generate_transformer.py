import torch

from src.models.transformer import MiniTransformerLanguageModel


CHECKPOINT_PATH = "checkpoints/transformer.pt"
PROMPT = "A inteligencia artificial"
MAX_NEW_TOKENS = 300
TEMPERATURE = 0.8
TOP_K = 10


def encode_prompt(prompt, stoi, unk_token):
    unk_id = stoi[unk_token]
    return [stoi.get(ch, unk_id) for ch in prompt]


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

    stoi = checkpoint["stoi"]
    itos = checkpoint["itos"]
    unk_token = checkpoint["unk_token"]

    prompt_ids = encode_prompt(PROMPT, stoi, unk_token)
    context = torch.tensor([prompt_ids], dtype=torch.long)

    with torch.no_grad():
        generated_ids = model.generate(
            context,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_k=TOP_K,
        )[0].tolist()

    generated_text = "".join(itos[i] for i in generated_ids)

    print(
        f"--- Texto gerado pelo transformer | "
        f"prompt={PROMPT!r} | "
        f"temperature={TEMPERATURE} | "
        f"top_k={TOP_K} ---"
    )
    print(generated_text)


if __name__ == "__main__":
    main()
