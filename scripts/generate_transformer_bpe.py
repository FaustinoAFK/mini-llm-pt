import argparse

import torch

from src.models.transformer import MiniTransformerLanguageModel
from src.tokenizer.bpe_tokenizer import BPETokenizer


CHECKPOINT_PATH = "checkpoints/transformer_bpe.pt"
DEFAULT_PROMPT = "A inteligencia artificial "
DEFAULT_MAX_NEW_TOKENS = 80
DEFAULT_TEMPERATURE = 0.5
DEFAULT_TOP_K = 5


def generate_text(checkpoint_path, prompt, max_new_tokens, temperature, top_k):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    tokenizer = BPETokenizer.load(checkpoint["tokenizer_path"])

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

    prompt_ids = tokenizer.encode(prompt)
    context = torch.tensor([prompt_ids], dtype=torch.long)

    with torch.no_grad():
        generated_ids = model.generate(
            context,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )[0].tolist()

    return tokenizer.decode(generated_ids)


def main():
    parser = argparse.ArgumentParser(
        description="Gera texto usando o checkpoint BPE do Mini Transformer."
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--checkpoint-path", default=CHECKPOINT_PATH)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    args = parser.parse_args()

    generated_text = generate_text(
        checkpoint_path=args.checkpoint_path,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )

    print(
        f"--- Texto gerado pelo transformer BPE | "
        f"prompt={args.prompt!r} | "
        f"temperature={args.temperature} | "
        f"top_k={args.top_k} ---"
    )
    print(generated_text)


if __name__ == "__main__":
    main()
