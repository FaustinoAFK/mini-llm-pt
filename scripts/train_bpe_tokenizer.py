import argparse
from pathlib import Path

from src.data_loader import read_text_file
from src.tokenizer.bpe_tokenizer import BPETokenizer


TRAIN_PATH = "data/splits/train.txt"
TOKENIZER_PATH = "artifacts/tokenizers/bpe.json"
NUM_MERGES = 500


def main():
    parser = argparse.ArgumentParser(
        description="Treina um tokenizer BPE simples a partir do train.txt."
    )
    parser.add_argument("--train-path", default=TRAIN_PATH)
    parser.add_argument("--tokenizer-path", default=TOKENIZER_PATH)
    parser.add_argument("--num-merges", type=int, default=NUM_MERGES)
    args = parser.parse_args()

    text = read_text_file(args.train_path)

    tokenizer = BPETokenizer()
    tokenizer.train(text, num_merges=args.num_merges)
    tokenizer.save(args.tokenizer_path)

    encoded = tokenizer.encode(text)
    compression = len(text) / len(encoded)

    print(f"Tokenizer salvo em: {args.tokenizer_path}")
    print(f"Caracteres no treino: {len(text)}")
    print(f"Tokens BPE no treino: {len(encoded)}")
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Merges aprendidos: {len(tokenizer.merges)}")
    print(f"Compressao chars/tokens: {compression:.2f}x")

    preview_path = Path(args.tokenizer_path).with_suffix(".preview.txt")
    preview_tokens = sorted(tokenizer.stoi.items(), key=lambda item: item[1])[:100]
    preview_path.write_text(
        "\n".join(f"{token_id}: {token!r}" for token, token_id in preview_tokens) + "\n",
        encoding="utf-8",
    )
    print(f"Preview do vocab salvo em: {preview_path}")


if __name__ == "__main__":
    main()
