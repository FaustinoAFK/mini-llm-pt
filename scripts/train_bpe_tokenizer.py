import argparse
from pathlib import Path

from src.data_loader import read_text_file
from src.tokenizer.bpe_tokenizer import BPETokenizer


TRAIN_PATH = "data/splits/train.txt"
TOKENIZER_PATH = "artifacts/tokenizers/bpe.json"
NUM_MERGES = 1000
VOCAB_SIZE = 2000
MIN_FREQUENCY = 2


def main():
    parser = argparse.ArgumentParser(
        description="Treina um tokenizer BPE simples a partir do train.txt."
    )
    parser.add_argument("--train-path", default=TRAIN_PATH)
    parser.add_argument("--tokenizer-path", default=TOKENIZER_PATH)
    parser.add_argument("--num-merges", type=int, default=NUM_MERGES)
    parser.add_argument("--vocab-size", type=int, default=VOCAB_SIZE)
    parser.add_argument("--min-frequency", type=int, default=MIN_FREQUENCY)
    args = parser.parse_args()

    text = read_text_file(args.train_path)

    tokenizer = BPETokenizer()
    tokenizer.train(
        text,
        num_merges=args.num_merges,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
    )
    tokenizer.save(args.tokenizer_path)

    encoded = tokenizer.encode(text, add_bos=True, add_eos=True)
    compression = len(text) / len(encoded)

    print(f"Tokenizer salvo em: {args.tokenizer_path}")
    print(f"Caracteres no treino: {len(text)}")
    print(f"Tokens BPE no treino: {len(encoded)}")
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Merges aprendidos: {len(tokenizer.merges)}")
    print(f"Special tokens: {', '.join(tokenizer.special_tokens)}")
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
