import argparse
from collections import Counter

from src.data_loader import read_text_file
from src.tokenizer.bpe_tokenizer import BPETokenizer


TOKENIZER_PATH = "artifacts/tokenizers/bpe.json"
TRAIN_PATH = "data/splits/train.txt"


def token_kind(token):
    if token == "<unk>":
        return "special"
    if token.isspace():
        return "space_only"
    if token.startswith(" ") or token.startswith("\n"):
        return "starts_with_space"
    if token.isalpha():
        return "letters_only"
    if token.isdigit():
        return "digits_only"
    if any(ch.isalpha() for ch in token) and any(ch.isdigit() for ch in token):
        return "letters_and_digits"
    if any(ch.isalpha() for ch in token):
        return "mixed_with_letters"
    return "symbols_or_punctuation"


def summarize_lengths(tokens):
    lengths = [len(token) for token in tokens]
    counter = Counter(lengths)
    return sorted(counter.items())


def print_token_list(title, items):
    print(f"\n{title}")
    for token, value in items:
        print(f"  {token!r}: {value}")


def main():
    parser = argparse.ArgumentParser(
        description="Analisa um tokenizer BPE salvo em JSON."
    )
    parser.add_argument("--tokenizer-path", default=TOKENIZER_PATH)
    parser.add_argument("--train-path", default=TRAIN_PATH)
    parser.add_argument("--top-n", type=int, default=30)
    args = parser.parse_args()

    tokenizer = BPETokenizer.load(args.tokenizer_path)
    tokens_by_id = sorted(tokenizer.stoi.items(), key=lambda item: item[1])
    tokens = [token for token, _ in tokens_by_id]

    print("=== Estrutura do BPE ===")
    print(f"Arquivo: {args.tokenizer_path}")
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Merges: {len(tokenizer.merges)}")
    print(f"UNK token: {tokenizer.unk_token!r}")

    print("\n=== Distribuicao por tamanho de token ===")
    for length, count in summarize_lengths(tokens):
        print(f"  tamanho {length}: {count}")

    kind_counts = Counter(token_kind(token) for token in tokens)
    print("\n=== Tipos de token no vocabulario ===")
    for kind, count in sorted(kind_counts.items()):
        print(f"  {kind}: {count}")

    longest = sorted(tokens, key=lambda token: len(token), reverse=True)[: args.top_n]
    print_token_list(
        f"Top {args.top_n} tokens mais longos",
        [(token, len(token)) for token in longest],
    )

    first_merges = tokenizer.merges[: args.top_n]
    print(f"\nPrimeiros {args.top_n} merges aprendidos")
    for pair in first_merges:
        print(f"  {pair!r} -> {''.join(pair)!r}")

    last_merges = tokenizer.merges[-args.top_n :]
    print(f"\nUltimos {args.top_n} merges aprendidos")
    for pair in last_merges:
        print(f"  {pair!r} -> {''.join(pair)!r}")

    text = read_text_file(args.train_path)
    encoded = tokenizer.encode(text)
    usage = Counter(encoded)

    id_to_token = tokenizer.itos
    most_used = [
        (id_to_token[token_id], count)
        for token_id, count in usage.most_common(args.top_n)
    ]
    print_token_list(f"Top {args.top_n} tokens mais usados no train.txt", most_used)

    single_char_ids = [token_id for token_id in encoded if len(id_to_token[token_id]) == 1]
    multi_char_ids = [token_id for token_id in encoded if len(id_to_token[token_id]) > 1]
    print("\n=== Uso no train.txt ===")
    print(f"Caracteres no treino: {len(text)}")
    print(f"Tokens codificados: {len(encoded)}")
    print(f"Compressao chars/tokens: {len(text) / len(encoded):.2f}x")
    print(f"Tokens de 1 caractere usados: {len(single_char_ids)}")
    print(f"Tokens multi-caractere usados: {len(multi_char_ids)}")
    print(f"Percentual multi-caractere: {len(multi_char_ids) / len(encoded) * 100:.2f}%")


if __name__ == "__main__":
    main()
