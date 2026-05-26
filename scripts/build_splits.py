import argparse
import random
from pathlib import Path


PROCESSED_DIR = Path("data/processed/wikipedia")
SPLITS_DIR = Path("data/splits")
MANIFEST_PATH = Path("data/splits/MANIFEST.md")
DEFAULT_SEED = 42
DEFAULT_TRAIN_RATIO = 0.8
DEFAULT_VAL_RATIO = 0.1
DEFAULT_TEST_RATIO = 0.1


def read_processed_texts(processed_dir):
    processed_dir = Path(processed_dir)

    if not processed_dir.exists():
        raise FileNotFoundError(f"Diretório processado não encontrado: {processed_dir}")

    files = sorted(processed_dir.glob("*.txt"))
    texts = []

    for file_path in files:
        text = file_path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        texts.append((file_path, text))

    return texts


def normalize_path(path):
    return str(path).replace("\\", "/")


def validate_ratios(train_ratio, val_ratio, test_ratio):
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-8:
        raise ValueError("As proporções de train, val e test precisam somar 1.0.")

    if train_ratio <= 0 or val_ratio <= 0 or test_ratio <= 0:
        raise ValueError("As proporções precisam ser maiores que zero.")


def split_texts(texts, train_ratio, val_ratio, test_ratio, seed):
    validate_ratios(train_ratio, val_ratio, test_ratio)

    texts = list(texts)
    rng = random.Random(seed)
    rng.shuffle(texts)

    total_files = len(texts)
    train_end = int(total_files * train_ratio)
    val_end = train_end + int(total_files * val_ratio)

    if total_files >= 3:
        train_end = max(1, train_end)
        val_end = max(train_end + 1, val_end)
        val_end = min(val_end, total_files - 1)

    train_items = texts[:train_end]
    val_items = texts[train_end:val_end]
    test_items = texts[val_end:]

    return train_items, val_items, test_items


def join_texts(items):
    return "\n\n".join(text for _, text in items).strip() + "\n"


def write_split(path, items):
    path.write_text(join_texts(items), encoding="utf-8")


def write_manifest(
    manifest_path,
    train_items,
    val_items,
    test_items,
    seed,
    train_ratio,
    val_ratio,
    test_ratio,
):
    def count_chars(items):
        return sum(len(text) for _, text in items)

    lines = [
        "# Manifesto dos splits",
        "",
        f"Seed: {seed}",
        f"Proporções: train={train_ratio}, val={val_ratio}, test={test_ratio}",
        "",
        "| split | arquivos | caracteres |",
        "|---|---:|---:|",
        f"| train | {len(train_items)} | {count_chars(train_items)} |",
        f"| val | {len(val_items)} | {count_chars(val_items)} |",
        f"| test | {len(test_items)} | {count_chars(test_items)} |",
        "",
        "## Arquivos por split",
        "",
        "### train",
        "",
    ]

    for file_path, _ in train_items:
        lines.append(f"- {normalize_path(file_path)}")

    lines.extend(["", "### val", ""])
    for file_path, _ in val_items:
        lines.append(f"- {normalize_path(file_path)}")

    lines.extend(["", "### test", ""])
    for file_path, _ in test_items:
        lines.append(f"- {normalize_path(file_path)}")

    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_splits(
    processed_dir=PROCESSED_DIR,
    splits_dir=SPLITS_DIR,
    manifest_path=MANIFEST_PATH,
    train_ratio=DEFAULT_TRAIN_RATIO,
    val_ratio=DEFAULT_VAL_RATIO,
    test_ratio=DEFAULT_TEST_RATIO,
    seed=DEFAULT_SEED,
):
    texts = read_processed_texts(processed_dir)

    if len(texts) < 3:
        raise ValueError("É necessário ter pelo menos 3 arquivos processados para criar splits.")

    train_items, val_items, test_items = split_texts(
        texts,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    splits_dir = Path(splits_dir)
    manifest_path = Path(manifest_path)
    splits_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    write_split(splits_dir / "train.txt", train_items)
    write_split(splits_dir / "val.txt", val_items)
    write_split(splits_dir / "test.txt", test_items)
    write_manifest(
        manifest_path,
        train_items,
        val_items,
        test_items,
        seed,
        train_ratio,
        val_ratio,
        test_ratio,
    )

    print(f"train: {len(train_items)} arquivos")
    print(f"val: {len(val_items)} arquivos")
    print(f"test: {len(test_items)} arquivos")
    print(f"Splits salvos em: {splits_dir}")
    print(f"Manifesto salvo em: {manifest_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Gera train/val/test a partir dos textos processados."
    )
    parser.add_argument("--processed-dir", default=str(PROCESSED_DIR))
    parser.add_argument("--splits-dir", default=str(SPLITS_DIR))
    parser.add_argument("--manifest-path", default=str(MANIFEST_PATH))
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_TRAIN_RATIO)
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO)
    parser.add_argument("--test-ratio", type=float, default=DEFAULT_TEST_RATIO)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    build_splits(
        processed_dir=args.processed_dir,
        splits_dir=args.splits_dir,
        manifest_path=args.manifest_path,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
