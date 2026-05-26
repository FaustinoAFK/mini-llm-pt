import argparse
from collections import Counter
from pathlib import Path


SPLITS_DIR = Path("data/splits")
SUSPICIOUS_CHARS = [
    "{",
    "}",
    "=",
    "^",
    "|",
    "~",
    "<",
    ">",
]


def analyze_file(path):
    text = path.read_text(encoding="utf-8")
    counter = Counter(text)
    suspicious = {
        char: counter[char]
        for char in SUSPICIOUS_CHARS
        if counter[char]
    }

    return {
        "path": path,
        "chars": len(text),
        "unique_count": len(counter),
        "suspicious": suspicious,
    }


def check_splits(splits_dir=SPLITS_DIR):
    splits_dir = Path(splits_dir)
    files = [splits_dir / "train.txt", splits_dir / "val.txt", splits_dir / "test.txt"]

    has_problem = False

    for path in files:
        if not path.exists():
            print(f"ERRO: arquivo não encontrado: {path}")
            has_problem = True
            continue

        report = analyze_file(path)
        print(f"\nArquivo: {path}")
        print(f"Caracteres: {report['chars']}")
        print(f"Caracteres únicos: {report['unique_count']}")

        if report["suspicious"]:
            has_problem = True
            print("Caracteres suspeitos encontrados:")
            for char, count in report["suspicious"].items():
                print(f"  {repr(char)}: {count}")
        else:
            print("Caracteres suspeitos: nenhum")

    if has_problem:
        print("\nResultado: dataset ainda contém ruído suspeito.")
        return 1

    print("\nResultado: dataset parece limpo para o treino atual.")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Verifica caracteres suspeitos nos splits do dataset."
    )
    parser.add_argument("--splits-dir", default=str(SPLITS_DIR))
    args = parser.parse_args()

    raise SystemExit(check_splits(args.splits_dir))


if __name__ == "__main__":
    main()
