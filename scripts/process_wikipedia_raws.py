import argparse
import re
from pathlib import Path


RAW_DIR = Path("data/raw/wikipedia")
PROCESSED_DIR = Path("data/processed/wikipedia")
MANIFEST_PATH = Path("data/processed/wikipedia/MANIFEST.md")


SECTION_TITLES_TO_DROP = {
    "ver também",
    "referências",
    "referencias",
    "bibliografia",
    "ligações externas",
    "ligacoes externas",
    "notas",
    "fontes",
}


def normalize_text(text):
    """Normaliza espaços e quebras de linha sem destruir parágrafos."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def should_drop_line(line):
    stripped = line.strip()
    lowered = stripped.lower()

    if not stripped:
        return False

    if lowered in SECTION_TITLES_TO_DROP:
        return True

    if stripped.startswith("Categoria:"):
        return True

    if stripped.startswith("Portal:"):
        return True

    return False


def clean_wikipedia_text(text):
    """Limpa um texto raw da Wikipedia para uso como dataset.

    O objetivo não é fazer uma limpeza agressiva, mas remover ruídos óbvios
    e normalizar o texto para treino.
    """
    text = normalize_text(text)
    cleaned_lines = []

    for line in text.split("\n"):
        line = line.strip()

        if should_drop_line(line):
            continue

        cleaned_lines.append(line)

    cleaned_text = "\n".join(cleaned_lines)
    cleaned_text = normalize_text(cleaned_text)
    return cleaned_text


def process_file(raw_path, processed_dir):
    raw_text = raw_path.read_text(encoding="utf-8")
    cleaned_text = clean_wikipedia_text(raw_text)

    processed_path = processed_dir / raw_path.name
    processed_path.write_text(cleaned_text + "\n", encoding="utf-8")

    return {
        "raw_file": str(raw_path).replace("\\", "/"),
        "processed_file": str(processed_path).replace("\\", "/"),
        "raw_chars": len(raw_text),
        "processed_chars": len(cleaned_text),
    }


def write_manifest(entries, manifest_path):
    lines = [
        "# Manifesto de textos processados da Wikipedia",
        "",
        "Este manifesto registra os arquivos gerados a partir de `data/raw/wikipedia/`.",
        "",
        "| raw | processed | raw chars | processed chars |",
        "|---|---|---:|---:|",
    ]

    for entry in entries:
        lines.append(
            f"| {entry['raw_file']} | {entry['processed_file']} | "
            f"{entry['raw_chars']} | {entry['processed_chars']} |"
        )

    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def process_all(raw_dir=RAW_DIR, processed_dir=PROCESSED_DIR, manifest_path=MANIFEST_PATH):
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    manifest_path = Path(manifest_path)

    if not raw_dir.exists():
        raise FileNotFoundError(f"Diretório raw não encontrado: {raw_dir}")

    processed_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    entries = []

    for raw_path in sorted(raw_dir.glob("*.txt")):
        entry = process_file(raw_path, processed_dir)
        entries.append(entry)
        print(
            f"Processado: {entry['raw_file']} -> {entry['processed_file']} "
            f"({entry['raw_chars']} -> {entry['processed_chars']} chars)"
        )

    write_manifest(entries, manifest_path)
    print(f"\nTotal de arquivos processados: {len(entries)}")
    print(f"Manifesto salvo em: {manifest_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Limpa arquivos raw da Wikipedia e salva em data/processed/wikipedia."
    )
    parser.add_argument("--raw-dir", default=str(RAW_DIR))
    parser.add_argument("--processed-dir", default=str(PROCESSED_DIR))
    parser.add_argument("--manifest-path", default=str(MANIFEST_PATH))
    args = parser.parse_args()

    process_all(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        manifest_path=args.manifest_path,
    )


if __name__ == "__main__":
    main()
