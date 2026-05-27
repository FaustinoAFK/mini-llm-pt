import argparse
import re
import unicodedata
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

NOISY_TERMS = {
    "displaystyle",
    "textstyle",
    "scriptstyle",
    "frac",
    "sqrt",
    "mathrm",
    "mathit",
    "mathbf",
    "operatorname",
    "left",
    "right",
    "begin",
    "end",
    "align",
    "matrix",
    "em inglês",
    "em ingles",
}

ALLOWED_CHARS_PATTERN = re.compile(
    r"[^0-9A-Za-zÀ-ÖØ-öø-ÿÇç\s\.,;:!?¿¡'\"“”‘’«»\-\(\)\[\]/%ºª]"
)


def normalize_text(text):
    """Normaliza espaços e quebras de linha sem destruir parágrafos."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_section_heading(line):
    return bool(re.fullmatch(r"=+\s*[^=]+\s*=+", line.strip()))


def has_noisy_terms(line):
    lowered = line.lower()
    return any(term in lowered for term in NOISY_TERMS)


def has_too_much_symbol_noise(line):
    compact = re.sub(r"\s+", "", line)
    if len(compact) < 20:
        return False

    letters = sum(char.isalpha() for char in compact)
    return letters / len(compact) < 0.45


def has_too_many_numbers(line):
    tokens = line.split()
    if len(tokens) < 6:
        return False

    numeric_tokens = sum(token.strip(".,;:!?()[]/%").isdigit() for token in tokens)
    return numeric_tokens / len(tokens) > 0.35


def is_numbered_list_noise(line):
    stripped = line.strip()

    if re.fullmatch(r"(\d+\s*){2,}", stripped):
        return True

    if re.fullmatch(r"\(?\s*[A-Za-z0-9]\s*\)?", stripped):
        return True

    if re.fullmatch(r"[\(\)\[\]\d\s,.;:/%-]+", stripped) and len(stripped) <= 30:
        return True

    return False


def has_too_many_parentheses(line):
    if len(line) < 12:
        return False

    parens = sum(char in "()[]" for char in line)
    return parens / len(line) > 0.20


def has_formula_like_pattern(line):
    stripped = line.strip()

    if re.search(r"\([A-Za-z0-9]\)", stripped):
        return True

    if re.search(r"\b[xXyYnN]\s*[\+\-\*/=]\s*\d", stripped):
        return True

    if re.search(r"\d\s*[\+\-\*/=]\s*\d", stripped):
        return True

    return False


def should_drop_line(line):
    stripped = line.strip()
    lowered = stripped.lower()

    if not stripped:
        return False

    if is_section_heading(stripped):
        return True

    if lowered in SECTION_TITLES_TO_DROP:
        return True

    if stripped.startswith("Categoria:"):
        return True

    if stripped.startswith("Portal:"):
        return True

    if has_noisy_terms(stripped):
        return True

    if is_numbered_list_noise(stripped):
        return True

    if has_formula_like_pattern(stripped):
        return True

    if has_too_many_parentheses(stripped):
        return True

    if has_too_much_symbol_noise(stripped):
        return True

    if has_too_many_numbers(stripped):
        return True

    return False


def clean_line(line):
    line = ALLOWED_CHARS_PATTERN.sub(" ", line)
    line = re.sub(r"\s+", " ", line)
    return line.strip()


def clean_wikipedia_text(text):
    """Limpa um texto raw da Wikipedia para uso como dataset.

    A limpeza remove títulos de seção, linhas com muito ruído simbólico,
    marcações matemáticas, termos comuns de LaTeX/MediaWiki, notas de tradução
    e caracteres incomuns que prejudicam o tokenizer.
    """
    text = normalize_text(text)
    cleaned_lines = []

    for line in text.split("\n"):
        line = line.strip()

        if should_drop_line(line):
            continue

        line = clean_line(line)

        if not line:
            continue

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
        "Limpeza aplicada:",
        "",
        "- normalização Unicode NFKC;",
        "- remoção de títulos de seção no formato `== seção ==`;",
        "- remoção de linhas com excesso de símbolos;",
        "- remoção de linhas com termos LaTeX/MediaWiki como `displaystyle`, `frac` e `sqrt`;",
        "- remoção de notas como `(em inglês)`;",
        "- remoção de linhas dominadas por números;",
        "- remoção de linhas curtas com listas/fórmulas;",
        "- remoção de linhas com excesso de parênteses;",
        "- filtragem de caracteres incomuns para reduzir ruído no tokenizer;",
        "- normalização de espaços.",
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
