import argparse
import re
import unicodedata
from pathlib import Path


RAW_DIR = Path("data/raw/wikipedia")
PROCESSED_DIR = Path("data/processed/wikipedia")
MANIFEST_PATH = Path("data/processed/wikipedia/MANIFEST.md")
PARAGRAPH_TOKEN = "<par>"


SECTION_TITLES_TO_DROP = {
    "ver tambem",
    "referencias",
    "bibliografia",
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
    "em ingles",
}

ALLOWED_CHARS_PATTERN = re.compile(
    r"[^0-9A-Za-zÀ-ÖØ-öø-ÿÇç\s\.,;:!?¿¡'\"“”‘’«»\-\(\)\[\]/%ºª]"
)


def normalize_text(text):
    """Normaliza espacos e quebras de linha sem destruir paragrafos."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_flat_text(text):
    """Normaliza o texto final preservando o marcador de paragrafos."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def strip_accents(text):
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(char for char in normalized if not unicodedata.combining(char))


def is_section_heading(line):
    return bool(re.fullmatch(r"=+\s*[^=]+\s*=+", line.strip()))


def has_noisy_terms(line):
    lowered = strip_accents(line.lower())
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
    lowered = strip_accents(stripped.lower())

    if not stripped:
        return False

    if is_section_heading(stripped):
        return True

    if lowered in SECTION_TITLES_TO_DROP:
        return True

    if strip_accents(stripped).startswith("Categoria:"):
        return True

    if strip_accents(stripped).startswith("Portal:"):
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

    A limpeza remove titulos de secao, linhas com muito ruido simbolico,
    marcacoes matematicas, termos comuns de LaTeX/MediaWiki, notas de traducao
    e caracteres incomuns que prejudicam o tokenizer.

    Em vez de achatar todo o documento em um unico bloco, a saida final
    preserva fronteiras de paragrafo com o marcador `<par>`.
    """
    text = normalize_text(text)
    paragraphs = []
    current_paragraph = []

    for line in text.split("\n"):
        line = line.strip()

        if not line:
            if current_paragraph:
                paragraphs.append(" ".join(current_paragraph))
                current_paragraph = []
            continue

        if should_drop_line(line):
            continue

        line = clean_line(line)

        if not line:
            continue

        if should_drop_line(line):
            continue

        current_paragraph.append(line)

    if current_paragraph:
        paragraphs.append(" ".join(current_paragraph))

    cleaned_text = f" {PARAGRAPH_TOKEN} ".join(paragraphs)
    cleaned_text = normalize_flat_text(cleaned_text)
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
        "- normalizacao Unicode NFKC;",
        "- remocao de titulos de secao no formato `== secao ==`;",
        "- remocao de linhas com excesso de simbolos;",
        "- remocao de linhas com termos LaTeX/MediaWiki como `displaystyle`, `frac` e `sqrt`;",
        "- remocao de notas como `(em ingles)`;",
        "- remocao de linhas dominadas por numeros;",
        "- remocao de linhas curtas com listas/formulas;",
        "- remocao de linhas com excesso de parenteses;",
        "- filtragem de caracteres incomuns para reduzir ruido no tokenizer;",
        f"- preservacao de fronteiras de paragrafo com o marcador `{PARAGRAPH_TOKEN}`;",
        "- normalizacao de espacos.",
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
        raise FileNotFoundError(f"Diretorio raw nao encontrado: {raw_dir}")

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
