import argparse
from pathlib import Path


MOJIBAKE_MARKERS = ("\u00c3", "\u00c2", "\ufffd")
PORTUGUESE_CHARS = set(
    "\u00e1\u00e0\u00e2\u00e3\u00e9\u00ea\u00ed\u00f3\u00f4\u00f5\u00fa\u00e7"
    "\u00c1\u00c0\u00c2\u00c3\u00c9\u00ca\u00cd\u00d3\u00d4\u00d5\u00da\u00c7"
)


def count_mojibake_markers(text):
    return sum(text.count(marker) for marker in MOJIBAKE_MARKERS)


def count_portuguese_chars(text):
    return sum(1 for char in text if char in PORTUGUESE_CHARS)


def repair_latin1_utf8_mojibake(text):
    try:
        return text.encode("latin-1").decode("utf-8")
    except UnicodeError:
        return text


def should_accept_repair(original, repaired):
    if repaired == original:
        return False

    original_markers = count_mojibake_markers(original)
    repaired_markers = count_mojibake_markers(repaired)
    if original_markers == 0:
        return False

    if repaired_markers >= original_markers:
        return False

    return count_portuguese_chars(repaired) >= count_portuguese_chars(original)


def fix_text(text):
    repaired = repair_latin1_utf8_mojibake(text)
    if should_accept_repair(text, repaired):
        return repaired
    return text


def iter_text_files(input_dir):
    input_dir = Path(input_dir)
    return sorted(input_dir.rglob("*.txt"))


def preview_change(original, repaired, max_len=120):
    original_lines = [line.strip() for line in original.splitlines() if line.strip()]
    repaired_lines = [line.strip() for line in repaired.splitlines() if line.strip()]

    if not original_lines or not repaired_lines:
        return None

    return (
        original_lines[0][:max_len],
        repaired_lines[0][:max_len],
    )


def fix_files(input_dir, apply=False, limit_preview=10):
    changed = []

    for path in iter_text_files(input_dir):
        original = path.read_text(encoding="utf-8")
        repaired = fix_text(original)

        if repaired == original:
            continue

        changed.append((path, original, repaired))
        if apply:
            path.write_text(repaired, encoding="utf-8")

    print(f"Arquivos analisados: {len(iter_text_files(input_dir))}")
    print(f"Arquivos com correcao: {len(changed)}")
    print(f"Modo: {'apply' if apply else 'dry-run'}")

    for path, original, repaired in changed[:limit_preview]:
        print(f"\nArquivo: {path}")
        preview = preview_change(original, repaired)
        if preview is None:
            continue
        before, after = preview
        print(f"Antes: {before}")
        print(f"Depois: {after}")

    if changed and not apply:
        print("\nNenhum arquivo foi alterado. Use --apply para gravar as correcoes.")

    return changed


def main():
    parser = argparse.ArgumentParser(
        description="Corrige mojibake UTF-8 lido como latin-1 em arquivos de texto."
    )
    parser.add_argument("--input-dir", default="data/processed/wikipedia")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--limit-preview", type=int, default=10)
    args = parser.parse_args()

    fix_files(
        input_dir=args.input_dir,
        apply=args.apply,
        limit_preview=args.limit_preview,
    )


if __name__ == "__main__":
    main()
