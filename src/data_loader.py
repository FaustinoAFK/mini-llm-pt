from pathlib import Path


def read_text_file(path):
    """Lê um arquivo de texto UTF-8 e retorna seu conteúdo como string."""
    file_path = Path(path)
    return file_path.read_text(encoding="utf-8")


def encode_text_file(path, tokenizer):
    """Lê um arquivo de texto e converte seu conteúdo em IDs usando o tokenizer."""
    text = read_text_file(path)
    return tokenizer.encode(text)
