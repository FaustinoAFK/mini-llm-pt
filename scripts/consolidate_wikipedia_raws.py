import re
import shutil
from pathlib import Path


TARGET_DIR = Path("data/raw/wikipedia")
RETRY_DIRS = [
    Path("data/raw/wikipedia_retry"),
    Path("data/raw/wikipedia_retry_v2"),
]
MANIFEST_NAMES = {"MANIFEST.md", "manifest.md"}
FILENAME_PATTERN = re.compile(r"^(?P<number>\d{3})_(?P<name>.+\.txt)$")


def get_existing_max_index(target_dir):
    max_index = 0

    if not target_dir.exists():
        return max_index

    for file_path in target_dir.glob("*.txt"):
        match = FILENAME_PATTERN.match(file_path.name)
        if not match:
            continue

        number = int(match.group("number"))
        max_index = max(max_index, number)

    return max_index


def strip_existing_number(filename):
    match = FILENAME_PATTERN.match(filename)
    if match:
        return match.group("name")
    return filename


def iter_retry_files(retry_dirs):
    for retry_dir in retry_dirs:
        if not retry_dir.exists():
            print(f"Aviso: diretório não encontrado, ignorando: {retry_dir}")
            continue

        for file_path in sorted(retry_dir.glob("*.txt")):
            if file_path.name in MANIFEST_NAMES:
                continue
            yield retry_dir, file_path


def consolidate():
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    next_index = get_existing_max_index(TARGET_DIR) + 1
    moved_files = []

    for retry_dir, source_path in iter_retry_files(RETRY_DIRS):
        clean_name = strip_existing_number(source_path.name)
        target_name = f"{next_index:03d}_{clean_name}"
        target_path = TARGET_DIR / target_name

        while target_path.exists():
            next_index += 1
            target_name = f"{next_index:03d}_{clean_name}"
            target_path = TARGET_DIR / target_name

        shutil.move(str(source_path), str(target_path))
        moved_files.append((source_path, target_path))
        print(f"Movido: {source_path} -> {target_path}")
        next_index += 1

    print(f"\nTotal de arquivos movidos: {len(moved_files)}")

    for retry_dir in RETRY_DIRS:
        if not retry_dir.exists():
            continue

        remaining_files = [path for path in retry_dir.iterdir() if path.is_file()]
        if remaining_files:
            print(f"Aviso: mantendo {retry_dir}, pois ainda contém arquivos:")
            for path in remaining_files:
                print(f"  - {path.name}")
            continue

        try:
            retry_dir.rmdir()
            print(f"Diretório vazio removido: {retry_dir}")
        except OSError:
            print(f"Aviso: não foi possível remover diretório: {retry_dir}")


if __name__ == "__main__":
    consolidate()
