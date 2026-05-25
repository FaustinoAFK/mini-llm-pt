from src.data_loader import encode_text_file, read_text_file
from src.tokenizer.char_tokenizer import CharTokenizer


def test_read_text_file(tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("porta", encoding="utf-8")

    text = read_text_file(file_path)

    assert text == "porta"


def test_encode_text_file(tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("porta", encoding="utf-8")

    tokenizer = CharTokenizer("porta")
    ids = encode_text_file(file_path, tokenizer)

    assert ids == tokenizer.encode("porta")
