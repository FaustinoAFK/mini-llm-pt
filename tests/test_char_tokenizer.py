from src.tokenizer.char_tokenizer import CharTokenizer


def test_encode_decode_roundtrip():
    text = "porta"
    tokenizer = CharTokenizer(text)

    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)

    assert decoded == text