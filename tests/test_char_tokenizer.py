from src.tokenizer.char_tokenizer import CharTokenizer


def test_encode_decode_roundtrip():
    text = "porta"
    tokenizer = CharTokenizer(text)

    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)

    assert decoded == text


def test_unknown_character_uses_unk_token():
    tokenizer = CharTokenizer("porta")

    ids = tokenizer.encode("portaria")

    assert tokenizer.stoi[tokenizer.unk_token] in ids


def test_vocab_size_matches_vocabulary_length():
    tokenizer = CharTokenizer("porta")

    assert tokenizer.vocab_size == len(tokenizer.stoi)
