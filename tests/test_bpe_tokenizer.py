from src.tokenizer.bpe_tokenizer import BPETokenizer


def test_bpe_train_builds_vocab_and_merges():
    tokenizer = BPETokenizer()

    tokenizer.train("abababab", num_merges=2)

    assert tokenizer.vocab_size > 1
    assert len(tokenizer.merges) > 0


def test_bpe_encode_decode_roundtrip_for_training_text():
    text = "banana bandana"
    tokenizer = BPETokenizer()
    tokenizer.train(text, num_merges=20)

    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)

    assert decoded == text


def test_bpe_unknown_character_uses_unk_token():
    tokenizer = BPETokenizer()
    tokenizer.train("porta", num_merges=10)

    ids = tokenizer.encode("portaria")

    assert tokenizer.stoi[tokenizer.unk_token] in ids


def test_bpe_save_and_load_roundtrip(tmp_path):
    text = "aprendizado de maquina"
    path = tmp_path / "bpe.json"
    tokenizer = BPETokenizer()
    tokenizer.train(text, num_merges=15)
    tokenizer.save(path)

    loaded = BPETokenizer.load(path)

    assert loaded.encode(text) == tokenizer.encode(text)
    assert loaded.decode(loaded.encode(text)) == text


def test_bpe_rejects_empty_training_text():
    tokenizer = BPETokenizer()

    try:
        tokenizer.train("")
    except ValueError as error:
        assert "vazio" in str(error)
    else:
        raise AssertionError("Era esperado ValueError para texto vazio.")
