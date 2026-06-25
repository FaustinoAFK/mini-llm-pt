import pytest

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


def test_bpe_byte_level_tokenizer_encodes_unseen_characters():
    tokenizer = BPETokenizer()
    tokenizer.train("porta", num_merges=10)

    ids = tokenizer.encode("portaria")

    assert ids
    assert tokenizer.decode(ids) == "portaria"


def test_bpe_save_and_load_roundtrip(tmp_path):
    text = "aprendizado de maquina"
    path = tmp_path / "bpe.json"
    tokenizer = BPETokenizer()
    tokenizer.train(text, num_merges=15)
    tokenizer.save(path)

    loaded = BPETokenizer.load(path)

    assert loaded.encode(text) == tokenizer.encode(text)
    assert loaded.decode(loaded.encode(text)) == text


def test_bpe_encode_supports_bos_and_eos_tokens():
    tokenizer = BPETokenizer()
    tokenizer.train("texto de treino", num_merges=10)

    ids = tokenizer.encode("texto", add_bos=True, add_eos=True)

    assert ids[0] == tokenizer.stoi[tokenizer.bos_token]
    assert ids[-1] == tokenizer.stoi[tokenizer.eos_token]


def test_bpe_decode_for_display_formats_paragraph_token():
    tokenizer = BPETokenizer()
    tokenizer.train("linha um <par> linha dois", num_merges=10)

    ids = tokenizer.encode(
        "linha um <par> linha dois",
        add_bos=True,
        add_eos=True,
    )
    rendered = tokenizer.decode_for_display(ids)

    assert tokenizer.bos_token not in rendered
    assert tokenizer.eos_token not in rendered
    assert "\n\n" in rendered


def test_bpe_train_accepts_vocab_size_and_min_frequency():
    tokenizer = BPETokenizer()
    tokenizer.train(
        "dados dados dados para tokenizer",
        num_merges=5,
        vocab_size=300,
        min_frequency=1,
    )

    assert tokenizer.vocab_size <= 300
    assert tokenizer.stoi[tokenizer.paragraph_token] >= 0


def test_bpe_rejects_empty_training_text():
    tokenizer = BPETokenizer()

    with pytest.raises(ValueError, match="vazio"):
        tokenizer.train("")


def test_bpe_rejects_invalid_vocab_size():
    tokenizer = BPETokenizer()

    with pytest.raises(ValueError, match="special tokens"):
        tokenizer.train("texto valido", vocab_size=2)


def test_bpe_rejects_invalid_min_frequency():
    tokenizer = BPETokenizer()

    with pytest.raises(ValueError, match="min_frequency"):
        tokenizer.train("texto valido", min_frequency=0)
