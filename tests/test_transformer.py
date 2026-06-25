import pytest
import torch
import torch.nn as nn

from src.models.transformer import FeedForward, MiniTransformerLanguageModel


def test_transformer_forward_without_targets():
    vocab_size = 10
    block_size = 4
    model = MiniTransformerLanguageModel(
        vocab_size=vocab_size,
        block_size=block_size,
        n_embd=16,
        n_head=4,
        n_layer=2,
    )
    idx = torch.tensor([[1, 2, 3, 4]])

    logits, loss = model(idx)

    assert logits.shape == (1, 4, vocab_size)
    assert loss is None


def test_transformer_forward_with_targets_returns_loss():
    vocab_size = 10
    block_size = 4
    model = MiniTransformerLanguageModel(
        vocab_size=vocab_size,
        block_size=block_size,
        n_embd=16,
        n_head=4,
        n_layer=2,
    )
    idx = torch.tensor([[1, 2, 3, 4]])
    targets = torch.tensor([[2, 3, 4, 5]])

    logits, loss = model(idx, targets)

    assert logits.shape == (1, 4, vocab_size)
    assert loss is not None
    assert loss.ndim == 0


def test_transformer_generate_adds_tokens():
    vocab_size = 10
    block_size = 4
    model = MiniTransformerLanguageModel(
        vocab_size=vocab_size,
        block_size=block_size,
        n_embd=16,
        n_head=4,
        n_layer=2,
    )
    idx = torch.tensor([[1]])

    generated = model.generate(idx, max_new_tokens=5)

    assert generated.shape == (1, 6)


def test_transformer_generate_accepts_temperature_and_top_k():
    vocab_size = 10
    block_size = 4
    model = MiniTransformerLanguageModel(
        vocab_size=vocab_size,
        block_size=block_size,
        n_embd=16,
        n_head=4,
        n_layer=2,
    )
    idx = torch.tensor([[1]])

    generated = model.generate(
        idx,
        max_new_tokens=5,
        temperature=0.8,
        top_k=3,
    )

    assert generated.shape == (1, 6)


def test_transformer_generate_accepts_top_p_and_repetition_controls():
    vocab_size = 10
    block_size = 4
    model = MiniTransformerLanguageModel(
        vocab_size=vocab_size,
        block_size=block_size,
        n_embd=16,
        n_head=4,
        n_layer=2,
    )
    idx = torch.tensor([[1, 2, 1]])

    generated = model.generate(
        idx,
        max_new_tokens=3,
        temperature=0.8,
        top_k=5,
        top_p=0.9,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
    )

    assert generated.shape == (1, 6)


def test_transformer_generate_rejects_invalid_temperature():
    model = MiniTransformerLanguageModel(
        vocab_size=10,
        block_size=4,
        n_embd=16,
        n_head=4,
        n_layer=2,
    )
    idx = torch.tensor([[1]])

    with pytest.raises(ValueError):
        model.generate(idx, max_new_tokens=1, temperature=0)


def test_transformer_generate_rejects_invalid_top_k():
    model = MiniTransformerLanguageModel(
        vocab_size=10,
        block_size=4,
        n_embd=16,
        n_head=4,
        n_layer=2,
    )
    idx = torch.tensor([[1]])

    with pytest.raises(ValueError):
        model.generate(idx, max_new_tokens=1, top_k=0)


def test_transformer_generate_rejects_invalid_top_p():
    model = MiniTransformerLanguageModel(
        vocab_size=10,
        block_size=4,
        n_embd=16,
        n_head=4,
        n_layer=2,
    )
    idx = torch.tensor([[1]])

    with pytest.raises(ValueError):
        model.generate(idx, max_new_tokens=1, top_p=1.1)


def test_transformer_generate_rejects_invalid_repetition_penalty():
    model = MiniTransformerLanguageModel(
        vocab_size=10,
        block_size=4,
        n_embd=16,
        n_head=4,
        n_layer=2,
    )
    idx = torch.tensor([[1]])

    with pytest.raises(ValueError):
        model.generate(idx, max_new_tokens=1, repetition_penalty=0.9)


def test_transformer_generate_rejects_invalid_no_repeat_ngram_size():
    model = MiniTransformerLanguageModel(
        vocab_size=10,
        block_size=4,
        n_embd=16,
        n_head=4,
        n_layer=2,
    )
    idx = torch.tensor([[1]])

    with pytest.raises(ValueError):
        model.generate(idx, max_new_tokens=1, no_repeat_ngram_size=-1)


def test_apply_no_repeat_ngram_bans_repeated_continuation():
    logits = torch.zeros((1, 5))
    idx = torch.tensor([[1, 2, 3, 1, 2]])

    filtered = MiniTransformerLanguageModel._apply_no_repeat_ngram(
        logits,
        idx,
        no_repeat_ngram_size=3,
    )

    assert filtered[0, 3] == float("-inf")
    assert filtered[0, 4] == 0


def test_apply_repetition_penalty_penalizes_seen_tokens():
    logits = torch.tensor([[2.0, -2.0, 1.0]])
    idx = torch.tensor([[0, 1]])

    filtered = MiniTransformerLanguageModel._apply_repetition_penalty(
        logits,
        idx,
        repetition_penalty=2.0,
    )

    assert filtered[0, 0] == pytest.approx(1.0)
    assert filtered[0, 1] == pytest.approx(-4.0)
    assert filtered[0, 2] == pytest.approx(1.0)


def test_transformer_rejects_sequence_longer_than_block_size():
    vocab_size = 10
    block_size = 4
    model = MiniTransformerLanguageModel(
        vocab_size=vocab_size,
        block_size=block_size,
        n_embd=16,
        n_head=4,
        n_layer=2,
    )
    idx = torch.tensor([[1, 2, 3, 4, 5]])

    with pytest.raises(ValueError):
        model(idx)


def test_feedforward_usa_gelu():
    ffwd = FeedForward(n_embd=16)

    tem_gelu = any(isinstance(m, nn.GELU) for m in ffwd.net)

    assert tem_gelu, "FeedForward deve usar GELU como ativação"


def test_transformer_generate_rejects_top_k_antes_do_loop():
    model = MiniTransformerLanguageModel(
        vocab_size=10,
        block_size=4,
        n_embd=16,
        n_head=4,
        n_layer=2,
    )
    idx = torch.tensor([[1]])

    with pytest.raises(ValueError, match="top_k"):
        model.generate(idx, max_new_tokens=5, top_k=-1)
