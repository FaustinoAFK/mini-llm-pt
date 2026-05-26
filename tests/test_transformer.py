import pytest
import torch

from src.models.transformer import MiniTransformerLanguageModel


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
