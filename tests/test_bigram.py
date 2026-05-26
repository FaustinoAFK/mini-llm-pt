import torch

from src.models.bigram import BigramLanguageModel


def test_bigram_forward_without_targets():
    vocab_size = 5
    model = BigramLanguageModel(vocab_size=vocab_size)
    idx = torch.tensor([[0, 1, 2]])

    logits, loss = model(idx)

    assert logits.shape == (1, 3, vocab_size)
    assert loss is None


def test_bigram_forward_with_targets_returns_loss():
    vocab_size = 5
    model = BigramLanguageModel(vocab_size=vocab_size)
    idx = torch.tensor([[0, 1, 2]])
    targets = torch.tensor([[1, 2, 3]])

    logits, loss = model(idx, targets)

    assert logits.shape == (1, 3, vocab_size)
    assert loss is not None
    assert loss.ndim == 0


def test_bigram_generate_adds_tokens():
    vocab_size = 5
    model = BigramLanguageModel(vocab_size=vocab_size)
    idx = torch.tensor([[0]])

    generated = model.generate(idx, max_new_tokens=4)

    assert generated.shape == (1, 5)
