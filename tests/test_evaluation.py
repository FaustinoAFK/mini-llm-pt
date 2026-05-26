import pytest
import torch

from src.evaluation import estimate_loss, estimate_loss_over_batches
from src.models.bigram import BigramLanguageModel


def test_estimate_loss_returns_float():
    model = BigramLanguageModel(vocab_size=5)
    x = torch.tensor([[0, 1, 2]])
    y = torch.tensor([[1, 2, 3]])

    loss = estimate_loss(model, x, y)

    assert isinstance(loss, float)


def test_estimate_loss_preserves_training_mode():
    model = BigramLanguageModel(vocab_size=5)
    model.train()
    x = torch.tensor([[0, 1, 2]])
    y = torch.tensor([[1, 2, 3]])

    estimate_loss(model, x, y)

    assert model.training is True


def test_estimate_loss_over_batches_returns_float():
    model = BigramLanguageModel(vocab_size=5)
    x = torch.tensor(
        [
            [0, 1, 2],
            [1, 2, 3],
            [2, 3, 4],
        ]
    )
    y = torch.tensor(
        [
            [1, 2, 3],
            [2, 3, 4],
            [3, 4, 0],
        ]
    )

    loss = estimate_loss_over_batches(
        model,
        x,
        y,
        batch_size=2,
        num_batches=3,
    )

    assert isinstance(loss, float)


def test_estimate_loss_over_batches_preserves_training_mode():
    model = BigramLanguageModel(vocab_size=5)
    model.train()
    x = torch.tensor([[0, 1, 2], [1, 2, 3]])
    y = torch.tensor([[1, 2, 3], [2, 3, 4]])

    estimate_loss_over_batches(model, x, y, batch_size=1, num_batches=2)

    assert model.training is True


def test_estimate_loss_over_batches_rejects_invalid_num_batches():
    model = BigramLanguageModel(vocab_size=5)
    x = torch.tensor([[0, 1, 2]])
    y = torch.tensor([[1, 2, 3]])

    with pytest.raises(ValueError):
        estimate_loss_over_batches(model, x, y, batch_size=1, num_batches=0)
