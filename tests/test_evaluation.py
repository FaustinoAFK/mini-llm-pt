import torch

from src.evaluation import estimate_loss
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
