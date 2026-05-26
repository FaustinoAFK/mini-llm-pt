import pytest
import torch

from src.batching import get_batch


def test_get_batch_returns_expected_shapes():
    x = torch.tensor(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ]
    )
    y = torch.tensor(
        [
            [2, 3, 4],
            [5, 6, 7],
            [8, 9, 10],
        ]
    )

    batch_x, batch_y = get_batch(x, y, batch_size=2)

    assert batch_x.shape == (2, 3)
    assert batch_y.shape == (2, 3)


def test_get_batch_rejects_different_number_of_examples():
    x = torch.tensor([[1, 2, 3], [4, 5, 6]])
    y = torch.tensor([[2, 3, 4]])

    with pytest.raises(ValueError):
        get_batch(x, y, batch_size=1)


def test_get_batch_rejects_invalid_batch_size():
    x = torch.tensor([[1, 2, 3]])
    y = torch.tensor([[2, 3, 4]])

    with pytest.raises(ValueError):
        get_batch(x, y, batch_size=0)


def test_get_batch_rejects_batch_size_larger_than_dataset():
    x = torch.tensor([[1, 2, 3]])
    y = torch.tensor([[2, 3, 4]])

    with pytest.raises(ValueError):
        get_batch(x, y, batch_size=2)
