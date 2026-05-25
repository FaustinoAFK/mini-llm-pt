import pytest

from src.training_data import create_training_example


def test_create_training_example():
    ids = [1, 2, 3, 4, 5]
    block_size = 3

    x, y = create_training_example(ids, block_size)

    assert x == [1, 2, 3]
    assert y == [2, 3, 4]


def test_create_training_example_requires_enough_ids():
    ids = [1, 2, 3]
    block_size = 3

    with pytest.raises(ValueError):
        create_training_example(ids, block_size)