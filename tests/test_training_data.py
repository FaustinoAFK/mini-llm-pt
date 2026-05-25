import pytest

from src.training_data import (
    create_training_example,
    create_training_examples,
    train_val_split_ids,
)


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


def test_create_training_examples_with_sliding_window():
    ids = [1, 2, 3, 4, 5]
    block_size = 3

    examples = create_training_examples(ids, block_size)

    assert examples == [
        ([1, 2, 3], [2, 3, 4]),
        ([2, 3, 4], [3, 4, 5]),
    ]


def test_create_training_examples_requires_enough_ids():
    ids = [1, 2, 3]
    block_size = 3

    with pytest.raises(ValueError):
        create_training_examples(ids, block_size)


def test_train_val_split_ids():
    ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    train_ids, val_ids = train_val_split_ids(ids, train_ratio=0.8)

    assert train_ids == [1, 2, 3, 4, 5, 6, 7, 8]
    assert val_ids == [9, 10]


def test_train_val_split_ids_rejects_invalid_ratio():
    ids = [1, 2, 3]

    with pytest.raises(ValueError):
        train_val_split_ids(ids, train_ratio=1.0)