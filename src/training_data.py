def create_training_example(ids, block_size):
    """Cria um único exemplo de treino para previsão do próximo token."""

    if len(ids) < block_size + 1:
        raise ValueError(
            "A sequência de IDs precisa ter pelo menos block_size + 1 elementos."
        )

    x = ids[:block_size]
    y = ids[1:block_size + 1]

    return x, y


def create_training_examples(ids, block_size):
    """Cria múltiplos exemplos de treino usando uma janela deslizante.

    Exemplo:
    ids = [1, 2, 3, 4, 5]
    block_size = 3

    exemplos:
    x = [1, 2, 3], y = [2, 3, 4]
    x = [2, 3, 4], y = [3, 4, 5]
    """
    if len(ids) < block_size + 1:
        raise ValueError(
            "A sequência de IDs precisa ter pelo menos block_size + 1 elementos."
        )

    examples = []

    for start in range(len(ids) - block_size):
        window = ids[start:start + block_size + 1]
        x = window[:block_size]
        y = window[1:block_size + 1]
        examples.append((x, y))

    return examples


def train_val_split_ids(ids, train_ratio=0.9):
    """Divide uma sequência de IDs em treino e validação."""
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio precisa estar entre 0 e 1.")

    split_index = int(len(ids) * train_ratio)

    train_ids = ids[:split_index]
    val_ids = ids[split_index:]

    return train_ids, val_ids
