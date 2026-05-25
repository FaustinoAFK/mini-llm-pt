def create_training_example(ids, block_size):
    """Cria um único exemplo de treino para previsão do próximo token."""

    if len(ids) < block_size + 1:
        raise ValueError(
            "A sequência de IDs precisa ter pelo menos block_size + 1 elementos."
        )

    x = ids[:block_size]
    y = ids[1:block_size + 1]

    return x, y