import torch


def get_batch(x, y, batch_size):
    """Retorna um mini-batch aleatório de exemplos de treino.

    Args:
        x: tensor de entradas com shape (num_examples, block_size).
        y: tensor de alvos com shape (num_examples, block_size).
        batch_size: quantidade de exemplos no lote.

    Returns:
        batch_x, batch_y: tensores com shape (batch_size, block_size).
    """
    if x.shape[0] != y.shape[0]:
        raise ValueError("x e y precisam ter a mesma quantidade de exemplos.")

    if batch_size <= 0:
        raise ValueError("batch_size precisa ser maior que zero.")

    if batch_size > x.shape[0]:
        raise ValueError("batch_size não pode ser maior que o número de exemplos.")

    indices = torch.randint(0, x.shape[0], (batch_size,))
    return x[indices], y[indices]
