import torch

from src.batching import get_batch


def estimate_loss(model, x, y):
    """Calcula a loss de um modelo sem atualizar pesos.

    Esta função coloca o modelo em modo de avaliação, desliga o cálculo
    de gradientes e retorna a loss como float Python.
    """
    was_training = model.training
    model.eval()

    with torch.no_grad():
        _, loss = model(x, y)

    if was_training:
        model.train()

    return loss.item()


def estimate_loss_over_batches(model, x, y, batch_size, num_batches):
    """Estima a loss média usando vários mini-batches aleatórios.

    Isso evita calcular a loss no dataset inteiro de uma vez, o que pode
    consumir muita memória quando o dataset fica grande.
    """
    if num_batches <= 0:
        raise ValueError("num_batches precisa ser maior que zero.")

    was_training = model.training
    model.eval()

    losses = []
    with torch.no_grad():
        for _ in range(num_batches):
            batch_x, batch_y = get_batch(x, y, batch_size)
            _, loss = model(batch_x, batch_y)
            losses.append(loss.item())

    if was_training:
        model.train()

    return torch.tensor(losses).mean().item()
