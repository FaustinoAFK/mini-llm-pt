import torch


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
