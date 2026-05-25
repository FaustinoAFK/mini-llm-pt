import torch
import torch.nn as nn
import torch.nn.functional as F


class BigramLanguageModel(nn.Module):
    """Modelo de linguagem Bigram.

    Este modelo prevê o próximo token olhando apenas para o token atual.
    Ele é simples, mas já contém o ciclo básico de uma LLM:
    embeddings, logits, loss e geração.
    """

    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        """Executa o modelo.

        idx: tensor com shape (batch, time)
        targets: tensor opcional com shape (batch, time)

        retorna:
        - logits com shape (batch, time, vocab_size)
        - loss, se targets for fornecido
        """
        logits = self.token_embedding_table(idx)

        loss = None
        if targets is not None:
            batch, time, channels = logits.shape
            logits_flat = logits.view(batch * time, channels)
            targets_flat = targets.view(batch * time)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        """Gera novos tokens a partir de um contexto inicial."""
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
