import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttentionHead(nn.Module):
    """Uma cabeça de atenção causal.

    Causal significa que cada posição só pode olhar para posições anteriores
    e para ela mesma, nunca para tokens futuros.
    """

    def __init__(self, n_embd, head_size, block_size, dropout=0.0):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        batch_size, time_steps, channels = x.shape

        key = self.key(x)
        query = self.query(x)

        weights = query @ key.transpose(-2, -1) * key.shape[-1] ** -0.5
        weights = weights.masked_fill(
            self.tril[:time_steps, :time_steps] == 0,
            float("-inf"),
        )
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)

        value = self.value(x)
        out = weights @ value
        return out


class MultiHeadCausalSelfAttention(nn.Module):
    """Várias cabeças de atenção causal em paralelo."""

    def __init__(self, n_embd, n_head, block_size, dropout=0.0):
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError("n_embd precisa ser divisível por n_head.")

        head_size = n_embd // n_head
        self.heads = nn.ModuleList(
            [
                CausalSelfAttentionHead(n_embd, head_size, block_size, dropout)
                for _ in range(n_head)
            ]
        )
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    """Rede simples aplicada independentemente em cada posição."""

    def __init__(self, n_embd, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),  # GELU em vez de ReLU, como no GPT-2 e modelos modernos
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """Bloco Transformer decoder-only.

    Contém:
    - LayerNorm
    - atenção causal multi-head
    - conexão residual
    - feed-forward
    - conexão residual
    """

    def __init__(self, n_embd, n_head, block_size, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadCausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffwd = FeedForward(n_embd, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class MiniTransformerLanguageModel(nn.Module):
    """Mini Transformer decoder-only para modelagem de linguagem.

    Ele recebe IDs com shape (B, T) e retorna logits com shape
    (B, T, vocab_size), onde cada posição tenta prever o próximo token.
    """

    def __init__(
        self,
        vocab_size,
        block_size,
        n_embd=64,
        n_head=4,
        n_layer=2,
        dropout=0.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[
                TransformerBlock(n_embd, n_head, block_size, dropout)
                for _ in range(n_layer)
            ]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        batch_size, time_steps = idx.shape
        if time_steps > self.block_size:
            raise ValueError("A sequência é maior que block_size.")

        token_embeddings = self.token_embedding_table(idx)
        position_ids = torch.arange(time_steps, device=idx.device)
        position_embeddings = self.position_embedding_table(position_ids)

        x = token_embeddings + position_embeddings
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            batch_size, time_steps, channels = logits.shape
            logits_flat = logits.view(batch_size * time_steps, channels)
            targets_flat = targets.view(batch_size * time_steps)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    def generate(
        self,
        idx,
        max_new_tokens,
        temperature=1.0,
        top_k=None,
        top_p=None,
        repetition_penalty=1.0,
        no_repeat_ngram_size=0,
    ):
        if temperature <= 0:
            raise ValueError("temperature precisa ser maior que zero.")

        if top_k is not None and top_k <= 0:
            raise ValueError("top_k precisa ser maior que zero.")

        if top_p is not None and not 0 < top_p <= 1:
            raise ValueError("top_p precisa estar entre 0 e 1.")

        if repetition_penalty < 1:
            raise ValueError("repetition_penalty precisa ser maior ou igual a 1.")

        if no_repeat_ngram_size < 0:
            raise ValueError("no_repeat_ngram_size precisa ser maior ou igual a zero.")

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            logits = logits / temperature
            logits = self._apply_repetition_penalty(logits, idx, repetition_penalty)
            logits = self._apply_no_repeat_ngram(
                logits,
                idx,
                no_repeat_ngram_size,
            )

            if top_k is not None:
                top_k = min(top_k, logits.size(-1))
                values, _ = torch.topk(logits, top_k)
                min_value = values[:, -1].unsqueeze(-1)
                logits = logits.masked_fill(logits < min_value, float("-inf"))

            if top_p is not None:
                logits = self._apply_top_p(logits, top_p)

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    @staticmethod
    def _apply_repetition_penalty(logits, idx, repetition_penalty):
        if repetition_penalty == 1:
            return logits

        logits = logits.clone()
        for batch_index in range(idx.size(0)):
            seen_tokens = set(idx[batch_index].tolist())
            for token_id in seen_tokens:
                if logits[batch_index, token_id] < 0:
                    logits[batch_index, token_id] *= repetition_penalty
                else:
                    logits[batch_index, token_id] /= repetition_penalty

        return logits

    @staticmethod
    def _apply_no_repeat_ngram(logits, idx, no_repeat_ngram_size):
        if no_repeat_ngram_size <= 0:
            return logits

        if idx.size(1) + 1 < no_repeat_ngram_size:
            return logits

        logits = logits.clone()
        prefix_size = no_repeat_ngram_size - 1

        for batch_index in range(idx.size(0)):
            tokens = idx[batch_index].tolist()
            prefix = tokens[-prefix_size:] if prefix_size > 0 else []
            banned_tokens = set()

            for start in range(len(tokens) - no_repeat_ngram_size + 1):
                ngram = tokens[start:start + no_repeat_ngram_size]
                if prefix_size == 0 or ngram[:-1] == prefix:
                    banned_tokens.add(ngram[-1])

            if len(banned_tokens) >= logits.size(-1):
                continue

            for token_id in banned_tokens:
                logits[batch_index, token_id] = float("-inf")

        return logits

    @staticmethod
    def _apply_top_p(logits, top_p):
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
        indices_to_remove.scatter_(
            dim=-1,
            index=sorted_indices,
            src=sorted_indices_to_remove,
        )
        return logits.masked_fill(indices_to_remove, float("-inf"))
