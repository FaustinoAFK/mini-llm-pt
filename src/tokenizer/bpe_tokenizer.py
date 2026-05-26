import json
from collections import Counter
from pathlib import Path


class BPETokenizer:
    def __init__(self, unk_token="<unk>"):
        self.unk_token = unk_token
        self.merges = []
        self.stoi = {self.unk_token: 0}
        self.itos = {0: self.unk_token}

    @property
    def vocab_size(self):
        return len(self.stoi)

    def train(self, text, num_merges=100):
        if not text:
            raise ValueError("O texto de treino não pode ser vazio.")

        if num_merges < 0:
            raise ValueError("num_merges precisa ser maior ou igual a zero.")

        base_tokens = sorted(set(text))
        self.stoi = {self.unk_token: 0}

        for token in base_tokens:
            self.stoi[token] = len(self.stoi)

        symbols = list(text)
        self.merges = []

        for _ in range(num_merges):
            pair_counts = self._count_pairs(symbols)

            if not pair_counts:
                break

            best_pair, best_count = pair_counts.most_common(1)[0]

            if best_count < 2:
                break

            merged_token = "".join(best_pair)
            symbols = self._merge_pair(symbols, best_pair, merged_token)

            if merged_token not in self.stoi:
                self.stoi[merged_token] = len(self.stoi)

            self.merges.append(best_pair)

        self.itos = {i: token for token, i in self.stoi.items()}
        return self

    def encode(self, text):
        if not self.stoi:
            raise ValueError("Tokenizer não treinado.")

        symbols = [ch if ch in self.stoi else self.unk_token for ch in text]

        for pair in self.merges:
            merged_token = "".join(pair)
            symbols = self._merge_pair(symbols, pair, merged_token)

        unk_id = self.stoi[self.unk_token]
        return [self.stoi.get(symbol, unk_id) for symbol in symbols]

    def decode(self, ids):
        tokens = []

        for token_id in ids:
            tokens.append(self.itos[token_id])

        return "".join(tokens)

    def to_dict(self):
        return {
            "unk_token": self.unk_token,
            "merges": [list(pair) for pair in self.merges],
            "stoi": self.stoi,
        }

    @classmethod
    def from_dict(cls, data):
        tokenizer = cls(unk_token=data["unk_token"])
        tokenizer.merges = [tuple(pair) for pair in data["merges"]]
        tokenizer.stoi = {token: int(token_id) for token, token_id in data["stoi"].items()}
        tokenizer.itos = {token_id: token for token, token_id in tokenizer.stoi.items()}
        return tokenizer

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path):
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(data)

    @staticmethod
    def _count_pairs(symbols):
        counts = Counter()

        for left, right in zip(symbols, symbols[1:]):
            counts[(left, right)] += 1

        return counts

    @staticmethod
    def _merge_pair(symbols, pair, merged_token):
        merged = []
        i = 0

        while i < len(symbols):
            if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == pair:
                merged.append(merged_token)
                i += 2
            else:
                merged.append(symbols[i])
                i += 1

        return merged
