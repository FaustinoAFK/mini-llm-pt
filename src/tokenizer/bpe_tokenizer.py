import json
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer


class BPETokenizer:
    def __init__(self, unk_token="<unk>"):
        self.unk_token = unk_token
        self._tokenizer = Tokenizer(BPE(unk_token=self.unk_token))
        self._tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        self._tokenizer.decoder = ByteLevelDecoder()
        self.stoi = {self.unk_token: 0}
        self.itos = {0: self.unk_token}
        self.merges = []

    @property
    def vocab_size(self):
        return self._tokenizer.get_vocab_size()

    def train(self, text, num_merges=100):
        if not text:
            raise ValueError("O texto de treino nao pode ser vazio.")

        if num_merges < 0:
            raise ValueError("num_merges precisa ser maior ou igual a zero.")

        alphabet = ByteLevel.alphabet()
        vocab_size = max(len(alphabet) + 1, len(alphabet) + 1 + num_merges)
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=2,
            special_tokens=[self.unk_token],
            initial_alphabet=alphabet,
        )
        self._tokenizer = Tokenizer(BPE(unk_token=self.unk_token))
        self._tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        self._tokenizer.decoder = ByteLevelDecoder()
        self._tokenizer.train_from_iterator([text], trainer=trainer)
        self._sync_vocab()
        return self

    def encode(self, text):
        return self._tokenizer.encode(text).ids

    def decode(self, ids):
        return self._tokenizer.decode(ids)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self._tokenizer.to_str(pretty=True), encoding="utf-8")

    @classmethod
    def load(cls, path):
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))

        if data.get("model", {}).get("type") == "BPE":
            tokenizer = cls(unk_token=data["model"].get("unk_token", "<unk>"))
            tokenizer._tokenizer = Tokenizer.from_str(json.dumps(data))
            tokenizer._sync_vocab()
            return tokenizer

        return cls._load_legacy(data)

    @classmethod
    def _load_legacy(cls, data):
        tokenizer = cls(unk_token=data["unk_token"])
        tokenizer.stoi = {token: int(token_id) for token, token_id in data["stoi"].items()}
        tokenizer.itos = {token_id: token for token, token_id in tokenizer.stoi.items()}
        tokenizer.merges = [tuple(pair) for pair in data["merges"]]
        tokenizer._tokenizer = _LegacyTokenizerAdapter(tokenizer)
        return tokenizer

    def _sync_vocab(self):
        vocab = self._tokenizer.get_vocab()
        self.stoi = dict(vocab)
        self.itos = {token_id: token for token, token_id in vocab.items()}
        model = json.loads(self._tokenizer.to_str()).get("model", {})
        self.merges = [
            tuple(merge.split(" ", 1)) if isinstance(merge, str) else tuple(merge)
            for merge in model.get("merges", [])
        ]


class _LegacyTokenizerAdapter:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def get_vocab_size(self):
        return len(self.tokenizer.stoi)

    def encode(self, text):
        return _LegacyEncoding(self.tokenizer._legacy_encode(text))

    def decode(self, ids):
        return self.tokenizer._legacy_decode(ids)

    def to_str(self, pretty=False):
        return json.dumps(
            {
                "unk_token": self.tokenizer.unk_token,
                "merges": [list(pair) for pair in self.tokenizer.merges],
                "stoi": self.tokenizer.stoi,
            },
            ensure_ascii=False,
            indent=2 if pretty else None,
        )


class _LegacyEncoding:
    def __init__(self, ids):
        self.ids = ids


def _legacy_merge_pair(symbols, pair, merged_token):
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


def _legacy_encode(self, text):
    symbols = [ch if ch in self.stoi else self.unk_token for ch in text]

    for pair in self.merges:
        merged_token = "".join(pair)
        symbols = _legacy_merge_pair(symbols, pair, merged_token)

    unk_id = self.stoi[self.unk_token]
    return [self.stoi.get(symbol, unk_id) for symbol in symbols]


def _legacy_decode(self, ids):
    return "".join(self.itos[token_id] for token_id in ids)


BPETokenizer._legacy_encode = _legacy_encode
BPETokenizer._legacy_decode = _legacy_decode
