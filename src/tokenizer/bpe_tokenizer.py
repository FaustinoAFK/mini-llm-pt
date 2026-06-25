import json
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer


class BPETokenizer:
    def __init__(
        self,
        unk_token="<unk>",
        bos_token="<bos>",
        eos_token="<eos>",
        paragraph_token="<par>",
    ):
        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.paragraph_token = paragraph_token
        self.special_tokens = [
            self.unk_token,
            self.bos_token,
            self.eos_token,
            self.paragraph_token,
        ]

        self._tokenizer = self._build_tokenizer()
        self.stoi = {token: index for index, token in enumerate(self.special_tokens)}
        self.itos = {index: token for index, token in enumerate(self.special_tokens)}
        self.merges = []

    @property
    def vocab_size(self):
        return self._tokenizer.get_vocab_size()

    def _build_tokenizer(self):
        tokenizer = Tokenizer(BPE(unk_token=self.unk_token))
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        tokenizer.decoder = ByteLevelDecoder()
        tokenizer.add_special_tokens(self.special_tokens)
        return tokenizer

    def train(self, text, num_merges=100, vocab_size=None, min_frequency=2):
        if not text:
            raise ValueError("O texto de treino nao pode ser vazio.")

        if num_merges < 0:
            raise ValueError("num_merges precisa ser maior ou igual a zero.")

        if vocab_size is not None and vocab_size <= len(self.special_tokens):
            raise ValueError(
                "vocab_size precisa ser maior que a quantidade de special tokens."
            )

        if min_frequency <= 0:
            raise ValueError("min_frequency precisa ser maior que zero.")

        alphabet = ByteLevel.alphabet()
        if vocab_size is None:
            vocab_size = max(
                len(alphabet) + len(self.special_tokens),
                len(alphabet) + len(self.special_tokens) + num_merges,
            )

        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=self.special_tokens,
            initial_alphabet=alphabet,
        )
        self._tokenizer = self._build_tokenizer()
        self._tokenizer.train_from_iterator([text], trainer=trainer)
        self._sync_vocab()
        return self

    def encode(self, text, add_bos=False, add_eos=False):
        ids = self._tokenizer.encode(text).ids
        if add_bos:
            ids = [self.stoi[self.bos_token]] + ids
        if add_eos:
            ids = ids + [self.stoi[self.eos_token]]
        return ids

    def decode(self, ids):
        return self._tokenizer.decode(ids, skip_special_tokens=False)

    def decode_for_display(self, ids):
        text = self.decode(ids)
        text = text.replace(self.bos_token, "")
        text = text.replace(self.eos_token, "")
        text = text.replace(self.paragraph_token, "\n\n")

        normalized_lines = [" ".join(line.split()) for line in text.split("\n")]
        text = "\n".join(normalized_lines)
        text = text.replace("\n \n", "\n\n")
        return text.strip()

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self._tokenizer.to_str(pretty=True), encoding="utf-8")

    @classmethod
    def load(cls, path):
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))

        if data.get("model", {}).get("type") == "BPE":
            added_tokens = {
                token["content"]
                for token in data.get("added_tokens", [])
                if isinstance(token, dict) and "content" in token
            }
            tokenizer = cls(
                unk_token=data["model"].get("unk_token", "<unk>"),
                bos_token="<bos>",
                eos_token="<eos>",
                paragraph_token="<par>",
            )
            tokenizer._tokenizer = Tokenizer.from_str(json.dumps(data))
            tokenizer._sync_vocab()
            return tokenizer

        return cls._load_legacy(data)

    @classmethod
    def _load_legacy(cls, data):
        tokenizer = cls(
            unk_token=data.get("unk_token", "<unk>"),
            bos_token=data.get("bos_token", "<bos>"),
            eos_token=data.get("eos_token", "<eos>"),
            paragraph_token=data.get("paragraph_token", "<par>"),
        )
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

    def decode(self, ids, skip_special_tokens=False):
        return self.tokenizer._legacy_decode(ids)

    def to_str(self, pretty=False):
        return json.dumps(
            {
                "unk_token": self.tokenizer.unk_token,
                "bos_token": self.tokenizer.bos_token,
                "eos_token": self.tokenizer.eos_token,
                "paragraph_token": self.tokenizer.paragraph_token,
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
