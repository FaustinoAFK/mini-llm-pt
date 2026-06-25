class CharTokenizer:
    def __init__(self, text):
        # 1. Descobre todos os caracteres únicos do texto
        chars = sorted(list(set(text)))

        # 2. Cria o vocabulário com um token especial para caracteres desconhecidos
        self.unk_token = "<unk>"
        self.stoi = {self.unk_token: 0}

        for ch in chars:
            self.stoi[ch] = len(self.stoi)

        self.itos = {i: ch for ch, i in self.stoi.items()}

    @property
    def vocab_size(self):
        # Retorna o tamanho do vocabulário conhecido pelo tokenizer
        return len(self.stoi)

    def encode(self, text):
        # Transforma texto em IDs
        unk_id = self.stoi[self.unk_token]
        return [self.stoi.get(ch, unk_id) for ch in text]

    def decode(self, ids):
        # Transforma IDs de volta em texto
        return "".join(self.itos[i] for i in ids)
