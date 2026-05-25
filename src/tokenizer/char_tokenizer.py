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

    def encode(self, text):
        # Transforma texto em IDs
        ids = []

        for ch in text:
            ids.append(self.stoi.get(ch, self.stoi[self.unk_token]))

        return ids

    def decode(self, ids):
        # Transforma IDs de volta em texto
        chars = []

        for i in ids:
            chars.append(self.itos[i])

        return "".join(chars)
