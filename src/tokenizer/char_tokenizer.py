class CharTokenizer:
    def __init__(self, text):
        # 1. Descobre todos os caracteres únicos do texto
                                                             = sorted(list(set(text)))

        # 2. Cria o vocabulário
        self.stoi = {ch: i for i, ch in enumerate(chars)}  # string to integer
        self.itos = {i: ch for i, ch in enumerate(chars)}  # integer to string

    def encode(self, text):
        # Transforma texto em IDs
        ids = []

        for ch in text:
            ids.append(self.stoi[ch])

        return ids

    def decode(self, ids):
        # Transforma IDs de volta em texto
        chars = []

        for i in ids:
            chars.append(self.itos[i])

        return "".join(chars)