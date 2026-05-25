from src.tokenizer.char_tokenizer import CharTokenizer
#teste
text = "porta"

tokenizer = CharTokenizer(text)

ids = tokenizer.encode("porta")
decoded = tokenizer.decode(ids)

assert decoded == "porta"
