from src.tokenizer.char_tokenizer import CharTokenizer

text = "portal"
tokenizer = CharTokenizer(text)
print(tokenizer.encode("portaria"))