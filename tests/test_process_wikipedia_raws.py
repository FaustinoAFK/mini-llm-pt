from scripts.process_wikipedia_raws import PARAGRAPH_TOKEN, clean_wikipedia_text


def test_clean_wikipedia_text_preserves_paragraph_boundaries():
    text = (
        "Primeiro paragrafo com conteudo.\n"
        "Ainda no primeiro.\n\n"
        "Segundo paragrafo limpo."
    )

    cleaned = clean_wikipedia_text(text)

    assert PARAGRAPH_TOKEN in cleaned
    assert cleaned.count(PARAGRAPH_TOKEN) == 1


def test_clean_wikipedia_text_drops_common_noise():
    text = (
        "Texto util para treino.\n"
        "== Referencias ==\n"
        "displaystyle x = 1\n"
        "Mais uma linha valida."
    )

    cleaned = clean_wikipedia_text(text)

    assert "Referencias" not in cleaned
    assert "displaystyle" not in cleaned
    assert "Texto util para treino." in cleaned
    assert "Mais uma linha valida." in cleaned
