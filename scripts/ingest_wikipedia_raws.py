import argparse
import json
import re
import time
from datetime import date
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlencode, urlparse
from urllib.request import Request, urlopen


SOURCES_PATH = "docs/wikipedia_sources.md"
OUTPUT_DIR = "data/raw/wikipedia"
MANIFEST_PATH = "data/raw/wikipedia/MANIFEST.md"
WIKIPEDIA_API_URL = "https://pt.wikipedia.org/w/api.php"
USER_AGENT = "mini-llm-pt/0.1 educational dataset builder (private educational project)"
LICENSE_NAME = "CC BY-SA"
DEFAULT_RETRIES = 3
DEFAULT_RETRY_SLEEP = 10.0


LINK_PATTERN = re.compile(
    r"- \[(?P<label>[^\]]+)\]\((?P<url>https://pt\.wikipedia\.org/wiki/[^\)]+)\)"
)


def extract_wikipedia_links(markdown_text):
    """Extrai links da Wikipedia em português de um arquivo markdown."""
    links = []
    seen_urls = set()

    for match in LINK_PATTERN.finditer(markdown_text):
        label = match.group("label").strip()
        url = match.group("url").strip()

        if url in seen_urls:
            continue

        seen_urls.add(url)
        links.append({"label": label, "url": url})

    return links


def title_from_wikipedia_url(url):
    """Obtém o título do artigo a partir da URL /wiki/Titulo."""
    path = urlparse(url).path
    title = path.removeprefix("/wiki/")
    title = unquote(title)
    title = title.replace("_", " ")
    return title


def slugify(text):
    """Cria um nome de arquivo simples e estável."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9áàâãéêíóôõúç]+", "_", text)
    text = text.strip("_")
    return text[:80] or "artigo"


def fetch_json_with_retries(url, retries=DEFAULT_RETRIES, retry_sleep=DEFAULT_RETRY_SLEEP):
    """Faz uma requisição HTTP com retry simples para erros temporários."""
    last_error = None

    for attempt in range(1, retries + 1):
        request = Request(url, headers={"User-Agent": USER_AGENT})

        try:
            with urlopen(request, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            last_error = error

            if error.code == 429:
                retry_after = error.headers.get("Retry-After")
                wait_seconds = float(retry_after) if retry_after else retry_sleep * attempt
                print(
                    f"  Aviso: limite da Wikipedia atingido (429). "
                    f"Tentativa {attempt}/{retries}. Aguardando {wait_seconds:.1f}s..."
                )
                time.sleep(wait_seconds)
                continue

            if 500 <= error.code < 600:
                wait_seconds = retry_sleep * attempt
                print(
                    f"  Aviso: erro HTTP temporário {error.code}. "
                    f"Tentativa {attempt}/{retries}. Aguardando {wait_seconds:.1f}s..."
                )
                time.sleep(wait_seconds)
                continue

            raise
        except URLError as error:
            last_error = error
            wait_seconds = retry_sleep * attempt
            print(
                f"  Aviso: erro de conexão. "
                f"Tentativa {attempt}/{retries}. Aguardando {wait_seconds:.1f}s..."
            )
            time.sleep(wait_seconds)

    raise last_error


def fetch_plaintext_article(title, retries=DEFAULT_RETRIES, retry_sleep=DEFAULT_RETRY_SLEEP):
    """Baixa o texto puro de um artigo usando a API pública da Wikipedia."""
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts|info",
        "explaintext": "1",
        "redirects": "1",
        "inprop": "url",
        "titles": title,
    }

    query = urlencode(params, encoding="utf-8")
    url = f"{WIKIPEDIA_API_URL}?{query}"
    payload = fetch_json_with_retries(url, retries=retries, retry_sleep=retry_sleep)

    pages = payload.get("query", {}).get("pages", {})
    page = next(iter(pages.values()))

    if "missing" in page:
        raise ValueError(f"Artigo não encontrado: {title}")

    resolved_title = page.get("title", title)
    full_url = page.get("fullurl", "")
    extract = page.get("extract", "").strip()

    if not extract:
        raise ValueError(f"Artigo sem texto extraído: {title}")

    return {
        "title": resolved_title,
        "url": full_url,
        "text": extract,
    }


def write_manifest(entries, manifest_path):
    lines = [
        "# Manifesto de fontes raw da Wikipedia",
        "",
        f"Data de geração: {date.today().isoformat()}",
        "",
        "Licença declarada para fontes Wikimedia/Wikipedia: CC BY-SA.",
        "",
        "Este manifesto registra os artigos baixados para `data/raw/wikipedia/`.",
        "",
        "| arquivo | título | url | licença | status | observação |",
        "|---|---|---|---|---|---|",
    ]

    for entry in entries:
        lines.append(
            "| {file} | {title} | {url} | {license} | {status} | {note} |".format(
                file=entry.get("file", ""),
                title=entry.get("title", "").replace("|", "-"),
                url=entry.get("url", ""),
                license=entry.get("license", LICENSE_NAME),
                status=entry.get("status", ""),
                note=entry.get("note", "").replace("|", "-"),
            )
        )

    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def ingest(
    start=1,
    limit=None,
    sleep_seconds=2.0,
    retries=DEFAULT_RETRIES,
    retry_sleep=DEFAULT_RETRY_SLEEP,
):
    sources_path = Path(SOURCES_PATH)
    output_dir = Path(OUTPUT_DIR)
    manifest_path = Path(MANIFEST_PATH)

    if start <= 0:
        raise ValueError("start precisa ser maior que zero.")

    markdown_text = sources_path.read_text(encoding="utf-8")
    all_links = extract_wikipedia_links(markdown_text)

    start_index = start - 1
    links = all_links[start_index:]

    if limit is not None:
        links = links[:limit]

    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_entries = []
    total = len(links)

    for offset, link in enumerate(links, start=0):
        source_index = start + offset
        original_title = title_from_wikipedia_url(link["url"])
        print(f"[{offset + 1}/{total} | fonte #{source_index}] Baixando: {original_title}")

        try:
            article = fetch_plaintext_article(
                original_title,
                retries=retries,
                retry_sleep=retry_sleep,
            )
            filename = f"{source_index:03d}_{slugify(article['title'])}.txt"
            file_path = output_dir / filename
            file_path.write_text(article["text"] + "\n", encoding="utf-8")

            manifest_entries.append(
                {
                    "file": str(file_path).replace("\\", "/"),
                    "title": article["title"],
                    "url": article["url"] or link["url"],
                    "license": LICENSE_NAME,
                    "status": "ok",
                    "note": "Texto extraído via MediaWiki API com explaintext=1.",
                }
            )
        except Exception as error:
            manifest_entries.append(
                {
                    "file": "",
                    "title": original_title,
                    "url": link["url"],
                    "license": LICENSE_NAME,
                    "status": "erro",
                    "note": str(error),
                }
            )
            print(f"  Erro: {error}")

        time.sleep(sleep_seconds)

    write_manifest(manifest_entries, manifest_path)
    print(f"\nManifesto salvo em: {manifest_path}")
    print(f"Arquivos raw salvos em: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Baixa artigos da Wikipedia listados em docs/wikipedia_sources.md."
    )
    parser.add_argument(
        "--start",
        type=int,
        default=1,
        help="Índice inicial 1-based da lista de fontes. Exemplo: --start 11.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limita a quantidade de artigos baixados. Útil para testes e blocos.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=2.0,
        help="Tempo de espera entre requisições, em segundos.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=DEFAULT_RETRIES,
        help="Quantidade de tentativas por artigo quando houver erro temporário.",
    )
    parser.add_argument(
        "--retry-sleep",
        type=float,
        default=DEFAULT_RETRY_SLEEP,
        help="Espera base entre retries em segundos.",
    )
    args = parser.parse_args()

    ingest(
        start=args.start,
        limit=args.limit,
        sleep_seconds=args.sleep,
        retries=args.retries,
        retry_sleep=args.retry_sleep,
    )


if __name__ == "__main__":
    main()
