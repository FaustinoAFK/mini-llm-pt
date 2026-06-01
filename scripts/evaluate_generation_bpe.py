import argparse
from datetime import datetime, timezone
from pathlib import Path

from scripts.generate_transformer_bpe import (
    CHECKPOINT_PATH,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    generate_text,
)


OUTPUT_PATH = "artifacts/evaluations/transformer_bpe_generation.txt"
DEFAULT_MAX_NEW_TOKENS = 120
PROMPTS = [
    "A inteligencia artificial ",
    "Python e uma linguagem ",
    "Redes neurais sao ",
    "O aprendizado de maquina ",
    "Um modelo de linguagem ",
]


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def build_report(checkpoint_path, max_new_tokens, temperature, top_k):
    lines = [
        "# Avaliacao qualitativa do Transformer BPE",
        "",
        f"timestamp_utc: {utc_now_iso()}",
        f"checkpoint_path: {checkpoint_path}",
        f"max_new_tokens: {max_new_tokens}",
        f"temperature: {temperature}",
        f"top_k: {top_k}",
        "",
    ]

    for index, prompt in enumerate(PROMPTS, start=1):
        generated_text = generate_text(
            checkpoint_path=checkpoint_path,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        lines.extend(
            [
                f"## Prompt {index}",
                "",
                f"prompt: {prompt!r}",
                "",
                generated_text,
                "",
            ]
        )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Avalia qualitativamente o checkpoint BPE com prompts fixos."
    )
    parser.add_argument("--checkpoint-path", default=CHECKPOINT_PATH)
    parser.add_argument("--output-path", default=OUTPUT_PATH)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    args = parser.parse_args()

    report = build_report(
        checkpoint_path=args.checkpoint_path,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")

    print(f"Relatorio de avaliacao salvo em: {output_path}")


if __name__ == "__main__":
    main()
