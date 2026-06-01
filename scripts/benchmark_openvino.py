import argparse
import time

import openvino as ov
import torch
import torch.nn as nn

from scripts.generate_transformer_bpe import CHECKPOINT_PATH, DEFAULT_PROMPT
from src.models.transformer import MiniTransformerLanguageModel
from src.tokenizer.bpe_tokenizer import BPETokenizer


DEFAULT_DEVICE = "CPU"
DEFAULT_WARMUP = 10
DEFAULT_ITERATIONS = 100


class LogitsOnlyModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, idx):
        logits, _ = self.model(idx)
        return logits


def load_model_and_context(checkpoint_path, prompt):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    tokenizer = BPETokenizer.load(checkpoint["tokenizer_path"])

    model = MiniTransformerLanguageModel(
        vocab_size=checkpoint["vocab_size"],
        block_size=checkpoint["block_size"],
        n_embd=checkpoint["n_embd"],
        n_head=checkpoint["n_head"],
        n_layer=checkpoint["n_layer"],
        dropout=checkpoint["dropout"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    prompt_ids = tokenizer.encode(prompt)
    if not prompt_ids:
        raise ValueError("O prompt precisa gerar pelo menos um token.")

    prompt_ids = prompt_ids[-checkpoint["block_size"] :]
    context = torch.tensor([prompt_ids], dtype=torch.long)
    return model, context


def benchmark_pytorch(model, context, warmup, iterations):
    with torch.no_grad():
        for _ in range(warmup):
            model(context)

        start = time.perf_counter()
        last_output = None
        for _ in range(iterations):
            last_output = model(context)
        elapsed = time.perf_counter() - start

    return elapsed * 1000 / iterations, last_output


def benchmark_openvino(compiled_model, context, warmup, iterations):
    input_array = context.numpy()

    for _ in range(warmup):
        compiled_model([input_array])

    start = time.perf_counter()
    last_output = None
    for _ in range(iterations):
        last_output = compiled_model([input_array])
    elapsed = time.perf_counter() - start

    output_tensor = next(iter(last_output.values()))
    return elapsed * 1000 / iterations, output_tensor


def main():
    parser = argparse.ArgumentParser(
        description="Compara inferencia PyTorch CPU com OpenVINO para o Transformer BPE."
    )
    parser.add_argument("--checkpoint-path", default=CHECKPOINT_PATH)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    args = parser.parse_args()

    if args.warmup < 0:
        raise ValueError("warmup precisa ser maior ou igual a zero.")
    if args.iterations <= 0:
        raise ValueError("iterations precisa ser maior que zero.")

    model, context = load_model_and_context(args.checkpoint_path, args.prompt)
    logits_model = LogitsOnlyModel(model).eval()

    core = ov.Core()
    print(f"OpenVINO version: {ov.__version__}")
    print(f"Available devices: {core.available_devices}")
    print(f"Selected device: {args.device}")
    print(f"Context shape: {tuple(context.shape)}")
    print(f"Iterations: {args.iterations} | warmup: {args.warmup}")

    pytorch_ms, pytorch_output = benchmark_pytorch(
        logits_model,
        context,
        warmup=args.warmup,
        iterations=args.iterations,
    )

    ov_model = ov.convert_model(logits_model, example_input=context)
    compiled_model = core.compile_model(ov_model, args.device)
    openvino_ms, openvino_output = benchmark_openvino(
        compiled_model,
        context,
        warmup=args.warmup,
        iterations=args.iterations,
    )

    reference = pytorch_output.detach().numpy()
    max_abs_diff = abs(reference - openvino_output).max()

    print("\n--- Benchmark ---")
    print(f"PyTorch CPU avg forward: {pytorch_ms:.3f} ms")
    print(f"OpenVINO {args.device} avg forward: {openvino_ms:.3f} ms")
    print(f"Speedup: {pytorch_ms / openvino_ms:.2f}x")
    print(f"Max abs diff: {max_abs_diff:.6f}")


if __name__ == "__main__":
    main()
