from types import SimpleNamespace

import pytest
import torch

from scripts import train_transformer_bpe
from src.tokenizer.bpe_tokenizer import BPETokenizer


def make_args(**overrides):
    values = {
        "block_size": 4,
        "batch_size": 2,
        "eval_batch_size": 2,
        "eval_num_batches": 1,
        "max_iters": 4,
        "learning_rate": 1e-3,
        "min_learning_rate": 1e-4,
        "warmup_iters": 1,
        "n_embd": 8,
        "n_head": 2,
        "n_layer": 1,
        "eval_interval": 1,
        "patience": 2,
        "grad_clip": 1.0,
        "gradient_accumulation_steps": 1,
        "dropout": 0.1,
        "disable_lr_schedule": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_validate_args_accepts_new_training_options():
    args = make_args(
        grad_clip=0.5,
        warmup_iters=0,
        min_learning_rate=1e-5,
        gradient_accumulation_steps=2,
    )

    train_transformer_bpe.validate_args(args)


def test_validate_args_rejects_invalid_new_training_options():
    with pytest.raises(ValueError):
        train_transformer_bpe.validate_args(make_args(grad_clip=0))

    with pytest.raises(ValueError):
        train_transformer_bpe.validate_args(make_args(warmup_iters=-1))

    with pytest.raises(ValueError):
        train_transformer_bpe.validate_args(make_args(min_learning_rate=2e-3))

    with pytest.raises(ValueError):
        train_transformer_bpe.validate_args(make_args(gradient_accumulation_steps=0))


def test_compute_learning_rate_warmup_cosine_and_disable():
    warmup_lr = train_transformer_bpe.compute_learning_rate(
        step=0,
        max_iters=10,
        learning_rate=1.0,
        min_learning_rate=0.1,
        warmup_iters=2,
    )
    peak_lr = train_transformer_bpe.compute_learning_rate(
        step=1,
        max_iters=10,
        learning_rate=1.0,
        min_learning_rate=0.1,
        warmup_iters=2,
    )
    final_lr = train_transformer_bpe.compute_learning_rate(
        step=10,
        max_iters=10,
        learning_rate=1.0,
        min_learning_rate=0.1,
        warmup_iters=2,
    )
    disabled_lr = train_transformer_bpe.compute_learning_rate(
        step=5,
        max_iters=10,
        learning_rate=1.0,
        min_learning_rate=0.1,
        warmup_iters=2,
        disable_lr_schedule=True,
    )

    assert warmup_lr == pytest.approx(0.5)
    assert peak_lr == pytest.approx(1.0)
    assert final_lr == pytest.approx(0.1)
    assert disabled_lr == pytest.approx(1.0)


def test_train_transformer_bpe_checkpoint_and_resume(tmp_path, monkeypatch):
    train_path = tmp_path / "train.txt"
    val_path = tmp_path / "val.txt"
    tokenizer_path = tmp_path / "bpe.json"
    checkpoint_path = tmp_path / "smoke.pt"
    resumed_checkpoint_path = tmp_path / "smoke_resume.pt"
    metrics_path = tmp_path / "smoke.jsonl"
    resumed_metrics_path = tmp_path / "smoke_resume.jsonl"

    train_text = "aprendizado de maquina em portugues. " * 4
    val_text = "modelo de linguagem em portugues. " * 2
    train_path.write_text(train_text, encoding="utf-8")
    val_path.write_text(val_text, encoding="utf-8")

    tokenizer = BPETokenizer()
    tokenizer.train(train_text, num_merges=10)
    tokenizer.save(tokenizer_path)

    monkeypatch.setattr(
        "sys.argv",
        [
            "train_transformer_bpe.py",
            "--train-path",
            str(train_path),
            "--val-path",
            str(val_path),
            "--tokenizer-path",
            str(tokenizer_path),
            "--checkpoint-path",
            str(checkpoint_path),
            "--metrics-path",
            str(metrics_path),
            "--device",
            "cpu",
            "--block-size",
            "4",
            "--batch-size",
            "2",
            "--eval-batch-size",
            "2",
            "--eval-num-batches",
            "1",
            "--max-iters",
            "2",
            "--eval-interval",
            "1",
            "--patience",
            "10",
            "--n-embd",
            "8",
            "--n-head",
            "2",
            "--n-layer",
            "1",
        ],
    )
    train_transformer_bpe.main()

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert checkpoint["completed_steps"] == 2
    assert "optimizer_state_dict" in checkpoint
    assert "scheduler_state_dict" in checkpoint
    assert "training_model_state_dict" in checkpoint
    assert checkpoint["torch_rng_state"] is not None

    monkeypatch.setattr(
        "sys.argv",
        [
            "train_transformer_bpe.py",
            "--train-path",
            str(train_path),
            "--val-path",
            str(val_path),
            "--tokenizer-path",
            str(tokenizer_path),
            "--resume-from",
            str(checkpoint_path),
            "--checkpoint-path",
            str(resumed_checkpoint_path),
            "--metrics-path",
            str(resumed_metrics_path),
            "--device",
            "cpu",
            "--block-size",
            "4",
            "--batch-size",
            "2",
            "--eval-batch-size",
            "2",
            "--eval-num-batches",
            "1",
            "--max-iters",
            "3",
            "--eval-interval",
            "1",
            "--patience",
            "10",
            "--n-embd",
            "8",
            "--n-head",
            "2",
            "--n-layer",
            "1",
        ],
    )
    train_transformer_bpe.main()

    resumed_checkpoint = torch.load(
        resumed_checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    assert resumed_checkpoint["completed_steps"] == 3
    assert resumed_checkpoint["resume_from"] == str(checkpoint_path)
