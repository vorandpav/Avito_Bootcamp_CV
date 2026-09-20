import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from config import load_config
from model import create_model
from reproducibility import select_device


DEFAULT_CONFIGS = [
    "configs/augmentation_v1.yaml",
    "configs/mobilenet_small_128x512.yaml",
    "configs/mobilenet_small.yaml",
    "configs/baseline.yaml",
]


def count_parameters(model) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
    )


def load_model(config, device: torch.device):
    model = create_model(
        name=config.model.name,
        pretrained=False,
        dropout=config.model.dropout,
    ).to(device)

    checkpoint_path = (
        Path("checkpoints")
        / config.experiment.name
        / "best.pt"
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model, checkpoint_path


@torch.no_grad()
def benchmark_forward(
        model,
        batch: torch.Tensor,
        device: torch.device,
        warmup: int,
        runs: int,
) -> tuple[float, float]:
    """
    Возвращает:
      - среднее время на батч (ms)
      - среднее время на картинку (ms)
    """

    for _ in range(warmup):
        _ = model(batch)

    if device.type == "cuda":
        torch.cuda.synchronize()

    started = time.perf_counter()

    for _ in range(runs):
        _ = model(batch)

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - started

    batch_ms = (elapsed / runs) * 1000.0
    image_ms = batch_ms / batch.size(0)

    return batch_ms, image_ms


def format_millions(value: int) -> str:
    return f"{value / 1_000_000:.2f}M"


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--configs",
        nargs="+",
        default=DEFAULT_CONFIGS,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=100,
    )

    args = parser.parse_args()

    device = select_device()

    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Warmup: {args.warmup}")
    print(f"Runs: {args.runs}")
    print()

    rows = []

    for config_path in args.configs:
        config = load_config(config_path)

        model, checkpoint_path = load_model(
            config=config,
            device=device,
        )

        parameters = count_parameters(model)
        checkpoint_mb = (
            checkpoint_path.stat().st_size / (1024 * 1024)
        )

        batch = torch.randn(
            args.batch_size,
            3,
            config.model.input_height,
            config.model.input_width,
            device=device,
        )

        batch_ms, image_ms = benchmark_forward(
            model=model,
            batch=batch,
            device=device,
            warmup=args.warmup,
            runs=args.runs,
        )

        images_per_second = 1000.0 / image_ms

        row = {
            "name": config.experiment.name,
            "model": config.model.name,
            "input": (
                f"{config.model.input_height}"
                f"x{config.model.input_width}"
            ),
            "params": parameters,
            "checkpoint_mb": checkpoint_mb,
            "batch_ms": batch_ms,
            "image_ms": image_ms,
            "images_per_second": images_per_second,
        }

        rows.append(row)

        print(f"Config: {config_path}")
        print(f"  experiment : {row['name']}")
        print(f"  model      : {row['model']}")
        print(f"  input      : {row['input']}")
        print(f"  params     : {format_millions(parameters)}")
        print(f"  checkpoint : {checkpoint_mb:.1f} MB")
        print(f"  batch      : {batch_ms:.2f} ms")
        print(f"  per image  : {image_ms:.3f} ms")
        print(f"  throughput : {images_per_second:.0f} img/s")
        print()

        del model
        del batch

        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("=" * 78)
    print(
        f"{'experiment':<28}"
        f"{'input':<10}"
        f"{'params':>8}"
        f"{'ckptMB':>8}"
        f"{'ms/img':>10}"
        f"{'img/s':>10}"
    )
    print("-" * 78)

    for row in rows:
        print(
            f"{row['name']:<28}"
            f"{row['input']:<10}"
            f"{format_millions(row['params']):>8}"
            f"{row['checkpoint_mb']:>8.1f}"
            f"{row['image_ms']:>10.3f}"
            f"{row['images_per_second']:>10.0f}"
        )

    print("=" * 78)


if __name__ == "__main__":
    main()
