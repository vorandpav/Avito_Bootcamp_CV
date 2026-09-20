import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from config import load_config
from data_preprocessing import InferenceDataset
from data_preprocessing.dataset import InferenceSample
from data_preprocessing.transforms import create_transform, create_inference_transform
from model import create_model
from reproducibility import select_device


def find_images(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file()
        and path.suffix.lower() == ".png"
    )


@torch.no_grad()
def predict(
        model,
        loader: DataLoader,
        device: torch.device,
) -> list[float]:
    model.eval()

    probabilities = []

    for images in loader:
        images = images.to(device)

        logits = model(images)
        probs = torch.softmax(logits, dim=1)

        p_180 = probs[:, 1]

        probabilities.extend(p_180.cpu().tolist())

    return probabilities


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline.yaml",
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/test/images",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="submission.csv",
    )

    args = parser.parse_args()

    config = load_config(args.config)
    device = select_device()

    print(f"Device: {device}")

    input_dir = Path(args.input_dir)

    if not input_dir.exists():
        raise FileNotFoundError(
            f"Test directory not found: {input_dir}"
        )

    image_paths = find_images(input_dir)

    if not image_paths:
        raise ValueError(
            f"No images found in: {input_dir}"
        )

    print(f"Found images: {len(image_paths)}")

    samples = [
        InferenceSample(path=path)
        for path in image_paths
    ]

    transform = create_inference_transform(
        input_height=config.model.input_height,
        input_width=config.model.input_width,
    )

    dataset = InferenceDataset(
        samples=samples,
        transform=transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=device.type == "cuda",
    )

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

    print(f"Checkpoint: {checkpoint_path}")

    probabilities = predict(
        model=model,
        loader=loader,
        device=device,
    )

    image_ids = [
        path.stem
        for path in image_paths
    ]

    submission = pd.DataFrame({
        "image_id": image_ids,
        "p_180": probabilities,
    })

    output_path = Path(args.output)
    submission.to_csv(
        output_path,
        index=False,
    )

    print(f"Submission saved to: {output_path}")

    print()
    print(submission.head())


if __name__ == "__main__":
    main()
