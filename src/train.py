import argparse

from torch.utils.data import DataLoader

from config import load_config
from data_preprocessing import create_datasets
from experiments import TRAINERS
from reproducibility import seed_everything, select_device


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline.yaml",
    )

    args = parser.parse_args()

    config = load_config(args.config)

    seed_everything(config.experiment.seed)

    device = select_device()

    print(f"Device: {device}")
    print(f"Config: {args.config}")

    train_dataset, validation_dataset, test_dataset = create_datasets(config)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=device.type == "cuda",
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=device.type == "cuda",
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=device.type == "cuda",
    )

    trainer = TRAINERS[config.experiment.trainer]

    trainer(
        config=config,
        device=device,
        train_loader=train_loader,
        validation_loader=validation_loader,
        test_loader=test_loader,
    )


if __name__ == "__main__":
    main()