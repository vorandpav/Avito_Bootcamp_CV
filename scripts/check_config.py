import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.reproducibility import select_device


def main():
    config = load_config("configs/baseline.yaml")
    device = select_device()

    print("=== Configuration Loaded Successfully ===")
    print(f"Experiment Name: {config.experiment.name}")
    print(f"Model Architecture: {config.model.name}")
    print(f"Input Resolution: {config.model.input_width}x{config.model.input_height}")
    print(f"Target Device: {device}")


if __name__ == "__main__":
    main()