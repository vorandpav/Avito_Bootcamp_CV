from pathlib import Path
import yaml
from pydantic import BaseModel


class ExperimentConfig(BaseModel):
    name: str = "baseline"
    trainer: str = "baseline"
    seed: int = 42


class SourceConfig(BaseModel):
    enabled: bool = True
    max_samples: int | None = None


class SplitConfig(BaseModel):
    train_ratio: float = 0.8
    validation_ratio: float = 0.1
    test_ratio: float = 0.1


class DataConfig(BaseModel):
    num_workers: int = 2
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")

    rustitw_gdrive_id: str = ""
    rustitw_target_max_crops: int = 35000

    hiertext_annotation_url: str = "https://raw.githubusercontent.com/google-research-datasets/hiertext/70b6620b2b112597d8219e11eee9773a1403827c/gt/validation.jsonl.gz"
    hiertext_archive_url: str = "https://open-images-dataset.s3.amazonaws.com/ocr/validation.tgz"
    hiertext_target_max_crops: int = 35000

    sources: dict[str, SourceConfig] = {
        "synthetic_crops": SourceConfig(max_samples=20000),
        "rustitw_crops": SourceConfig(max_samples=35000),
        "hiertext_crops": SourceConfig(max_samples=35000),
    }

    split: SplitConfig = SplitConfig()


class SyntheticConfig(BaseModel):
    train_samples: int = 2048
    validation_samples: int = 512
    languages: list[str] = ["ru", "en"]
    font_assets_dir: str = "assets/fonts"
    min_font_size: int = 16
    max_font_size: int = 48


class ModelConfig(BaseModel):
    name: str = "mobilenet_v3_large"
    pretrained: bool = True
    input_height: int = 96
    input_width: int = 384
    dropout: float = 0.2


class TrainingConfig(BaseModel):
    batch_size: int = 32
    frozen_epochs: int = 1
    finetune_epochs: int = 5
    frozen_learning_rate: float = 1e-3
    finetune_learning_rate: float = 1e-4
    weight_decay: float = 1e-2


class AugmentationConfig(BaseModel):
    enabled: bool = False

    rotation_probability: float = 1.0
    rotation_std: float = 20.0
    rotation_limit: float = 60.0

    color_jitter_probability: float = 0.5
    brightness: float = 0.2
    contrast: float = 0.2
    saturation: float = 0.2
    hue: float = 0.05

    blur_probability: float = 0.25

    noise_probability: float = 0.25
    noise_std: float = 0.02

    compression_probability: float = 0.25
    compression_quality_min: int = 70
    compression_quality_max: int = 95

    perspective_probability: float = 0.25
    perspective_distortion_scale: float = 0.15


class Config(BaseModel):
    experiment: ExperimentConfig = ExperimentConfig()
    data: DataConfig = DataConfig()
    synthetic: SyntheticConfig = SyntheticConfig()
    model: ModelConfig = ModelConfig()
    augmentation: AugmentationConfig = AugmentationConfig()
    training: TrainingConfig = TrainingConfig()

    def to_dict(self) -> dict:
        return self.model_dump()


def load_config(path: str | Path = "configs/baseline.yaml") -> Config:
    path = Path(path)
    src_dir = Path(__file__).resolve().parent
    project_root = src_dir.parent
    path = project_root / path

    with open(path, "r", encoding="utf-8") as f:
        raw_data = yaml.safe_load(f) or {}

    return Config(**raw_data)
