from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


@dataclass(frozen=True)
class SourceSample:
    path: Path
    source: str


@dataclass(frozen=True)
class Sample:
    path: Path
    source: str
    label: int
    rotate: bool


@dataclass(frozen=True)
class InferenceSample:
    path: Path


class OrientationDataset(Dataset):
    def __init__(
        self,
        samples: list[Sample],
        transform=None,
    ):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]

        image = Image.open(sample.path).convert("RGB")

        if sample.rotate:
            image = image.rotate(180)

        if self.transform is not None:
            image = self.transform(image)

        return image, sample.label


class InferenceDataset(Dataset):
    def __init__(
        self,
        samples: list[InferenceSample],
        transform=None,
    ):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]

        image = Image.open(sample.path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image