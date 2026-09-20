from pathlib import Path

from .dataset import (
    InferenceSample,
    OrientationDataset,
    Sample,
    SourceSample,
)
from .split import split_samples
from .transforms import create_transform


def _find_images(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file()
        and path.suffix.lower() == ".png"
    )


def _load_source_samples(config) -> list[SourceSample]:
    samples = []

    for source_name, source_config in config.data.sources.items():
        if not source_config.enabled:
            continue

        source_dir = config.data.processed_dir / source_name

        paths = _find_images(source_dir)

        if source_config.max_samples is not None:
            paths = paths[:source_config.max_samples]

        samples.extend(
            SourceSample(
                path=path,
                source=source_name,
            )
            for path in paths
        )

    return samples


def _make_labeled_samples(
        source_samples: list[SourceSample],
) -> list[Sample]:
    samples = []

    for source_sample in source_samples:
        samples.append(
            Sample(
                path=source_sample.path,
                source=source_sample.source,
                label=0,
                rotate=False,
            )
        )

        samples.append(
            Sample(
                path=source_sample.path,
                source=source_sample.source,
                label=1,
                rotate=True,
            )
        )

    return samples


def _make_inference_samples(
        source_samples: list[SourceSample],
) -> list[InferenceSample]:
    return [
        InferenceSample(path=sample.path)
        for sample in source_samples
    ]


def create_datasets(config):
    source_samples = _load_source_samples(config)

    train_sources, validation_sources, test_sources = split_samples(
        samples=source_samples,
        train_ratio=config.data.split.train_ratio,
        validation_ratio=config.data.split.validation_ratio,
        test_ratio=config.data.split.test_ratio,
    )

    train_samples = _make_labeled_samples(train_sources)
    validation_samples = _make_labeled_samples(validation_sources)
    test_samples = _make_labeled_samples(test_sources)

    transform = create_transform(
        input_height=config.model.input_height,
        input_width=config.model.input_width,
        augmentation_config=config.augmentation,
    )

    train_dataset = OrientationDataset(
        samples=train_samples,
        transform=transform,
    )

    validation_dataset = OrientationDataset(
        samples=validation_samples,
        transform=transform,
    )

    test_dataset = OrientationDataset(
        samples=test_samples,
        transform=transform,
    )

    return train_dataset, validation_dataset, test_dataset
