import random

from .dataset import SourceSample


def split_samples(
        samples: list[SourceSample],
        train_ratio: float,
        validation_ratio: float,
        test_ratio: float,
) -> tuple[
    list[SourceSample],
    list[SourceSample],
    list[SourceSample],
]:
    total_ratio = train_ratio + validation_ratio + test_ratio

    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(
            "train_ratio + validation_ratio + test_ratio must be equal to 1"
        )

    samples = samples.copy()
    random.shuffle(samples)

    total = len(samples)

    train_size = int(total * train_ratio)
    validation_size = int(total * validation_ratio)

    train_samples = samples[:train_size]

    validation_samples = samples[
        train_size:train_size + validation_size
    ]

    test_samples = samples[
        train_size + validation_size:
    ]

    return train_samples, validation_samples, test_samples
