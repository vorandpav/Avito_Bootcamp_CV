from .baseline import train_baseline


TRAINERS = {
    "baseline": train_baseline,
}


__all__ = [
    "TRAINERS",
    "train_baseline",
]