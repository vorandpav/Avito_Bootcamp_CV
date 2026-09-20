import torch.nn as nn
from torchvision.models import (
    MobileNet_V3_Large_Weights,
    MobileNet_V3_Small_Weights,
    mobilenet_v3_large,
    mobilenet_v3_small,
)


def create_model(
        name: str = "mobilenet_v3_large",
        pretrained: bool = True,
        dropout: float = 0.2,
):
    if name == "mobilenet_v3_large":
        weights = (
            MobileNet_V3_Large_Weights.DEFAULT
            if pretrained
            else None
        )

        model = mobilenet_v3_large(weights=weights)

    elif name == "mobilenet_v3_small":
        weights = (
            MobileNet_V3_Small_Weights.DEFAULT
            if pretrained
            else None
        )

        model = mobilenet_v3_small(weights=weights)

    else:
        raise ValueError(
            f"Unknown model architecture: {name}"
        )

    in_features = model.classifier[-1].in_features

    model.classifier[-1] = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, 2),
    )

    return model
