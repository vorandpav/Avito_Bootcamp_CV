from pathlib import Path

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader


def freeze_backbone(model):
    for parameter in model.features.parameters():
        parameter.requires_grad = False


def unfreeze_backbone(model):
    for parameter in model.features.parameters():
        parameter.requires_grad = True


def train_one_epoch(
        model,
        loader: DataLoader,
        criterion,
        optimizer,
        device: torch.device,
):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(
            loader,
            desc="Training",
            leave=False,
    ):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def validate(
        model,
        loader: DataLoader,
        criterion,
        device: torch.device,
):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0
    total_brier = 0.0

    for images, labels in tqdm(
            loader,
            desc="Validation",
            leave=False,
    ):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()

        probabilities = torch.softmax(outputs, dim=1)[:, 1]

        brier = torch.sum(
            (probabilities - labels.float()) ** 2
        )

        total_brier += brier.item()
        total += labels.size(0)

    loss = total_loss / total
    accuracy = correct / total
    brier_score = total_brier / total

    score = 1.0 - brier_score

    return loss, accuracy, brier_score, score


def get_checkpoint_path(config) -> Path:
    return (
            Path("checkpoints")
            / config.experiment.name
            / "best.pt"
    )


def save_checkpoint(
        model,
        config,
):
    path = get_checkpoint_path(config)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
        },
        path,
    )
