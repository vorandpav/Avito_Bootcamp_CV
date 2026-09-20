import torch

from model import create_model
from trainer import (
    freeze_backbone,
    get_checkpoint_path,
    save_checkpoint,
    train_one_epoch,
    unfreeze_backbone,
    validate,
)


def train_baseline(
        config,
        device,
        train_loader,
        validation_loader,
        test_loader,
):
    model = create_model(
        name=config.model.name,
        pretrained=config.model.pretrained,
        dropout=config.model.dropout,
    ).to(device)

    criterion = torch.nn.CrossEntropyLoss()

    # ============================================================
    # Phase 1: frozen backbone
    # ============================================================

    freeze_backbone(model)

    optimizer = torch.optim.AdamW(
        filter(
            lambda parameter: parameter.requires_grad,
            model.parameters(),
        ),
        lr=config.training.frozen_learning_rate,
        weight_decay=config.training.weight_decay,
    )

    best_validation_score = 0.0

    for epoch in range(config.training.frozen_epochs):
        train_loss, train_accuracy = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        (
            validation_loss,
            validation_accuracy,
            validation_brier,
            validation_score,
        ) = validate(
            model=model,
            loader=validation_loader,
            criterion=criterion,
            device=device,
        )

        print(
            f"Frozen epoch {epoch + 1}/"
            f"{config.training.frozen_epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"train_acc={train_accuracy:.4f} | "
            f"val_loss={validation_loss:.4f} | "
            f"val_acc={validation_accuracy:.4f} | "
            f"val_brier={validation_brier:.4f} | "
            f"val_score={validation_score:.4f}"
        )

        if validation_score > best_validation_score:
            best_validation_score = validation_score

            save_checkpoint(
                model=model,
                config=config,
            )

    # ============================================================
    # Phase 2: fine-tuning
    # ============================================================

    unfreeze_backbone(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.finetune_learning_rate,
        weight_decay=config.training.weight_decay,
    )

    for epoch in range(config.training.finetune_epochs):
        train_loss, train_accuracy = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        (
            validation_loss,
            validation_accuracy,
            validation_brier,
            validation_score,
        ) = validate(
            model=model,
            loader=validation_loader,
            criterion=criterion,
            device=device,
        )

        print(
            f"Finetune epoch {epoch + 1}/"
            f"{config.training.finetune_epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"train_acc={train_accuracy:.4f} | "
            f"val_loss={validation_loss:.4f} | "
            f"val_acc={validation_accuracy:.4f} | "
            f"val_brier={validation_brier:.4f} | "
            f"val_score={validation_score:.4f}"
        )

        if validation_score > best_validation_score:
            best_validation_score = validation_score

            save_checkpoint(
                model=model,
                config=config,
            )

    print(
        f"Best validation score: "
        f"{best_validation_score:.4f}"
    )

    # ============================================================
    # Final test
    # ============================================================

    checkpoint_path = get_checkpoint_path(config)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    (
        test_loss,
        test_accuracy,
        test_brier,
        test_score,
    ) = validate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
    )

    print(
        f"Test | "
        f"loss={test_loss:.4f} | "
        f"accuracy={test_accuracy:.4f} | "
        f"brier={test_brier:.4f} | "
        f"score={test_score:.4f}"
    )
