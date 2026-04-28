"""
Usage:
    python3 -m homework.train_planner --your_args here
"""
import argparse

import torch
import torch.nn as nn

from .datasets.road_dataset import load_data
from .models import MODEL_FACTORY, save_model
from .metrics import PlannerMetric

CNN_MODELS = {"cnn_planner"}


def train(
    model_name: str = "mlp_planner",
    num_epoch: int = 50,
    lr: float = 1e-3,
    batch_size: int = 128,
    **model_kwargs,
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    use_image = model_name in CNN_MODELS
    pipeline = "default" if use_image else "state_only"

    train_loader = load_data(
        "drive_data/train",
        transform_pipeline=pipeline,
        return_dataloader=True,
        num_workers=2,
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = load_data(
        "drive_data/val",
        transform_pipeline=pipeline,
        return_dataloader=True,
        num_workers=2,
        batch_size=batch_size,
        shuffle=False,
    )

    model = MODEL_FACTORY[model_name](**model_kwargs).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epoch)
    loss_fn = nn.L1Loss(reduction="none")

    print(f"Training {model_name} for {num_epoch} epochs | pipeline={pipeline}")

    for epoch in range(num_epoch):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            waypoints = batch["waypoints"].to(device)
            waypoints_mask = batch["waypoints_mask"].to(device)

            if use_image:
                preds = model(image=batch["image"].to(device))
            else:
                preds = model(
                    track_left=batch["track_left"].to(device),
                    track_right=batch["track_right"].to(device),
                )

            loss = loss_fn(preds, waypoints)
            loss = loss * waypoints_mask[..., None]
            loss = loss.sum() / (waypoints_mask.sum() * 2 + 1e-8)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        model.eval()
        metric = PlannerMetric()

        with torch.no_grad():
            for batch in val_loader:
                waypoints = batch["waypoints"].to(device)
                waypoints_mask = batch["waypoints_mask"].to(device)

                if use_image:
                    preds = model(image=batch["image"].to(device))
                else:
                    preds = model(
                        track_left=batch["track_left"].to(device),
                        track_right=batch["track_right"].to(device),
                    )

                metric.add(preds, waypoints, waypoints_mask)

        results = metric.compute()
        avg_loss = total_loss / len(train_loader)

        print(
            f"Epoch {epoch+1:3d}/{num_epoch} | "
            f"loss={avg_loss:.4f} | "
            f"long={results['longitudinal_error']:.4f} | "
            f"lat={results['lateral_error']:.4f}"
        )

    save_model(model)
    print("Model saved.")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="mlp_planner")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    train(
        model_name=args.model,
        num_epoch=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
    )
