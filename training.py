import os
import json
import shutil
import logging
from rich.logging import RichHandler
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
import torch
import matplotlib

matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter

from monai.data import Dataset, DataLoader, decollate_batch, load_decathlon_datalist
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import (
    AsDiscrete,
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    RandFlipd,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
)
from monai.utils import set_determinism

import architectures

import typer
from typing import Annotated
from pathlib import Path

app = typer.Typer()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[RichHandler()],
)
logger = logging.getLogger("mousetumor")


CONFIG = {
    # Data
    "num_workers": 4,
    # Training
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "max_epochs": 150,
    "batch_size": 4,
    "val_interval": 5,  # validate every N epochs
    "dice_weight": 0.75,
    "weight_decay": 1e-4,
    # Scheduler
    "learning_rate": 5e-4,
    "warm_restarts": True,
    "T_0": 50,
    "T_mult": 1,
    "lr_min": 1e-6,
    # Model
    "model_name": "unetpp_default",
    # Checkpointing
    "seed": 42,
}


def build_transforms(train: bool = True):
    load_transforms = [
        LoadImaged(keys=["image", "label"], reader="ITKReader"),
        EnsureChannelFirstd(keys=["image", "label"]),
        EnsureTyped(keys=["image", "label"]),
    ]

    augmentation_transforms = [
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandRotate90d(keys=["image", "label"], prob=0.1, max_k=3),
        RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
    ]

    if train:
        return Compose(load_transforms + augmentation_transforms)
    else:
        return Compose(load_transforms)


def build_dataloaders(train_files, val_files):
    train_ds = Dataset(data=train_files, transform=build_transforms(train=True))
    val_ds = Dataset(data=val_files, transform=build_transforms(train=False))
    train_loader = DataLoader(
        train_ds,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=CONFIG["num_workers"],
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
    )
    return train_loader, val_loader


def build_model(model_name, device):
    func = getattr(architectures, model_name)

    model = func().to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"{model_name} parameters: {total_params:,}")

    return model


def build_loss():
    return DiceCELoss(
        sigmoid=True,
        to_onehot_y=False,
        lambda_dice=CONFIG["dice_weight"],
        lambda_ce=1 - CONFIG["dice_weight"],
    )


def build_optimizer(model: torch.nn.Module):
    return torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
    )


def build_scheduler(optimizer):
    if CONFIG["warm_restarts"]:
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=CONFIG["T_0"],
            T_mult=CONFIG["T_mult"],
            eta_min=CONFIG["lr_min"],
        )
    else:
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=CONFIG["max_epochs"],
        )


def build_metric():
    dice_metric = DiceMetric(
        include_background=False, reduction="mean", ignore_empty=True
    )
    return dice_metric


def build_post_transforms():
    post_trans = Compose([AsDiscrete(threshold=0.5)])

    return post_trans


def train_one_epoch(model, loader, optimizer, loss_fn, device, epoch):
    model.train()
    epoch_loss = 0.0
    steps = 0

    with Progress(
        TextColumn("[bold blue]Epoch {task.fields[epoch]}[/]"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        "•",
        TimeElapsedColumn(),
        "•",
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("Training", total=len(loader), epoch=epoch)

        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(images)
            if isinstance(
                outputs, list
            ):  # unet++ returns list because of deep_supervision
                outputs = outputs[-1]

            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            steps += 1

            progress.update(task, advance=1)

    mean_loss = epoch_loss / steps
    logger.info(f"Epoch {epoch} | Train Loss: {mean_loss:.4f}")
    return mean_loss


def validate(model, loader, dice_metric, loss_fn, post_trans, device, epoch):
    model.eval()
    epoch_loss = 0.0
    steps = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            outputs = model(images)
            if isinstance(outputs, list):
                outputs = outputs[-1]

            loss = loss_fn(outputs, labels)

            outputs = [post_trans(i) for i in decollate_batch(outputs)]
            dice_metric(y_pred=outputs, y=labels)
            epoch_loss += loss
            steps += 1

    mean_dice = dice_metric.aggregate().item()
    mean_loss = epoch_loss / steps
    dice_metric.reset()

    logger.info(f"Epoch {epoch} | Val Dice: {mean_dice:.4f}")
    return mean_dice, mean_loss


def update_training_plot(
    output_dir: str,
    train_losses: list,
    val_metrics: list = None,
    filename: str = "training_plot.png",
):
    plot_path = os.path.join(output_dir, filename)

    fig, ax1 = plt.subplots(figsize=(10, 5), dpi=200)

    # Plot training loss on left y-axis
    ax1.plot(
        range(1, len(train_losses) + 1), train_losses, color="blue", label="Train Loss"
    )
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train Loss", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")
    ax1.grid(True, which="both", linestyle="--", alpha=0.3)

    # Plot validation metric on right y-axis
    if val_metrics:
        ax2 = ax1.twinx()
        val_epochs, val_values = zip(*val_metrics)
        ax2.plot(val_epochs, val_values, color="orange", label="Val Dice", marker="o")
        ax2.set_ylabel("Val Dice", color="orange")
        ax2.tick_params(axis="y", labelcolor="orange")

    # Legends
    lines, labels = ax1.get_legend_handles_labels()
    if val_metrics:
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines += lines2
        labels += labels2
    ax1.legend(lines, labels, loc="upper center")

    plt.title("Training Loss & Validation Dice")
    fig.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close(fig)


def train(data_dir, output_dir, datalist_name):
    if CONFIG["seed"] is not None:
        set_determinism(seed=CONFIG["seed"])
    os.makedirs(output_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(output_dir, "tensorboard"))
    device = torch.device(CONFIG["device"])

    train_files = load_decathlon_datalist(
        os.path.join(data_dir, datalist_name + ".json"),
        data_list_key="train",
        base_dir=data_dir,
    )
    val_files = load_decathlon_datalist(
        os.path.join(data_dir, datalist_name + ".json"),
        data_list_key="val",
        base_dir=data_dir,
    )

    train_loader, val_loader = build_dataloaders(train_files, val_files)

    model = build_model(CONFIG["model_name"], device)
    loss_fn = build_loss()
    optimizer = build_optimizer(model)
    scheduler = build_scheduler(optimizer)
    metric_fn = build_metric()
    post_trans = build_post_transforms()

    best_dice = -1.0
    best_epoch = 0

    train_losses = []
    val_metrics = []

    for epoch in range(1, CONFIG["max_epochs"] + 1):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, epoch
        )
        scheduler.step()
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("LR", scheduler.get_last_lr()[0], epoch)
        train_losses.append(train_loss)

        if epoch % CONFIG["val_interval"] == 0:
            val_dice, val_loss = validate(
                model, val_loader, metric_fn, loss_fn, post_trans, device, epoch
            )
            writer.add_scalar("Dice/val", val_dice, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)
            val_metrics.append((epoch, val_dice))

            if val_dice > best_dice:
                best_dice = val_dice
                best_epoch = epoch
                torch.save(
                    model.state_dict(),
                    os.path.join(output_dir, "best_model.pth"),
                )
                logger.info(f"  --> New best model saved (Dice: {best_dice:.4f})")

        writer.flush()
        update_training_plot(output_dir, train_losses, val_metrics)

    logger.info(f"Training complete. Best Dice: {best_dice:.4f} at epoch {best_epoch}")
    writer.close()


@app.command()
def run_training(
    data_dir: Annotated[Path, typer.Argument()],
    output_dir: Annotated[Path, typer.Argument()],
    datalist_name: Annotated[str, typer.Argument],
    max_epochs: Annotated[int, typer.Option("--n-epochs", "-e")] = CONFIG["max_epochs"],
    model_name: Annotated[str, typer.Option("--model_name", "-m")] = CONFIG[
        "model_name"
    ],
):
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    CONFIG["max_epochs"] = max_epochs
    CONFIG["model_name"] = model_name
    CONFIG["dataset"] = str(data_dir)
    CONFIG["datalist"] = datalist_name

    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(CONFIG, f, indent=2)

    datalist_path = os.path.join(data_dir, datalist_name + ".json")

    shutil.copy(datalist_path, os.path.join(output_dir, "datalist.json"))

    train(data_dir, output_dir, datalist_name)


if __name__ == "__main__":
    app()
