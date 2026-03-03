import os
import json

import numpy as np
import tifffile as tiff
import pandas as pd

import architectures
import torch

import napari

from rich.progress import (
    Progress,
    BarColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from training import build_transforms, build_post_transforms
from monai.metrics import DiceMetric
from monai.data import Dataset, DataLoader, decollate_batch, load_decathlon_datalist

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import seaborn as sns
import mplcursors

import typer
from typing import Annotated
from pathlib import Path

app = typer.Typer()


def load_model(model_name, checkpoint_path, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    func = getattr(architectures, model_name)

    model = func()

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    return model


def build_metric(config):
    # loss_fn = DiceCELoss(
    #     sigmoid=True,
    #     to_onehot_y=False,
    #     lambda_dice=config["dice_weight"],
    #     lambda_ce=1 - config["dice_weight"],
    # )
    metric_fn = DiceMetric(
        include_background=False, reduction="none", ignore_empty=True
    )

    return metric_fn


def build_val_loader(val_files, config):
    val_ds = Dataset(data=val_files, transform=build_transforms(train=False))
    val_loader = DataLoader(
        val_ds,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
    )
    return val_loader


def batch_prediction(model, loader, metric_fn, post_trans, device):
    model.eval()

    with torch.no_grad():
        with Progress(
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            "•",
            TimeElapsedColumn(),
            "•",
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task("Training", total=len(loader))
            for batch in loader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)

                outputs = model(images)
                if isinstance(outputs, list):
                    outputs = outputs[-1]

                outputs = [post_trans(i) for i in decollate_batch(outputs)]
                metric_fn(y_pred=outputs, y=labels)

                progress.update(task, advance=1)
    dice_scores = metric_fn.aggregate().cpu().numpy()

    metric_fn.reset()

    return dice_scores.reshape(-1)


def single_image_prediction(model, image, labels, post_trans, metric_fn, device):
    model.eval()

    image_tensor = torch.tensor(image).unsqueeze(0).unsqueeze(0).to(device)
    labels_tensor = torch.tensor(labels).unsqueeze(0).unsqueeze(0).float().to(device)

    with torch.no_grad():
        output = model(image_tensor)
    if isinstance(output, list):
        output = output[-1]
    output = torch.sigmoid(output)
    binary = post_trans(output)

    metric_fn(y_pred=binary, y=labels_tensor)
    dice_score = metric_fn.aggregate()
    dice_score = dice_score.cpu().numpy()
    binary = binary.cpu().numpy().astype(bool)
    metric_fn.reset()
    return binary, dice_score[0, 0]


def compute_volumes(file_paths, voxel_size=None):
    volumes = []
    for file in file_paths:
        labels = tiff.imread(file)

        vol = np.count_nonzero(labels)
        volumes.append(vol)

    volumes = np.array(volumes)
    if voxel_size is not None:
        volumes *= voxel_size**3

    return volumes


@app.command()
def dice_scatter_plot(
    run_path: Annotated[Path, typer.Argument()],
    load: Annotated[bool, typer.Option()] = False,
):
    print(matplotlib.get_backend())

    if load and os.path.isfile(os.path.join(run_path, "dice_scores.csv")):
        df = pd.read_csv(os.path.join(run_path, "dice_scores.csv"))

    else:
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        device = "cpu"

        with open(os.path.join(run_path, "config.json"), "r") as f:
            config = json.load(f)

        model_name = config["model_name"]

        model = load_model(model_name, os.path.join(run_path, "best_model.pth"), device)

        data_dir = config["dataset"]
        datalist_path = os.path.join(run_path, "datalist.json")

        val_files = load_decathlon_datalist(
            datalist_path,
            data_list_key="val",
            base_dir=data_dir,
        )

        val_loader = build_val_loader(val_files, config)

        metric_fn = build_metric(config)

        post_trans = build_post_transforms()

        dice_scores = batch_prediction(model, val_loader, metric_fn, post_trans, device)

        file_paths = [dict["label"] for dict in val_files]
        file_names = [
            os.path.basename(filepath).removesuffix(".tiff") for filepath in file_paths
        ]

        volumes = compute_volumes(file_paths)

        df = pd.DataFrame({"dice": dice_scores, "volume": volumes, "name": file_names})

        df.to_csv(os.path.join(run_path, "dice_scores.csv"), index=False)

    plt.figure()
    ax = sns.scatterplot(data=df, x="volume", y="dice")

    cursor = mplcursors.cursor(ax.collections[0], hover=True)

    @cursor.connect("add")
    def on_add(sel):
        sel.annotation.set_text(df["name"].iloc[sel.index])

    plt.show()


@app.command()
def show_prediction(
    run_path: Annotated[Path, typer.Argument()],
    img_name: Annotated[str, typer.Argument()],
):
    device = "cpu"
    with open(os.path.join(run_path, "config.json"), "r") as f:
        config = json.load(f)

    model_name = config["model_name"]

    model = load_model(model_name, os.path.join(run_path, "best_model.pth"), device)

    data_dir = config["dataset"]

    metric_fn = build_metric(config)

    post_trans = build_post_transforms()

    image_path = os.path.join(data_dir, "images", img_name)
    label_path = os.path.join(data_dir, "labels", img_name)

    image = tiff.imread(image_path)
    labels = tiff.imread(label_path)

    mask, dice_score = single_image_prediction(
        model, image, labels, post_trans, metric_fn, device
    )

    image_name = os.path.basename(image_path).removesuffix(".tiff")
    print(f"{image_name} DICE: {dice_score:.3f}")

    viewer = napari.Viewer()

    viewer.add_image(image, name="image", colormap="plasma")
    gt_layer = viewer.add_labels(labels, name="ground truth")
    pred_layer = viewer.add_labels(2 * mask, name="prediction")

    gt_layer.contour = 1
    pred_layer.contour = 1

    napari.run()


@app.command()
def show_random_prediction(
    run_path: Annotated[Path, typer.Argument()],
    n: Annotated[int, typer.Option("-n")] = 1,
):
    device = "cpu"
    with open(os.path.join(run_path, "config.json"), "r") as f:
        config = json.load(f)

    model_name = config["model_name"]

    model = load_model(
        model_name, os.path.join(run_path, "best_model_weights.pt"), device
    )

    data_dir = config["dataset"]
    datalist_path = os.path.join(run_path, "datalist.json")

    metric_fn = build_metric(config)

    post_trans = build_post_transforms()

    val_files = load_decathlon_datalist(
        datalist_path,
        data_list_key="val",
        base_dir=data_dir,
    )

    random_dicts = np.random.choice(val_files, size=n, replace=False)

    viewer = napari.Viewer()
    viewer.grid.enabled = True
    viewer.grid.stride = 3

    for i, dict in enumerate(random_dicts):
        image_name = os.path.basename(dict["image"]).removesuffix(".tiff")
        image = tiff.imread(dict["image"])
        labels = tiff.imread(dict["label"])

        mask, dice_score = single_image_prediction(
            model, image, labels, post_trans, metric_fn, device
        )
        print(f"{image_name} DICE: {dice_score}")

        viewer.add_image(image, name=image_name, colormap="plasma")
        gt_layer = viewer.add_labels(labels, name=f"{image_name} - GT")
        pred_layer = viewer.add_labels(2 * mask, name=f"{image_name} - pred")

        gt_layer.contour = 1
        pred_layer.contour = 1

    napari.run()


if __name__ == "__main__":
    app()
