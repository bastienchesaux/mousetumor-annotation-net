import logging

import matplotlib.pyplot as plt
import seaborn as sns
import napari

from tqdm import tqdm

import pandas as pd
import numpy as np

import tifffile as tiff
import os
import re

import typer
from typing import Annotated
from pathlib import Path

from mesh_viewer import MeshViewer

import edt

from scipy.ndimage import distance_transform_cdt


app = typer.Typer()

# =============================================================================
# Mesh viewer and size analysis
# =============================================================================


def select_imgs_prompt(image_dir) -> list[str]:
    """
    prompts user to select a set of experiments (omero projects) to download files
    from
    """
    available_imgs = [
        file for file in os.listdir(image_dir) if file.endswith("corrected_pred.tiff")
    ]
    typer.echo("\nAvailable projects:")
    for i, name in enumerate(available_imgs):
        typer.echo(f"  [{i}] {name}")

    typer.echo("\nEnter indices:")
    raw = input("> ")

    indices = [int(x) for x in raw.strip().split()]
    selected = [available_imgs[i] for i in indices]

    typer.echo(f"\nSelected: {selected}")
    return selected


class MaxDisp(Exception):
    pass


@app.command()
def visualize_tumors(
    image_dir: Annotated[Path, typer.Argument()],
    max_display: Annotated[
        int, typer.Option(help="max number of tumors to add to the viewer")
    ] = 60,
    unit: Annotated[str, typer.Option(help="px or um")] = "px",
):
    image_names = select_imgs_prompt(image_dir)

    if unit == "px":
        vsize = -1
    elif unit == "um":
        vsize = 40.864
    else:
        logging.error('Unvalid unit, please choose between "um" and "px"')

    viewer = MeshViewer(smooth_iter=0, voxel_size=vsize)

    n_tumors = 0

    try:
        for filename in image_names:
            labels = tiff.imread(os.path.join(image_dir, filename))

            pattern = r"(?P<case>C\d{5})_scan(?P<scan_no>\d+)_"

            match = re.search(pattern, filename)

            if not match:
                logging.error(f"pattern not found in {filename}")
                continue
            case = match.group("case")
            scan_no = match.group("scan_no")

            for i in tqdm(
                np.unique(labels[labels != 0]), desc=f"Preparing {case}_{scan_no}"
            ):
                mesh_name = f"{case}_sc{int(scan_no)}_{i}"
                viewer.add_mesh(labels, i, mesh_name)

                if n_tumors > max_display:
                    raise MaxDisp
    except MaxDisp:
        pass

    viewer.show()


def measure_tumors_extents(image_dir):
    df = pd.DataFrame(columns=["case", "scan", "label", "extents"])

    label_files = [
        file for file in os.listdir(image_dir) if file.endswith("corrected_pred.tiff")
    ]

    for filename in tqdm(label_files):
        if filename.endswith("corrected_pred.tiff"):
            labels = tiff.imread(os.path.join(image_dir, filename))

            pattern = r"(?P<case>C\d{5})_scan(?P<scan_no>\d+)_"

            match = re.search(pattern, filename)

            if not match:
                logging.error(f"pattern not found in {filename}")
                continue
            case = match.group("case")
            scan_no = match.group("scan_no")

            for i in np.unique(labels[labels != 0]):
                tumor_pixels = np.where(labels == i)

                x_extents = tumor_pixels[0].max() - tumor_pixels[0].min()
                y_extents = tumor_pixels[1].max() - tumor_pixels[1].min()
                z_extents = tumor_pixels[2].max() - tumor_pixels[2].min()

                new_row = {
                    "case": case,
                    "scan_no": scan_no,
                    "label": i,
                    "extents": (x_extents, y_extents, z_extents),
                }

                df.loc[len(df)] = new_row

    return df


@app.command()
def tumor_extents_histogram(image_dir: Annotated[Path, typer.Argument()]):
    df = measure_tumors_extents(image_dir)

    print(f"Scans contain {len(df)} tumors in total")

    df["min_extent"] = df["extents"].apply(min)
    df["max_extent"] = df["extents"].apply(max)

    min_p2 = df["min_extent"].quantile(0.02)
    min_p98 = df["min_extent"].quantile(0.98)
    max_p2 = df["max_extent"].quantile(0.02)
    max_p98 = df["max_extent"].quantile(0.98)

    fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)

    sns.histplot(data=df, x="min_extent", stat="percent", ax=axs[0])
    axs[0].axvline(x=min_p2, color="r")
    axs[0].axvline(x=min_p98, color="r")
    sns.histplot(data=df, x="max_extent", stat="percent", ax=axs[1])
    axs[1].axvline(x=max_p2, color="r")
    axs[1].axvline(x=max_p98, color="r")

    fig.suptitle("Histograms of min. and max. extents of the tumors")

    plt.show()


# =============================================================================
# Testing methods for simulating user-placed point
# =============================================================================


def extract_binary_tight(labelled: np.ndarray, label: int, pad_width: int = 3):
    """
    Extracts the binary image of a target label with minimal dimensions.

    Args:
    - labelled (np.ndarray): input labelled image
    - label (int): target label
    - pad_width (int): number of layers of zero-padding around the cuboid in the binary output

    Returns:
    - binary (np.ndarray): tight binary image of the target label
    - z0, z1, y0, y1, x0, x1 (int): x, y, and z position of the binary image in the original labelled image
    """

    if not np.any(labelled == label):
        print(f"Label{label} doesn't appear in provided image")
        return None

    z, y, x = np.where(labelled == label)

    # padding is necessary for mesh generation, the cuboid can't touch the border of the binary image
    z0, z1 = z.min() - pad_width, z.max() + pad_width + 1
    y0, y1 = y.min() - pad_width, y.max() + pad_width + 1
    x0, x1 = x.min() - pad_width, x.max() + pad_width + 1

    z0 = max(z0, 0)
    z1 = min(z1, labelled.shape[0] + 1)
    y0 = max(y0, 0)
    y1 = min(y1, labelled.shape[1] + 1)
    x0 = max(x0, 0)
    x1 = min(x1, labelled.shape[2] + 1)

    tight = labelled[z0:z1, y0:y1, x0:x1]
    binary = (tight == label).astype(bool)

    return binary


def threshold_edt(binary):
    dist = edt.edt(binary)

    thresh = np.percentile(dist[dist != 0], 95)

    plt.figure()
    plt.hist(x=dist[dist != 0], bins=20)
    plt.axvline(thresh, color="r")
    plt.show()

    eroded = dist > thresh

    viewer = napari.Viewer()

    viewer.add_image(dist, name="edt")
    viewer.add_labels(binary.astype(int), name="original")
    viewer.add_labels(3 * eroded.astype(int), name="core")

    napari.run()


@app.command()
def demo_tumor_edt_erosion(
    image_dir: Annotated[Path, typer.Argument()],
    seed: Annotated[
        int | None, typer.Option(help="seed for random selection of a tumor")
    ] = None,
):
    if seed is not None:
        np.random.seed(seed)

    label_files = [
        file for file in os.listdir(image_dir) if file.endswith("corrected_pred.tiff")
    ]

    random_file = np.random.choice(label_files)

    labels = tiff.imread(os.path.join(image_dir, random_file))

    tumor_idx = np.random.choice(np.unique(labels[labels != 0]))

    print(f"Randomly selected tumor {tumor_idx} in {random_file}")

    binary = extract_binary_tight(labels, tumor_idx, pad_width=5)

    threshold_edt(binary)


@app.command()
def demo_find_empty_window(
    image_dir: Annotated[Path, typer.Argument()],
    winsize: Annotated[int, typer.Argument()],
    seed: Annotated[
        int | None, typer.Option(help="seed for random selection of a tumor")
    ] = None,
    dist_to_edge: Annotated[int, typer.Option()] = 0,
):
    if seed is not None:
        np.random.seed(seed)

    label_files = [
        file for file in os.listdir(image_dir) if file.endswith("corrected_pred.tiff")
    ]

    random_file = np.random.choice(label_files)

    labels = tiff.imread(os.path.join(image_dir, random_file))

    background = labels == 0

    chebyshev_dist = distance_transform_cdt(background)

    valid_win_locations = chebyshev_dist > winsize

    dist_to_edge = min(winsize // 2, dist_to_edge)

    valid_win_locations[:dist_to_edge, :, :] = 0
    valid_win_locations[-dist_to_edge:, :, :] = 0

    valid_win_locations[:, :dist_to_edge, :] = 0
    valid_win_locations[:, -dist_to_edge:, :] = 0

    valid_win_locations[:, :, :dist_to_edge] = 0
    valid_win_locations[:, :, -dist_to_edge:] = 0

    viewer = napari.Viewer()

    viewer.add_image(chebyshev_dist, name="dist")
    viewer.add_labels(labels, name="labels")
    viewer.add_labels(3 * valid_win_locations.astype(int), name="valid_windows")

    napari.run()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.WARNING,  # DEBUG, INFO, WARNING, ERROR, CRITICAL
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger("matplotlib").setLevel(logging.ERROR)
    app()
