import logging

from tqdm import tqdm

import pandas as pd
import numpy as np

import tifffile as tiff
import os
import re
import json

import typer
from typing import Annotated
from pathlib import Path


import edt

from scipy.ndimage import distance_transform_cdt
from skimage.measure import block_reduce

from line_profiler import LineProfiler

app = typer.Typer()

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("mousetumor")


def full_scan_normalize_hu(image, lb=-1000, ub=500):
    """Scale intensity range using  typical HU range for soft tissue ([-1000, 500])"""
    normalized = (image - lb) / (ub - lb)
    normalized = np.clip(normalized, 0, 1)
    return normalized


def full_scan_normalize(image, clip_percentile=96):
    ub = np.percentile(image, clip_percentile)
    lb = image.min()

    normalized = (image - lb) / (ub - lb)
    normalized = np.clip(normalized, 0, 1)
    return normalized


def extract_binary_tight(labelled: np.ndarray, label: int, pad_width: int = 5):
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
    z1 = min(z1, labelled.shape[0])
    y0 = max(y0, 0)
    y1 = min(y1, labelled.shape[1])
    x0 = max(x0, 0)
    x1 = min(x1, labelled.shape[2])

    tight = labelled[z0:z1, y0:y1, x0:x1]
    binary = (tight == label).astype(bool)

    return binary, np.array([z0, y0, x0])


def noisy_tumor_center(binary, dist_quantile=0.96):
    dist = edt.edt(binary)

    thresh = np.quantile(dist[dist != 0], dist_quantile)
    if np.count_nonzero(dist > thresh) == 0:
        logger.info("invalid threshold, using default")
    thresh = min(thresh, dist.max() - 2)  # safety for very small tumors

    valid_coords = np.stack(np.where(dist > thresh), -1)

    noisy_center = valid_coords[np.random.choice(len(valid_coords))]

    return np.array(noisy_center)


def compute_downsample_ratio(
    binary,
    win_center,
    win_size,
    safety_margin_px=10,
    limit=3,
):
    object_coords = np.stack(np.nonzero(binary), axis=-1)

    min_bounds = np.min(object_coords, axis=0)
    max_bounds = np.max(object_coords, axis=0)

    max_extent = max((max_bounds - win_center).max(), (win_center - min_bounds).max())

    downsample = int(np.ceil(max_extent / (win_size // 2 + safety_margin_px)))

    if downsample > limit:
        raise RuntimeError("Tumor is too big")

    elif downsample > 1:
        logger.info(
            f"Tumor doesn't fit in {win_size}³ window: using {downsample * win_size}³"
        )
    return downsample


def extract_tumor_window(img, labels, target_label, win_size, dist_quantile=0.96):
    binary, offset = extract_binary_tight(labels, target_label)

    win_center = noisy_tumor_center(binary, dist_quantile=dist_quantile)

    ds_ratio = compute_downsample_ratio(binary, win_center, win_size)

    win_center += offset

    if labels[win_center[0], win_center[1], win_center[2]] != target_label:
        raise RuntimeError(f"Window center {win_center} outside of tumor")

    half = int(ds_ratio * (win_size // 2))

    img_padded = np.pad(img, half, mode="constant", constant_values=0)
    lbl_padded = np.pad(labels, half, mode="constant", constant_values=0)

    # center is shifted by half because of padding
    z, y, x = win_center + half

    img_win = img_padded[
        z - half : z + half,
        y - half : y + half,
        x - half : x + half,
    ]
    lbl_win = lbl_padded[
        z - half : z + half,
        y - half : y + half,
        x - half : x + half,
    ]

    lbl_win = lbl_win == target_label
    lbl_win = lbl_win.astype(bool)

    if ds_ratio > 1:
        img_win = block_reduce(img_win, block_size=ds_ratio, func=np.median)
        lbl_win = block_reduce(lbl_win, block_size=ds_ratio, func=np.max).astype(bool)

    return img_win, lbl_win


@app.command()
def generate_tumor_windows(
    source_scan_dir: Annotated[Path, typer.Argument()],
    dataset_dir: Annotated[Path, typer.Argument()],
    win_size: Annotated[int, typer.Argument()],
    skip_existing: Annotated[bool, typer.Option()] = False,
):
    if not os.path.isdir(dataset_dir):
        os.mkdir(dataset_dir)
        os.mkdir(os.path.join(dataset_dir, "images"))
        os.mkdir(os.path.join(dataset_dir, "labels"))

    scans_df = pd.read_csv(os.path.join(source_scan_dir, "scan.csv"))
    scans_df["time_tag"] = scans_df["time_tag"].str.lower()

    for (case, scan), group in tqdm(scans_df.groupby(["specimen", "time_tag"])):
        if not {"roi", "corrected_pred"}.issubset(group["class"].unique()):
            logger.warning(f"{case} {scan} not a valid file pair")
            continue

        img_name = "_".join([case, scan, "roi.tiff"])
        labels_name = "_".join([case, scan, "corrected_pred.tiff"])

        try:
            img = tiff.imread(os.path.join(source_scan_dir, img_name))
            labels = tiff.imread(os.path.join(source_scan_dir, labels_name))
        except Exception as e:
            logging.warning(f"Failed to load {case}_{scan} with: {e}")
            continue

        if img.shape != labels.shape:
            logger.warning(f"Skip {case} {scan}: image shapes don't match")
            continue

        img = full_scan_normalize(img)

        lp = LineProfiler()
        lp_wrapper = lp(extract_tumor_window)

        for i in np.unique(labels[labels != 0]):
            try:
                output_name = "_".join([case, scan, f"tum{i}.tiff"])
                img_path = os.path.join(dataset_dir, "images", output_name)
                lbl_path = os.path.join(dataset_dir, "labels", output_name)
                if (
                    skip_existing
                    and os.path.isfile(img_path)
                    and os.path.isfile(lbl_path)
                ):
                    continue

                img_win, label_win = lp_wrapper(img, labels, i, win_size)

                # img_win, label_win = extract_tumor_window(img, labels, i, win_size)

                tiff.imwrite(
                    img_path,
                    img_win.astype(np.float32),
                )
                tiff.imwrite(
                    lbl_path,
                    label_win.astype(np.uint8),
                )
            except Exception as e:
                logger.error(f"Failed to generate {output_name} with {e}")
    lp.dump_stats("output.lprof")


def random_empty_window_center(labels, mask, win_size):
    background = labels == 0

    chebyshev_dist = distance_transform_cdt(background)

    valid_win_locations = chebyshev_dist > win_size

    dist_to_edge = win_size // 2 + 1

    valid_win_locations[:dist_to_edge, :, :] = 0
    valid_win_locations[-dist_to_edge:, :, :] = 0

    valid_win_locations[:, :dist_to_edge, :] = 0
    valid_win_locations[:, -dist_to_edge:, :] = 0

    valid_win_locations[:, :, :dist_to_edge] = 0
    valid_win_locations[:, :, -dist_to_edge:] = 0

    valid_win_locations *= mask

    if np.count_nonzero(valid_win_locations) == 0:
        return None

    valid_coords = np.stack(np.where(valid_win_locations), -1)

    noisy_center = valid_coords[np.random.choice(len(valid_coords))]

    return noisy_center


def extract_empty_window(img, labels, lung_mask, win_size):
    win_center = random_empty_window_center(labels, lung_mask, win_size)

    if win_center is None:
        return None, None

    half = win_size // 2

    img_padded = np.pad(img, half, mode="constant", constant_values=0)
    lbl_padded = np.pad(labels, half, mode="constant", constant_values=0)

    # center is shifted by half due to padding
    z, y, x = win_center + half

    img_win = img_padded[
        z - half : z + half,
        y - half : y + half,
        x - half : x + half,
    ]
    lbl_win = lbl_padded[
        z - half : z + half,
        y - half : y + half,
        x - half : x + half,
    ]

    lbl_win = lbl_win != 0
    lbl_win = lbl_win.astype(bool)

    return img_win, lbl_win


@app.command()
def generate_empty_windows(
    source_scan_dir: Annotated[Path, typer.Argument()],
    dataset_dir: Annotated[Path, typer.Argument()],
    win_size: Annotated[int, typer.Argument()],
    n: Annotated[int, typer.Argument(help="number of windows to generate")],
):
    if not os.path.isdir(dataset_dir):
        os.mkdir(dataset_dir)
        os.mkdir(os.path.join(dataset_dir, "images"))
        os.mkdir(os.path.join(dataset_dir, "labels"))

    lung_files = [
        file for file in os.listdir(source_scan_dir) if file.endswith("lung_mask.tiff")
    ]

    valid_count = 0

    pbar = tqdm(total=n)

    while valid_count < n:
        random_file = np.random.choice(lung_files)

        pattern = r"(?P<case>C\d{5})_scan(?P<scan_no>\d+)_"

        match = re.search(pattern, random_file)
        if not match:
            logger.error(f"pattern not found in {random_file}")
            continue
        case = match.group("case")
        scan = "scan" + match.group("scan_no")

        img_name = "_".join([case, scan, "roi.tiff"])
        labels_name = "_".join([case, scan, "corrected_pred.tiff"])

        img = tiff.imread(os.path.join(source_scan_dir, img_name))
        labels = tiff.imread(os.path.join(source_scan_dir, labels_name))
        lung_mask = tiff.imread(os.path.join(source_scan_dir, random_file))

        img = full_scan_normalize(img)

        img_win, label_win = extract_empty_window(img, labels, lung_mask, win_size)

        if img_win is None:
            continue

        if np.count_nonzero(label_win) > 0:
            logger.error("window is not empty")
            continue

        output_name = "_".join([case, scan, f"empty{valid_count}.tiff"])

        tiff.imwrite(
            os.path.join(dataset_dir, "images", output_name), img_win.astype(np.float32)
        )
        tiff.imwrite(
            os.path.join(dataset_dir, "labels", output_name), label_win.astype(np.uint8)
        )

        valid_count += 1
        pbar.update(1)

    pbar.close()


def split_prompt():
    while True:
        raw = typer.prompt("Train Val Test (e.g. 80 10 10)")
        try:
            values = tuple(int(x) for x in raw.split())
            if len(values) != 3:
                raise ValueError
            if sum(values) != 100:
                typer.echo("Values must sum to 100, try again.")
                continue
            return values
        except ValueError:
            typer.echo("Please enter 3 integers, try again.")


def split_files(files, split):
    files = np.random.choice(files, len(files), replace=False)
    n = len(files)
    train_end = int(np.ceil(n * split[0] / 100))
    val_end = train_end + int(n * split[1] / 100)

    splits = [files[:train_end], files[train_end:val_end], files[val_end:]]
    labels = ["train", "val", "test"]

    return {f: label for files, label in zip(splits, labels) for f in files}


@app.command()
def generate_datalist(
    dataset_dir: Annotated[Path, typer.Argument()],
    n_files: Annotated[int | None, typer.Option("--n-files", "-n")] = None,
    name: Annotated[str, typer.Option()] = "datalist.json",
):
    split_percent = split_prompt()

    datalist = {"train": [], "val": [], "test": []}

    tumor_files = [
        file
        for file in os.listdir(os.path.join(dataset_dir, "images"))
        if "tum" in file
    ]
    empty_files = [
        file
        for file in os.listdir(os.path.join(dataset_dir, "images"))
        if "empty" in file
    ]

    tumor_fraction = len(tumor_files) / (len(tumor_files) + len(empty_files))

    typer.echo(f"Tumor fraction: {100 * tumor_fraction:.1f}%")

    if n_files is not None:
        n_tumor_files = int(np.ceil(n_files * tumor_fraction))
        n_empty_files = n_files - n_tumor_files

        tumor_files = np.random.choice(tumor_files, size=n_tumor_files, replace=False)
        empty_files = np.random.choice(empty_files, size=n_empty_files, replace=False)

    tumor_files_split = split_files(tumor_files, split_percent)
    empty_files_split = split_files(empty_files, split_percent)

    files_split = {**tumor_files_split, **empty_files_split}

    for file, split_dest in files_split.items():
        datalist[split_dest].append(
            {
                "image": os.path.join("images", file),
                "label": os.path.join("labels", file),
            }
        )

    with open(os.path.join(dataset_dir, name), "w") as f:
        json.dump(datalist, f, indent=2)


if __name__ == "__main__":
    app()
