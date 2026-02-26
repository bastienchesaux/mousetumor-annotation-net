from depalma_napari_omero.omero_client._project import (
    ProjectScanner,
    OmeroClient,
)

from mousetumorpy import LungsPredictor

from getpass import getpass, getuser
from tqdm import tqdm

import pandas as pd

import tifffile as tiff
import os

import typer
from typing import Annotated
from pathlib import Path


app = typer.Typer()


# =============================================================================
# Download data from omero server
# =============================================================================


def connect_to_omero():
    """
    prompts user for omero password and returns an omero client
    """
    user = getuser()
    password = getpass(f"Password for {user}: ")

    omero_client = OmeroClient(
        user=user,
        password=password,
    )

    omero_client.connect()

    return omero_client


def select_projects_prompt(omero_client) -> list[str]:
    """
    prompts user to select a set of experiments (omero projects) to download files
    from
    """
    projects = omero_client.projects
    typer.echo("\nAvailable projects:")
    keys = list(projects.keys())
    for i, key in enumerate(keys):
        typer.echo(f"  [{i}] {key} ({projects[key]} files)")

    typer.echo("\nEnter indices:")
    raw = input("> ")

    indices = [int(x) for x in raw.strip().split()]
    selected = [keys[i] for i in indices]

    typer.echo(f"\nSelected: {selected}")
    return selected


def scan_projects(project_names: list[str], omero_client):
    dfs = []
    for project_name in project_names:
        project_id = omero_client.projects.get(project_name)

        scanner = ProjectScanner(
            omero_client,
            project_id=project_id,
            project_name=project_name,
            launch_scan=True,
        )

        df = scanner.view.df

    dfs.append(df)

    full_df = pd.concat(dfs)

    return full_df


def filter_files(df: pd.DataFrame) -> pd.DataFrame:
    def filter_groups(group: pd.DataFrame) -> pd.DataFrame:
        if {"corrected_pred", "roi"}.issubset(group["class"].unique()):
            return group[group["class"].isin(["corrected_pred", "roi"])]
        return group.iloc[0:0]  # if pair not found return empty group

    sub_df = df.groupby(["dataset_id", "time"], group_keys=False).apply(
        filter_groups, include_groups=False
    )
    print(sub_df)
    n_valid = sub_df["class"].eq("corrected_pred").sum()
    print(f"Number of valid pairs: {n_valid}")

    return sub_df


@app.command()
def download_scans(
    save_dir: Annotated[Path, typer.Argument()],
    skip_existing: Annotated[bool, typer.Option()],
):
    omero_client = connect_to_omero()

    projects = select_projects_prompt(omero_client)

    all_files_df = scan_projects(projects, omero_client)

    file_subset_df = filter_files(all_files_df)

    spreadsheet_path = os.path.join(save_dir, "scan.csv")

    if os.path.isfile(spreadsheet_path):
        old_spreadsheet = pd.read_csv(spreadsheet_path)
        file_subset_df = pd.concat([old_spreadsheet, file_subset_df])

    file_subset_df.to_csv(os.path.join(save_dir, "scan.csv"))

    for idx, row in tqdm(
        file_subset_df.iterrows(), desc="Downloading Images", total=len(file_subset_df)
    ):
        file_name = "_".join([row["specimen"], row["time_tag"].lower(), row["class"]])
        save_path = os.path.join(save_dir, file_name + ".tiff")
        if skip_existing and os.path.isfile(save_path):
            continue
        image_arr = omero_client.download_image(image_id=row["image_id"])

        tiff.imwrite(save_path, image_arr)

    omero_client.quit()


@app.command()
def generate_lung_masks(
    image_dir: Annotated[Path, typer.Argument()],
    model: Annotated[str, typer.Option()] = "v1",
):
    predictor = LungsPredictor(model)

    image_files = [file for file in os.listdir(image_dir) if file.endswith("roi.tiff")]

    for file in tqdm(image_files, desc="Constructing lungs masks"):
        image = tiff.imread(os.path.join(image_dir, file))
        lung_mask = predictor.fast_predict(image, skip_level=2)

        lung_filename = file.removesuffix("roi.tiff") + "lungs_mask.tiff"

        tiff.imwrite(os.path.join(image_dir, lung_filename), lung_mask.astype(bool))


if __name__ == "__main__":
    app()
