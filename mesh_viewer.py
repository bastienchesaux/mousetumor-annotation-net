import math

import numpy as np
import pyvista as pv

import trimesh

import logging

from skimage.morphology import remove_small_objects
from skimage.measure import marching_cubes


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

    return binary, z0, z1, y0, y1, x0, x1


def generate_trimesh(binary):
    """
    Generate a surface mesh from a tight binary image using Lewiner marching cubes.
    Use standard mesh fixing operation if the resulting mesh is not watertight.

    Args:
    - binary (np.ndarray): tight binary image
    """

    binary = binary.astype(bool)  # safety check

    binary = remove_small_objects(
        binary, min_size=100
    )  # remove noise to avoid watertighness issues
    try:
        verts, faces, _, _ = marching_cubes(binary)
    except RuntimeError:
        return
    mesh = trimesh.Trimesh(verts, faces)
    mesh.vertices -= mesh.center_mass
    trimesh.repair.fix_inversion(mesh)
    if not mesh.is_watertight:
        trimesh.repair.fill_holes(mesh)
        if not mesh.is_watertight:
            print("Mesh for cuboid is not watertight")
    return mesh


def postprocess_mesh(mesh: trimesh.Trimesh, decimate_percent=0, smooth_iter=3):
    if smooth_iter > 0:
        trimesh.smoothing.filter_taubin(
            mesh, lamb=0.5, nu=0.53, iterations=2 * smooth_iter
        )

    if decimate_percent > 0:
        mesh = mesh.simplify_quadric_decimation(percent=decimate_percent)

    faces = np.hstack(
        [np.full((mesh.faces.shape[0], 1), 3, dtype=np.int64), mesh.faces]
    )
    pv_mesh = pv.PolyData(mesh.vertices, faces)
    pv_mesh.compute_normals(
        auto_orient_normals=True,
        cell_normals=False,
        point_normals=True,
        inplace=True,
    )

    return pv_mesh


class MeshViewer:
    def __init__(
        self,
        meshes=[],
        extents=[],
        mesh_names=[],
        smooth_iter=3,
        decimation_percent=0,
        init_page=0,
        grid_size=2,
        voxel_size=40.864,
    ):
        self.meshes = meshes
        self.extents = extents
        self.names = mesh_names
        self.smooth_iter = smooth_iter
        self.decimation = decimation_percent
        self.current_page = init_page
        self.grid_size = grid_size
        self.voxel_size = voxel_size

        if voxel_size > 0:
            self.unit = "um"
        else:
            self.unit = "px"

        self.plotter = pv.Plotter(shape=(grid_size, grid_size), border=True)

        self.plotter.add_key_event("Right", self.next_page)
        self.plotter.add_key_event("Left", self.previous_page)

    def add_mesh(self, labels_img, target_label, name=None):
        if name is None:
            name = f"tumor{len(self.meshes)}"

        binary = extract_binary_tight(labels_img, target_label)[0]

        mesh = generate_trimesh(binary)

        if mesh is None:
            logging.warning(f"Failed to construct mesh {name}")
            return

        pv_mesh = postprocess_mesh(mesh, self.decimation, self.smooth_iter)

        self.meshes.append(pv_mesh)

        if self.unit == "um":
            self.extents.append(mesh.extents * self.voxel_size)
        else:
            self.extents.append(mesh.extents)

        self.names.append(name)

    def slice_meshes(self):
        per_page = self.grid_size**2
        start = self.current_page * per_page
        end = min(start + per_page, len(self.meshes))
        return self.meshes[start:end], self.extents[start:end], self.names[start:end]

    def update_plotter(self):
        self.plotter.clear()

        page_meshes, page_extents, page_names = self.slice_meshes()

        for idx, mesh in enumerate(page_meshes):
            row, col = divmod(idx, self.grid_size)
            self.plotter.subplot(row, col)
            self.plotter.remove_all_lights()
            self.plotter.add_light(pv.Light(light_type="headlight", intensity=1.0))
            self.plotter.add_mesh(
                mesh, smooth_shading=True, specular=0.5, ambient=0.3, diffuse=0.8
            )

            text = f"{page_names[idx]}\n"
            text += f"Extents:\nx={page_extents[idx][0]:.0f}{self.unit}\n"
            text += f"y={page_extents[idx][1]:.0f}{self.unit}\n"
            text += f"z={page_extents[idx][2]:.0f}{self.unit}"
            self.plotter.add_text(text=text, position="upper_left", font_size=8)

        self.plotter.link_views()
        self.plotter.reset_camera()
        self.plotter.render()

    def next_page(self):
        self.current_page += 1
        if self.current_page > math.ceil(len(self.meshes) / self.grid_size**2) - 1:
            self.current_page = 0
        elif self.current_page < 0:
            self.current_page = math.ceil(len(self.meshes) // self.grid_size**2) - 1

        self.update_plotter()

    def previous_page(self):
        self.current_page -= 1
        if self.current_page > math.ceil(len(self.meshes) / self.grid_size**2) - 1:
            self.current_page = 0
        elif self.current_page < 0:
            self.current_page = math.ceil(len(self.meshes) // self.grid_size**2) - 1
        self.update_plotter()

    def show(self):
        self.update_plotter()
        self.plotter.show()
