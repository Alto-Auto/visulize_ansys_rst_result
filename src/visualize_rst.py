#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
visualize_ansys_rst.py

Load and visualize an ANSYS MAPDL .rst file:
- Plots deformed shape colored by displacement magnitude
- Plots von Mises stress
- Interactive probe: press 'a' to toggle, 'c' to clear markers
"""

import os
# Limit OpenMP threads to avoid crashes
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


import numpy as np
import pyvista as pv
from ansys.mapdl import reader
from matplotlib.colors import ListedColormap
from ansys.dpf import core as dpf


# ------------------- CONFIG -------------------
RST_PATH = "your_rst_file"
SCALE = 1  # deformation scaling


# ------------------- CORE FUNCTIONS -------------------
def extract_stiffness_energy_from_rst(rst_path):
    model = dpf.Model(rst_path)
    op = model.results.stiffness_matrix_energy()
    fields = op.eval()
    total_energy = sum(sum(field.data) for field in fields)
    return total_energy


def get_meaningful_min_idx(values, threshold=1e-10):
    filtered_indices = np.where(values > threshold)[0]
    return None if filtered_indices.size == 0 else filtered_indices[np.argmin(values[filtered_indices])]


def extract_displacement(result, load_step=0):
    nnum, displacement = result.nodal_displacement(load_step)
    magnitude = np.linalg.norm(displacement, axis=1)
    max_node = nnum[np.argmax(magnitude)]
    print(f"[INFO] Max displacement: {magnitude.max():.6f} at node {max_node}")
    return displacement, magnitude


def extract_von_mises_stress(result, load_step=0):
    nnum, stress = result.nodal_stress(load_step)
    stress = np.nan_to_num(stress)
    sx, sy, sz, txy, tyz, tzx = stress[:,0], stress[:,1], stress[:,2], stress[:,3], stress[:,4], stress[:,5]
    von_mises = np.sqrt(0.5 * ((sx - sy)**2 + (sy - sz)**2 + (sz - sx)**2) + 3*(txy**2 + tyz**2 + tzx**2))
    max_node = nnum[np.argmax(von_mises)]
    print(f"[INFO] Max von Mises stress: {von_mises.max():.6f} at node {max_node}")
    return von_mises


# ------------------- PLOTTING -------------------
def plot_deformed_contour(grid, displacement, scalar_field, scalar_name, scale=SCALE, style="surface", show_undeformed=True):
    if displacement.shape[0] == 3:
        displacement = displacement.T

    deformed_grid = grid.copy()
    deformed_grid.point_data.set_vectors(displacement, name="displacement")
    deformed_grid = deformed_grid.warp_by_vector("displacement", factor=scale)
    deformed_grid.point_data[scalar_name] = scalar_field

    max_idx = np.argmax(scalar_field)
    min_idx = get_meaningful_min_idx(scalar_field)
    max_val, min_val = scalar_field[max_idx], scalar_field[min_idx]
    max_point, min_point = deformed_grid.points[max_idx], deformed_grid.points[min_idx]
    print(f"[INFO] {scalar_name}: Max={max_val:.4f} at {max_point}, Min={min_val:.4f} at {min_point}")

    vmin, vmax = np.percentile(scalar_field, 0), np.percentile(scalar_field, 100)
    cmap = ListedColormap(['blue', 'royalblue', 'cyan', '#00FA9A', '#7CFC00', 
                           '#ADFF2F', 'yellow', 'orange', 'red']).with_extremes()

    plotter = pv.Plotter()
    edges = grid.extract_feature_edges()
    if show_undeformed:
        if style == "wireframe":
            plotter.add_mesh(grid, style="wireframe", opacity=0.3, color="grey", show_edges=True)
        elif style == "edge":
            plotter.add_mesh(edges, color="black", line_width=1.5)
        else:
            plotter.add_mesh(grid, color="gray", style="surface", opacity=0.3, smooth_shading=True)

    plotter.add_mesh(deformed_grid, scalars=scalar_name, cmap=cmap, show_edges=True, lighting=False,
                     clim=[vmin, vmax],
                     scalar_bar_args=dict(title=scalar_name, fmt="%.2f", vertical=False))
    plotter.add_point_labels([max_point, min_point],
                             [f"Max: {max_val:.3f}", f"Min: {min_val:.3f}"],
                             font_size=10, point_size=10, text_color='black',
                             render_points_as_spheres=True, always_visible=True)

    # Interactive probe tool
    picked_actors = []
    probe_mode = [False]

    def callback(point, picker):
        idx = deformed_grid.find_closest_point(point)
        value = scalar_field[idx]
        print(f"[PROBE] {point} nearest idx={idx} {scalar_name}={value:.4f}")
        sphere = pv.Sphere(radius=0.01*scale, center=point)
        actor_sphere = plotter.add_mesh(sphere, color="black", opacity=0.8)
        actor_label = plotter.add_point_labels([point], [f"{scalar_name}: {value:.4f}"],
                                               font_size=10, point_size=8,
                                               text_color="black", shape_opacity=0.5,
                                               always_visible=True)
        picked_actors.extend([actor_sphere, actor_label])

    def toggle_picking():
        if probe_mode[0]:
            plotter.disable_picking()
            probe_mode[0] = False
            print("[INFO] Probe mode DISABLED.")
        else:
            plotter.enable_point_picking(callback=callback, show_message=True, use_picker=True, show_point=False)
            probe_mode[0] = True
            print("[INFO] Probe mode ENABLED. Click on mesh to probe values.")

    def clear_picks():
        print("[INFO] Clearing all probe markers...")
        for actor in picked_actors:
            plotter.remove_actor(actor)
        picked_actors.clear()

    plotter.add_key_event("a", toggle_picking)
    plotter.add_key_event("c", clear_picks)
    plotter.add_text("a = toggle probe || c = clear all", position="lower_left", font_size=7, color="black", shadow=True)
    plotter.show()


# ------------------- MAIN -------------------
def main():
    print("[INFO] Loading ANSYS .rst result file...")
    result = reader.read_binary(RST_PATH)
    grid = result.grid

    displacement, disp_mag = extract_displacement(result)
    plot_deformed_contour(grid, displacement, disp_mag, "Displacement", style="surface")

    von_mises = extract_von_mises_stress(result)
    plot_deformed_contour(grid, displacement, von_mises, "von Mises", style="surface")


if __name__ == "__main__":
    main()
