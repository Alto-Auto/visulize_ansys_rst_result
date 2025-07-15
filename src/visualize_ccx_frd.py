#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
visualize_ccx_frd.py

Loads a CalculiX .frd file, converts to .vtu using ccx2paraview,
then visualizes:
- Total displacement
- Von Mises stress

Includes interactive probe: press 'a' to toggle, 'c' to clear all markers.
"""

# Limit OpenMP threads to avoid crashes
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pyvista as pv
from matplotlib.colors import ListedColormap
from ccx2paraview import Converter


# ------------------- CONVERSION -------------------
def convert_frd_to_vtu(frd_file):
    """
    Converts a CalculiX .frd file to .vtu files using ccx2paraview Python API.
    Returns the first timestep .vtu file path.
    """
    base_name = os.path.splitext(frd_file)[0]
    print(f"[INFO] Converting {frd_file} to VTU using ccx2paraview...")
    c = Converter(frd_file, ["vtu"])
    c.run()

    first_vtu = f"{base_name}.0.vtu"
    if os.path.exists(first_vtu):
        return first_vtu

    candidates = [f for f in os.listdir(os.path.dirname(frd_file)) if f.endswith(".vtu")]
    if candidates:
        return os.path.join(os.path.dirname(frd_file), candidates[0])

    raise FileNotFoundError("No .vtu files found after conversion!")


# ------------------- UTILS -------------------
def get_meaningful_min_idx(values, threshold=1e-10):
    filtered_indices = np.where(values > threshold)[0]
    return None if filtered_indices.size == 0 else filtered_indices[np.argmin(values[filtered_indices])]


# ------------------- PLOTTING -------------------
def plot_deformed_contour(grid, displacement, scalar_field, scalar_name, scale=1000, style="surface", show_undeformed=True):
    """
    Plots deformed shape colored by scalar_field (displacement magnitude or von Mises).
    Includes interactive probe mode: press 'a' to toggle, 'c' to clear all.
    """
    if displacement.shape[0] == 3:
        displacement = displacement.T

    warped = grid.copy()
    warped.point_data.set_vectors(displacement, name="displacement")
    warped = warped.warp_by_vector("displacement", factor=scale)
    warped.point_data[scalar_name] = scalar_field

    max_idx, min_idx = np.argmax(scalar_field), get_meaningful_min_idx(scalar_field)
    max_val, min_val = scalar_field[max_idx], scalar_field[min_idx]
    max_pt, min_pt = warped.points[max_idx], warped.points[min_idx]
    print(f"[INFO] {scalar_name}: Max={max_val:.4f} at {max_pt}, Min={min_val:.4f} at {min_pt}")

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
            plotter.add_mesh(grid, color="grey", style="surface", opacity=0.3, smooth_shading=True)

    plotter.add_mesh(warped, scalars=scalar_name, cmap=cmap, show_edges=True, lighting=False,
                     clim=[vmin, vmax],
                     scalar_bar_args=dict(title=scalar_name, fmt="%.2f", vertical=False))
    plotter.add_point_labels([max_pt, min_pt],
                             [f"Max: {max_val:.3f}", f"Min: {min_val:.3f}"],
                             font_size=10, point_size=10, text_color='black',
                             render_points_as_spheres=True, always_visible=True)

    # ----------- Interactive probe tool -----------
    picked_actors = []
    probe_mode = [False]

    def callback(point, picker):
        idx = warped.find_closest_point(point)
        value = scalar_field[idx]
        print(f"[PROBE] {point} nearest idx={idx} {scalar_name}={value:.4f}")
        sphere = pv.Sphere(radius=0.01, center=point)
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
            print("[INFO] Probe mode ENABLED. Click to probe values.")

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
    frd_path = r"C:\Users\altoa\AppData\Local\Temp\fcfem_glotntc0\FEMMeshGmsh.frd"
    vtu_path = convert_frd_to_vtu(frd_path)
    print(f"[INFO] Converted file: {vtu_path}")

    grid = pv.read(vtu_path)
    print(f"[INFO] Available point_data fields: {list(grid.point_data.keys())}")

    # Displacement
    if 'Displacement' in grid.point_data:
        displacement = grid.point_data['Displacement']
    elif 'U' in grid.point_data:
        displacement = grid.point_data['U']
    else:
        raise KeyError(f"No displacement field found in: {list(grid.point_data.keys())}")

    disp_mag = np.linalg.norm(displacement, axis=1)
    print(f"[INFO] Max displacement: {disp_mag.max():.6f} at node {np.argmax(disp_mag)}")
    plot_deformed_contour(grid, displacement, disp_mag, "Displacement", scale=1000, style="surface")

    # Von Mises stress
    stress_vm = None
    if 'von_Mises' in grid.point_data:
        stress_vm = grid.point_data['von_Mises']
    elif all(x in grid.point_data for x in ['Sxx', 'Syy', 'Szz', 'Sxy', 'Syz', 'Szx']):
        sx, sy, sz = grid.point_data['Sxx'], grid.point_data['Syy'], grid.point_data['Szz']
        txy, tyz, tzx = grid.point_data['Sxy'], grid.point_data['Syz'], grid.point_data['Szx']
        stress_vm = np.sqrt(0.5 * ((sx - sy)**2 + (sy - sz)**2 + (sz - sx)**2) + 3*(txy**2 + tyz**2 + tzx**2))
    elif 'S' in grid.point_data:
        S = grid.point_data['S']
        sx, sy, sz, txy, tyz, tzx = S[:,0], S[:,1], S[:,2], S[:,3], S[:,4], S[:,5]
        stress_vm = np.sqrt(0.5 * ((sx - sy)**2 + (sy - sz)**2 + (sz - sx)**2) + 3*(txy**2 + tyz**2 + tzx**2))
    else:
        print("[WARN] No stress fields found to compute von Mises.")
        stress_vm = np.zeros(len(disp_mag))

    print(f"[INFO] Max von Mises stress: {stress_vm.max():.6f}")
    plot_deformed_contour(grid, displacement, stress_vm, "von Mises", scale=1000, style="surface")


if __name__ == "__main__":
    main()
