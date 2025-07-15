import numpy as np
import pyvista as pv
from ansys.mapdl import reader
from matplotlib.colors import ListedColormap
from ansys.dpf import core as dpf


import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"    #prevent crash

def extract_stiffness_energy_from_rst(rst_path):

    model = dpf.Model(rst_path)

    # Access global stiffness energy via correct result operator
    op = model.results.stiffness_matrix_energy()
    fields = op.eval()

    total_energy = 0.0
    for field in fields:
        total_energy += sum(field.data)

    return total_energy

def get_meaningful_min_idx(values, threshold=1e-10000):
    """
    Returns the index of the smallest *non-trivial* value.
    Trivial means values below the threshold (default near-zero).
    """
    filtered_indices = np.where(values > threshold)[0]
    if filtered_indices.size == 0:
        return None  # All values too small
    min_idx = filtered_indices[np.argmin(values[filtered_indices])]
    return min_idx


def plot_deformed_contour(grid: pv.UnstructuredGrid,
                          displacement: np.ndarray,
                          scalar_field: np.ndarray,
                          scalar_name: str,
                          scale: float = 100000,
                          style = "surface",
                          show_undeformed: bool = True):
    """
    Plot a deformed mesh overlaid with a scalar field (e.g., displacement magnitude or stress).

    Parameters:
        grid (pv.UnstructuredGrid): Original mesh from the result file.
        displacement (np.ndarray): Nodal displacement vectors (N, 3).
        scalar_field (np.ndarray): Scalar values to color the deformed mesh (N,).
        scalar_name (str): Label for scalar field (used in legend and color bar).
        scale (float): Scale factor for visualizing deformation.
        show_undeformed (bool): Whether to overlay the undeformed wireframe/surface/edge.
    """
    if displacement.shape[0] == 3:
        displacement = displacement.T

    # Warp mesh
    deformed_grid = grid.copy()
    deformed_grid.point_data.set_vectors(displacement, name="displacement")
    deformed_grid = deformed_grid.warp_by_vector("displacement", factor=scale)
    deformed_grid.point_data[scalar_name] = scalar_field


    # Find min/max values and locations
    max_idx = np.argmax(scalar_field)
    min_idx = get_meaningful_min_idx(scalar_field)
    max_val = scalar_field[max_idx]
    min_val = scalar_field[min_idx]
    max_point = deformed_grid.points[max_idx]
    min_point = deformed_grid.points[min_idx]
    print(max_point,min_point)
    vmin = np.percentile(scalar_field, 0)
    vmax = np.percentile(scalar_field, 100)

    # Plot
    plotter = pv.Plotter()
    edges = grid.extract_feature_edges()
    if show_undeformed:
        if style == "wireframe":
            plotter.add_mesh(grid,style = "wireframe",opacity=0.3,smooth_shading=True,show_vertices=False, color="grey")
        elif style == "edge":
            plotter.add_mesh(edges, color="black", line_width=1.5)
        elif style == "surface":
            plotter.add_mesh(grid, color="gray", style="surface", opacity=0.3,smooth_shading=True,show_vertices=False)


    cmap1 = ListedColormap(['blue', 'royalblue', 'cyan', '#00FA9A', '#7CFC00', '#ADFF2F', 'yellow', 'orange', 'red']).with_extremes()
    plotter.add_mesh(deformed_grid, scalars=scalar_name, cmap=cmap1, show_edges=True,clim=[vmin, vmax], lighting=False,reset_camera=False,
        scalar_bar_args=dict(
        title=scalar_name,
        n_labels=len(cmap1.colors),
        n_colors=len(cmap1.colors),
        fmt="%.2f",
        vertical=False
    ))
    plotter.add_axes(interactive=True)

    # Add markers at min/max points
    label_points = np.array([max_point, min_point])
    label_texts = [f"Max: {max_val:.3f}", f"Min: {min_val:.3f}"]
    plotter.add_point_labels(label_points, label_texts,
                             font_size=11,
                             text_color='black',
                             point_size=13,
                             render_points_as_spheres=True,
                             always_visible=True)

    # ----------- Interactive probe toggle with 'a' key -----------

    picked_actors = []
    probe_mode = [False]  # use list for mutability inside closures

    def callback(point, picker):
        idx = deformed_grid.find_closest_point(point)
        value = scalar_field[idx]
        print(f"Clicked at {point}, nearest idx: {idx}, {scalar_name} = {value:.4f}")

        # Add tiny sphere marker
        sphere = pv.Sphere(radius=0.01 * scale, center=point)
        actor_sphere = plotter.add_mesh(sphere, color="black", opacity=0.8)

        # Add floating label
        actor_label = plotter.add_point_labels([point], [f"{scalar_name}: {value:.4f}"],
                                               font_size=10, point_size=8,
                                               text_color="black", shape_opacity=0.5,
                                               always_visible=True)

        picked_actors.append(actor_sphere)
        picked_actors.append(actor_label)

    def toggle_picking():
        if probe_mode[0]:
            # currently on, turn off
            plotter.disable_picking()
            probe_mode[0] = False
            print("Interactive picking DISABLED. Press 'a' again to enable.")
        else:
            # currently off, turn on
            plotter.enable_point_picking(callback=callback, show_message=True, use_picker=True, show_point=False)
            probe_mode[0] = True
            print("Interactive picking ENABLED. Click to probe values. Press 'a' to disable.")

    plotter.add_key_event("a", toggle_picking)

    # ----------- Clear all with 's' -----------
    def clear_picks():
        print("Clearing all probe markers...")
        for actor in picked_actors:
            plotter.remove_actor(actor)
        picked_actors.clear()

    plotter.add_key_event("c", clear_picks)

    plotter.add_text(
        "a = activate probe ||  c = clear all",
        position="lower_left",
        font_size=7,
        color="black",
        shadow=True,
    )

    plotter.show()

def extract_displacement(result, load_step=0):
    nnum, displacement = result.nodal_displacement(load_step)
    magnitude = np.linalg.norm(displacement, axis=1)
    max_disp_node = nnum[np.argmax(magnitude)]
    print(f"Max Displacement: {np.max(magnitude):.6f} at node {max_disp_node}")
    return displacement, magnitude

def extract_von_mises_stress(result, load_step=0):
    # Get nodal stress from the result file
    nnum, stress = result.nodal_stress(load_step)
    stress = np.nan_to_num(stress)  # Replace NaNs with 0

    # Unpack stress components
    sx, sy, sz = stress[:, 0], stress[:, 1], stress[:, 2]
    txy, tyz, tzx = stress[:, 3], stress[:, 4], stress[:, 5]

    # Calculate von Mises stress using full 3D formula
    von_mises = np.sqrt(
        0.5 * ((sx - sy)**2 + (sy - sz)**2 + (sz - sx)**2) +
        3 * (txy**2 + tyz**2 + tzx**2)
    )

    # Get max value and location
    max_val = np.max(von_mises)
    max_node = nnum[np.argmax(von_mises)]
    print(f"Max von Mises Stress: {max_val:.6f} at node {max_node}")

    return von_mises


def main():
    # === Load result file ===
    rst_path = r"D:\AltoAuto FEA Automation\AltoAuto FEA Automation - python template\AltoAuto_Main\Tools\gmsh workflow\cad_topology\mesh_runs\book_shelf.rst"
    result = reader.read_binary(rst_path)
    print("Loaded result:", result)
    total_energy = extract_stiffness_energy_from_rst(rst_path)
    print(total_energy)
    grid = result.grid
    load_step = 0

    # === Plot displacement ===
    displacement, disp_mag = extract_displacement(result, load_step)
    plot_deformed_contour(grid, displacement, disp_mag, "Displacement", scale=1, show_undeformed=True,style="surface")

    # === Plot von Mises stress (optional) ===
    von_mises = extract_von_mises_stress(result, load_step)
    plot_deformed_contour(grid, displacement, von_mises, " Mises Stress", scale=1,show_undeformed=True,style="surface")

if __name__ == "__main__":
    main()
