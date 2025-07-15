# visulize_ansys_rst_result

A small Python project to load and visualize results from ANSYS `.rst` files (simulation result files).

This tool uses [PyVista](https://github.com/pyvista/pyvista) and [ansys-mapdl-reader](https://github.com/pyansys/ansys-mapdl-reader) to extract and plot stress, displacement, or other FEA results from ANSYS simulations.

---

## Features
- Loads ANSYS `.rst` result files
- Plots nodal displacements, vm_stress, or custom fields
- press a to activate probe
- press c to clear all


## Results 
 - Total deformation
 - Von mises stress\n
(feel free to visulize your own result: "https://mapdl.docs.pyansys.com/version/stable/examples/gallery_examples/00-mapdl-examples/basic_dpf_example.html")
Here’s what it looks like:

![Deformed mesh plot](examples/example.png)
