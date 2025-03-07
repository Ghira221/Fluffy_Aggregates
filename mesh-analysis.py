import pyvista as pv
import os
import matplotlib.pyplot as plt
import numpy as np


folder = "1" # loading and unloading of a 
compr = "10"
parent_dir = f'C:/Users/spaan/Thesis/SecondSesh_13-02-2025/Compression/data_run{folder}_{compr}/vtk/'


# reading vtu files
# specify full path to mesh
meshes = []
warped_meshes = []
n = 0
for subdir, dirs, files in os.walk(parent_dir):
        for file in files:
            fdir = parent_dir + file

            pv_mesh = pv.read(fdir)
            meshes.append(pv_mesh)
            n +=1

print("Reading is done!")


# keep in mind that the factor of course works exponentially, so don't use the warp except for the final result, or scale by smt. else.
for mesh in meshes:
    warped_mesh = mesh.warp_by_vector()
    warped_meshes.append(warped_mesh)

# -------------- the Mesh Contents --------------------
k =20# step X, max deformation
warp = 1

mesh = meshes[k]

sol = mesh["sol"] # the solution (deformation u)
ux = sol[:, 0]
uy = sol[:, 1]
uz = sol[:, 2]

mesh_lo = meshes[3]
mesh_hi = meshes[20]


warped_mesh_hi = mesh_hi.warp_by_vector(vectors = 'sol', factor = warp, )
warped_mesh_lo = mesh_lo.warp_by_vector('sol', factor = warp)

#print(mesh_hi['sol'])

# ---------------- Plots smt. -----------------------
pl = pv.Plotter(shape=(1, 2))
pl.subplot(0,0)
pl.add_text("Low deformation")
pl.show_bounds()
pl.add_mesh(warped_mesh_lo, scalars = 'sol', cmap = 'magma', show_edges=True, show_vertices = True)
pl.subplot(0,1)
pl.add_text("High deformation")
pl.show_bounds()
pl.add_mesh(warped_mesh_hi, scalars = 'sol', cmap = 'magma', show_edges = True, show_vertices = True)
pl.show_axes()
# pl.show()


init_mesh = meshes[0]
# creating a str list of point labels
n_points = len(init_mesh.points)
p_labels = list(map(str, np.arange(0, n_points)))


# --------------- Plots smt. ----------------------
pl = pv.Plotter(shape=(1, 2))
pl.subplot(0,0)
pl.add_text("Cube at rest")
pl.add_mesh(init_mesh, show_edges=True, show_vertices=True, color = 'cyan')
pl.show_axes()
pl.add_floor(pad=1.0, color = 'silver')
pl.subplot(0,1)
pl.add_text("Node numbers")
pl.add_mesh(init_mesh, show_edges=True, show_vertices=True, opacity=0.2, color = 'cyan')
pl.show_axes()
pl.add_floor(pad=1.0, color = 'silver')
pl.add_point_labels(init_mesh.points, p_labels, font_size = 10, always_visible =False)
#pl.show()

# # cell volumes
# # Compute volumes and areas
# sized = warped_mesh_lo.compute_cell_sizes()
# print(sized)

# --------------- Plane Widget -------------------------

# plane widget, for clipping
# this is not working yet. The slider is nicer.
p_plane = pv.Plotter()
p_plane.add_mesh(warped_mesh_lo)

def callback(normal, origin):
    slc = mesh.slice(normal=normal, origin=origin)
    origin = list(origin)
    origin[2] = slc.bounds[5]

p_plane.add_plane_widget(callback, normal_rotation = True)
p_plane.add_mesh_clip_plane(warped_mesh_lo)
# p_plane.show()

# p_plane.plane_clipped_meshes


# ----------------- Sider widgets ---------------------

""" Okay the slider thing is tricky. I know how to add a slider, the trick is to somehow
    add the coordinates of the mesh as values so that I can supply it as the scalars that
    the slider should go over. So some function that takes a coordinate as input to then
    display the volume up until that coordinate, which is the arg. of the callback function.

    I can add slices, but that just gives me one slice, not a cut-off of the volume.

    Update: scalars = mesh.points[:, i]??? With i = the coordinate??? Getting closer.
    So I want the colormap to work on the solution, but the slider to work with the coordinate. 

    THERE'S AN ELEVATION FILTER, see this whole thing: https://docs.pyvista.org/examples/01-filter/using-filters#sphx-glr-examples-01-filter-using-filters-py

    Update 18/02/2025: it was as simple as the below code? Whaaaat? So easy >///< 
"""

# slice slider
# what does the underscore do? Makes no difference in th final figure...
pl = pv.Plotter()
mesh = warped_mesh_hi
_ = pl.add_mesh(mesh.outline(), )
_ = pl.add_mesh_slice(mesh, normal=[1, 0, 0], show_edges = True)
pl.show_axes()
# pl.show_bounds()
pl.show()


# slider bar widget based on solution value
mesh = warped_mesh_hi

p = pv.Plotter()
p.add_mesh_threshold(mesh, show_edges = True)
p.show()

# ------------------ Warped vs unwarped comparison ---------------
low_mesh_num = 3
hi_mesh_num = 18

pl2 = pv.Plotter(shape=(2, 2))

# low compression
pl2.subplot(0,0)
pl2.add_text(f"Unwarped - {low_mesh_num}")
pl2.add_mesh(meshes[low_mesh_num], scalars = 'sol', show_edges=True, cmap = 'magma')
pl2.show_axes()
pl2.show_bounds()
pl2.add_floor(pad=1.0, color = 'silver')

pl2.subplot(0,1)
pl2.add_text(f"Warped - {low_mesh_num}")
pl2.add_mesh(warped_meshes[low_mesh_num], scalars = 'sol', show_edges=True, cmap = 'magma')
pl2.show_axes()
pl2.show_bounds()
pl2.add_floor(pad=1.0, color = 'silver')

# high compression
pl2.subplot(1,0)
pl2.add_text(f"Unwarped - {hi_mesh_num}")
pl2.add_mesh(meshes[hi_mesh_num], scalars = 'sol', show_edges=True, cmap = 'magma')
pl2.show_axes()
pl2.show_bounds()
pl2.add_floor(pad=1.0, color = 'silver')

pl2.subplot(1,1)
pl2.add_text(f"Warped - {hi_mesh_num}")
pl2.add_mesh(warped_meshes[hi_mesh_num], scalars = 'sol', show_edges=True, cmap = 'magma')
pl2.show_axes()
pl2.show_bounds()
pl2.add_floor(pad=1.0, color = 'silver')


# pl2.show()

