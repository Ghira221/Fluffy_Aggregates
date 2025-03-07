import pyvista
from dolfinx import mesh, fem, plot, io, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
import ufl
import numpy as np

# for diagnostics
import matplotlib.pyplot as plt

L = 1 # length
W = 1 # width and height (it's a square cross-section)
mu = 1 # lamé 1
rho = 0.1 # material density
delta = W / L # don't know?
gamma = 0.4 * delta**2 # what?
# beta = 1.25
# lambda_ = beta # lamé 2, why not define it as lambda_ outright
lambda_ = 1.25 # this works fine
g = gamma # help? This is supposed to be the gravitational acceleration


def epsilon(u):
    return ufl.sym(ufl.grad(u))  # Equivalent to 0.5*(ufl.nabla_grad(u) + ufl.nabla_grad(u).T)


def sigma(u):
    return lambda_ * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2 * mu * epsilon(u)



# create a box-shaped mesh from [0,0,0] to [L, W, W] with 6x6x6 hexagonal cells
domain = mesh.create_box(MPI.COMM_WORLD, [np.array([0, 0, 0]), np.array([L, W, W])],
                        [20, 20, 20], cell_type=mesh.CellType.hexahedron) 
V = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim, )))

fdim = domain.topology.dim - 1
boundary_facets = mesh.locate_entities_boundary(domain, fdim, marker = lambda x: np.isclose(x[2], 0.0))

u_D = np.array([0, 0, 0], dtype=default_scalar_type)
bc = fem.dirichletbc(u_D, fem.locate_dofs_topological(V, fdim, boundary_facets), V)

# we want the traction on the boundary to remain 0
T = fem.Constant(domain, default_scalar_type((0, 0, 0)))
ds = ufl.Measure("ds", domain=domain)


F = -rho * g

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
f = fem.Constant(domain, default_scalar_type((0, 0, F))) # it's a body force, as specified. I think that's why we apply it to the entire domain
a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
L = ufl.dot(f, v) * ufl.dx + ufl.dot(T, v) * ds

problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()

pyvista.start_xvfb()

# Create plotter and pyvista grid
p = pyvista.Plotter(off_screen=True)
topology, cell_types, geometry = plot.vtk_mesh(V)
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

# Attach vector values to grid and warp grid by vector
grid["u"] = uh.x.array.reshape((geometry.shape[0], 3))
actor_0 = p.add_mesh(grid, style="wireframe", color="k")
warped = grid.warp_by_vector("u", factor=1.5)
actor_1 = p.add_mesh(warped, show_edges=True)
p.show_axes()
if not pyvista.OFF_SCREEN:
    p.show()
else:
    figure_as_array = p.screenshot(f"deflection_{rho}.png")

