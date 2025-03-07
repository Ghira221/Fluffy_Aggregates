# Import some useful modules.
import jax
import jax.numpy as np
import os
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull



# Import JAX-FEM specific modules.
from jax_fem.problem import Problem # type: ignore
from jax_fem.solver import solver # type: ignore
from jax_fem.utils import save_sol # type: ignore
from jax_fem.generate_mesh import box_mesh_gmsh, get_meshio_cell_type, Mesh # type: ignore

class DensityError(Exception):
    pass

class Plasticity(Problem):

    def custom_init(self, additional_info = [0.1]):
        """_summary_

        Args:
            density (list, optional): the density field. Defaults to [.1].
        """
        # ceate objects for r
        self.fe = self.fes[0]
        self.rho = additional_info
        self.epsilons_old = np.zeros((len(self.fe.cells), self.fe.num_quads, self.fe.vec, self.dim))
        self.sigmas_old = np.zeros_like(self.epsilons_old)

        # creating a constant density field of length cells because it's a cell quantity
        self.density = np.full((self.epsilons_old.shape[0], self.epsilons_old.shape[1]), self.rho)
        print(self.density.shape)

        # these are things that can change throughout the sim
        self.internal_vars = [self.sigmas_old, self.epsilons_old, self.density]

        # elasticity coefficients
        self.E = 100 # Pa
        self.nu = 0.05

    def get_tensor_map(self):
        # meaning: get the tensors you want to solve for.
        # we throw away the strain and density
        _,_, stress_return_map = self.get_maps()
        return stress_return_map
    
    def get_maps(self):
        def safe_sqrt(x):  
            # np.sqrt is not differentiable at 0
            safe_x = np.where(x > 0., np.sqrt(x), 0.)
            return safe_x

        def safe_divide(x, y):
            return np.where(y == 0., 0., x/y)


        def strain(u_grad):
            return 0.5*(u_grad + u_grad.T)
        
        def stress(eps, E):
            return E*self.nu/((1+self.nu)*(1-2*self.nu))*np.trace(eps)*np.eye(self.dim) + 2*E/(2.*(1. + self.nu))*eps
        
        def E_comp(density):
            """Calculates the Young's modulus of the material in each cell as a function of the density.

            Args:
                density (_type_): _description_

            Returns:
                _type_: _description_
            """


            return 0

        def P_comp(density):
            """Calculates the compressive strength of a material as a function of volume filling factor for homogeneous, monodisperse monomers. Adapted from Tatsuuma et al. 2023.
            The equation takes the density rather than volume filling factor as input. 

            Args:
                density (float, array-like): N dim. array of cell volumes

            Returns:
                float, array-like: compressive strength of each cell in the material
            """
            # can clean this up later
            # move all constants outside?

            xi_crit = 8*1e-10 # critical rolling energy, 8 angstrom in m
            rho_0 = 2.65 * 1e3 / 1e6 # silicate, g/cm^3 in kg/m^3
            r0 = 0.1e-6 # monomer size, 0.1 micron in m
            gamma = 20*1e3 # surface energy, mJ/m^2 in J/m^2
            m0 = rho_0 * (4*np.pi)/3 * r0**3 # monomer mass in kg, silicate 
            Eroll = 6*np.pi**2 * gamma * r0 * xi_crit # rolling energy 
            phi_max = 0.78 # max volume filling factor
            # C = (4*np.pi*r0**3)/(3*m0) # constant
            # print(density)
            C = 1 # for a density varying between 0 and 1, C <= phi_max
            phi = C*density
            Y = (Eroll/(r0**3)) * (1/(C*density) - 1/phi_max)**(-3)

            # this was supposed to catch negative value, but jax is stupid with booleans evals
            # if phi.value >= phi_max:
            #     terminal_vals = []
            #     raise DensityError(
            #         f"Material has exceeded maximum packing at {phi:.3e}"
            #     )

            #     terminal_vals.append(phi, Y)
            #     print(terminal_vals)

            return phi, Y
        
        def density_return_map(density, cell_volume):
            return density * (1/cell_volume)

        def stress_return_map(u_grad, sigma_old, eps_old, density):
            """ Calculate the new stress field from the old deformation gradient, old stress, and old strain. 

            Args:
                u_grad (_type_): 3x3 matrix, gradient of the solution
                sigma_old (_type_): 3x3 matrix of the stresses
                eps_old (_type_): 3x3 matrix of the strains
                density (_type_): scalar value of the density

            Returns:
                _type_: updated stress

            """
    
            # calculate the yield strength and Young's modulus from the density field
            phi, Y = P_comp(density)
            E = 1e5*density # E = constant * density

            # print(density.shape)
            # print(yield_strength.shape)
            # print(E.shape)
            # print(sigma_old.shape)
            # print(eps_old.shape)

            #print("Yield strength: ", Y)
            #print("E first cell: ", E)

            
            # calculate the new stress field with a radial return map
            deps = strain(u_grad) - eps_old
            sigma_trial = stress(deps, E) + sigma_old
            s_dev = sigma_trial - 1./self.dim*np.trace(sigma_trial)*np.eye(self.dim)
            von_Mises = safe_sqrt(3./2.*np.sum(s_dev*s_dev))
            f_yield = von_Mises - Y
            f_yield_plus = np.where(f_yield > 0., f_yield, 0.)
            sigma = sigma_trial - safe_divide(f_yield_plus*s_dev,von_Mises)
            #print(sigma_trial)

            return sigma
        
        return density_return_map, strain, stress_return_map
    
    def stress_strain_funcs(self):
        """Return mapping functions for stress, strain, density

        Returns:
            _type_: _description_
        """
        density_return_map, strain, stress_return_map = self.get_maps()
        vmap_strain = jax.vmap(jax.vmap(strain))
        vmap_stress_return_map = jax.vmap(jax.vmap(stress_return_map))
        vmap_density = jax.vmap(jax.vmap(density_return_map))
        return vmap_strain, vmap_stress_return_map, vmap_density
        

    def update_stress_strain_density(self, sol, density):
        # set new values for old stress, strain, density
        u_grads = self.fe.sol_to_grad(sol)
        #print(sol.shape)
        #print(u_grads.shape)
        vmap_strain, vmap_stress_rm, vmap_density = self.stress_strain_funcs()
        self.density = density
        self.sigmas_old = vmap_stress_rm(u_grads, self.sigmas_old, self.epsilons_old, self.density) # use old or new density? I think old
        self.epsilons_old = vmap_strain(u_grads)
        self.internal_vars = [self.sigmas_old, self.epsilons_old, self.density]
        print(self.density.shape, self.sigmas_old.shape, self.epsilons_old.shape)


    def compute_avg_stress(self):
        # For post-processing only: Compute volume averaged stress.
        # (num_cells*num_quads, vec, dim) * (num_cells*num_quads, 1, 1) -> (vec, dim)
        sigma = np.sum(self.sigmas_old.reshape(-1, self.fe.vec, self.dim) * self.fe.JxW.reshape(-1)[:, None, None], 0)
        vol = np.sum(self.fe.JxW)
        avg_sigma = sigma/vol
        return avg_sigma

test_num = 2 # for file naming
solver_type = 'jax_solver'
Nx, Ny, Nz = 10, 10, 10 # cells in a row
N_cells = Nx*Ny*Nz
steps = 20
deform = 0.6
name = '60' # deformation percentage
E_val = '1e5rho' # make this into an actual modifiable param

# Specify mesh-related information(first-order hexahedron element).
ele_type = 'HEX8'
cell_type = get_meshio_cell_type(ele_type)
data_dir = os.path.join(os.path.dirname('/home/elouan/Compression/'), f'density-model-{test_num}_{name}_{solver_type}_{E_val}')

# box dims
Lx, Ly, Lz = 1., 1., 1.
meshio_mesh = box_mesh_gmsh(Nx=Nx, Ny=Ny, Nz=Nz, Lx=Lx, Ly=Ly, Lz=Lz, data_dir=data_dir, ele_type=ele_type)
mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])



# Define boundary locations.
# this is so inefficient, I can't even use an N-d array for the center conditions. Or at least I don't know how to.
def top(point):
    return np.isclose(point[2], Lz, atol=1e-5)

def bottom(point):
    return np.isclose(point[2], 0., atol=1e-5)

def center(point):
    center_cond = np.array([[Lx/2], [Ly/2], [0.]])
    return np.isclose(point, center_cond, atol = 1e-2)

def center_x(point):
    return np.isclose(point[0], Lx/2, atol = 1e-2)

def center_y(point):
    return np.isclose(point[1], Ly/2, atol = 1e-2)
    



# Define Dirichlet boundary values.
# We fix the z-component of the displacement field to be zero on the 'bottom' 
# side, and control the z-component on the 'top' side.
def dirichlet_val_bottom(point):
    return 0.

def get_dirichlet_top(disp):
    def val_fn(point):
        return disp
    return val_fn

# compression specs
disps = np.linspace(0.,-deform,steps) # compression in the z-dir
density_init = [0.0001 * 1/(Nx * Ny * Nz)] # the extra factor is so that this will be the actual density, not the density x a factor from the grid size
location_fns = [center_x, center_y, bottom, top]
value_fns = [dirichlet_val_bottom]*3 + [get_dirichlet_top(disps[0])]
vecs = [0, 1, 2, 2]

dirichlet_bc_info = [location_fns, vecs, value_fns]



# Define problem, solve it and save solutions to local folder
problem = Plasticity(mesh, vec=3, dim=3, dirichlet_bc_info=dirichlet_bc_info, additional_info=density_init)
avg_stresses = []
cell_vols = []
mat_vol = []
#print(problem.density)


for i, disp in enumerate(disps):
    print(f"\nStep {i} in {len(disps)}, disp = {disp}")

    # update Dirichlet conditions 
    dirichlet_bc_info[-1][-1] = get_dirichlet_top(disp)
    
    # solve the problem
    problem.fe.update_Dirichlet_boundary_conditions(dirichlet_bc_info)
    sol_list = solver(problem, solver_options={f'{solver_type}': {}})
    vtk_path = os.path.join(data_dir, f'vtk/u_{i:03d}.vtu')
    save_sol(problem.fe, sol_list[0], vtk_path)

    # compute new densities from solution
    # find coordinates of nodes belonging to each cell
    coords_cell_nodes = problem.fe.points[problem.fe.cells] + sol_list[0][problem.fe.cells]
    #print(coords_cell_nodes[0]) # checking if the coordinates of the first cell even change

    # compute cell volumes with ConvexHull
    cell_volume = np.zeros(N_cells)
    for i in range(N_cells):
        points = coords_cell_nodes[i]
        hull = ConvexHull(points)
        # cell volumes in this iteration
        cell_volume = cell_volume.at[i].set(hull.volume) # simple item assignement doesn't work with jax aparently, hence this code line instead

    # save volumes for each iteration to a list
    cell_vols.append(cell_volume)
    mat_vol = np.sum(cell_volume)
    density = np.repeat(density_init[0]*(1/cell_volume), 8).reshape((-1, 8))
    #print(1/cell_volume[0])
    print(density)

    # update the stress, strain and density with the solution
    problem.update_stress_strain_density(sol_list[0], density)
    avg_stress = problem.compute_avg_stress()
    avg_stresses.append(avg_stress)



avg_stresses = np.array(avg_stresses)


# Plot the volume-averaged stress versus the vertical displacement of the top surface.
fig = plt.figure(figsize=(10, 8))
plt.plot(disps, avg_stresses[:, 2, 2], color='red', marker='o', markersize=8, linestyle='-') 
plt.xlabel(r'Displacement of top surface [m]', fontsize=20)
plt.ylabel(r'Volume averaged stress (z-z) [Pa]', fontsize=20)
plt.tick_params(labelsize=18)
plt.savefig(f"./density-model-{test_num}_{name}_{solver_type}_{E_val}/stress-strain-{test_num}.png")
plt.show()


