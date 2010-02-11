from dolfin import *
from numpy import array, append, zeros

# Constants related to the geometry of the channel and the obstruction
channel_length  = 3.0
channel_height  = 1.0
structure_left  = 1.4
structure_right = 1.6
structure_top   = 0.5
nx = 30
ny = 10

# Create the complete mesh
mesh = Rectangle(0.0, 0.0, channel_length, channel_height, nx, ny)

# Define structure subdomain
class Structure(SubDomain):
    def inside(self, x, on_boundary):
        return (x[0] >= structure_left) and (x[0] <= structure_right) \
            and (x[1] <= structure_top)

# Create structure subdomain
structure = Structure()
    
# Create subdomain markers (0: fluid 1: structure)
sub_domains = MeshFunction("uint", mesh, mesh.topology().dim())
sub_domains.set_all(0)
structure = Structure()
structure.mark(sub_domains, 1)
plot(sub_domains, interactive=True)

# Create cell_domain markers (0: fluid 1: structure)
cell_domains = MeshFunction("uint", mesh, mesh.topology().dim())
cell_domains.set_all(0)
structure.mark(cell_domains, 1)

# Create interior_facet_domains
interior_facet_domains = MeshFunction("uint", mesh, mesh.topology().dim()-1)
interior_facet_domains.set_all(0)
structure.mark(interior_facet_domains, 1)

#info(interior_facet_domains, True)
#plot(cell_domains, interactive=True)

# Extract submeshes for fluid and structure
Omega = mesh
Omega_F = SubMesh(mesh, sub_domains, 0)
Omega_S = SubMesh(mesh, sub_domains, 1)
omega_F0 = Mesh(Omega_F)
omega_F1 = Mesh(Omega_F) 

# Extract matching indices for fluid and structure
structure_to_fluid = compute_vertex_map(Omega_S, Omega_F)

# Extract matching indices for fluid and structure
fluid_indices = array([i for i in structure_to_fluid.itervalues()])
structure_indices = array([i for i in structure_to_fluid.iterkeys()])

# Extract matching dofs for fluid and structure
fdofs = append(fluid_indices, fluid_indices + Omega_F.num_vertices())
sdofs = append(structure_indices, structure_indices + Omega_S.num_vertices())

# Create time series for storing primal
primal_u_F = TimeSeries("primal_u_F")
primal_p_F = TimeSeries("primal_p_F")
primal_U_S = TimeSeries("primal_U_S")
primal_U_M = TimeSeries("primal_U_M")

# Parameters
t = 0
T = 0.25
dt = 0.25
tol = 1e-2
