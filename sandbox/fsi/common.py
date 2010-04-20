from dolfin import *
from numpy import array, append, zeros

# Constants related to the geometry of the channel and the obstruction
channel_length  = 4.0
channel_height  = 1.0
structure_left  = 1.4
structure_right = 1.6
structure_top   = 0.5
nx = 80
ny = nx/4 

# Parameters
dt = 0.1
T = 1.0
tol = 1e-4

# Create the complete mesh
mesh = Rectangle(0.0, 0.0, channel_length, channel_height, nx, ny)

# Define dimension of mesh
D = mesh.topology().dim()

# Initialize mesh conectivity 
mesh.init(D-1, D)

# Define structure subdomain
class Structure(SubDomain):
    def inside(self, x, on_boundary):
        return (x[0] >= structure_left) and (x[0] <= structure_right) \
            and (x[1] <= structure_top)

# Create structure subdomain
structure = Structure()

# Create subdomain markers (0=fluid,  1=structure)
sub_domains = MeshFunction("uint", mesh, D)
sub_domains.set_all(0)
structure.mark(sub_domains, 1)

# Create cell_domain markers (0=fluid,  1=structure)
cell_domains = MeshFunction("uint", mesh, D)
cell_domains.set_all(0)
structure.mark(cell_domains, 1)

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

# Create facet marker for the entire mesh
interior_facet_domains = MeshFunction("uint", Omega, D-1)
interior_facet_domains.set_all(0)

# Create facet marker for outflow
right = compile_subdomains("x[0] == channel_length")
exterior_boundary = MeshFunction("uint", Omega, D-1)
right.mark(exterior_boundary, 2)

# Create facet orientation for the entire mesh
facet_orientation = mesh.data().create_mesh_function("facet orientation", D - 1)
facet_orientation.set_all(0)

# Define inflow boundary
def inflow(x):
    return x[0] < DOLFIN_EPS and x[1] > DOLFIN_EPS and x[1] < channel_height - DOLFIN_EPS

# Define outflow boundary
def outflow(x):
    return x[0] > channel_length - DOLFIN_EPS and x[1] > DOLFIN_EPS and x[1] < channel_height - DOLFIN_EPS

# Define noslip boundary
def noslip(x, on_boundary):
    return on_boundary and not inflow(x) and not outflow(x)

# Structure BCs 
def dirichlet_boundaries(x):
    #FIXME: Figure out how to use the constants above in the
    #following boundary definitions
    #bottom ="x[1] == 0.0 && x[0] >= 1.4 && x[0] <= 1.6"
    #return [bottom]
    return x[1] < DOLFIN_EPS and x[0] >= structure_left and x[0] <= structure_right

# Functions for adding vectors between domains
def fsi_add_f2s(xs, xf):
    "Compute xs += xf for corresponding indices"
    xs_array = xs.array()
    xf_array = xf.array()
    xs_array[sdofs] += xf_array[fdofs]
    xs[:] = xs_array

def fsi_add_s2f(xf, xs):
    "Compute xs += xf for corresponding indices"
    xf_array = xf.array()
    xs_array = xs.array()
    xf_array[fdofs] += xs_array[sdofs]
    xf[:] = xf_array


# Mark facet orientation for the fsi boundary
for facet in facets(mesh):

    # Skip facets on the non-boundary
    if facet.num_entities(D) == 1:
        continue

    # Get the two cell indices
    c0, c1 = facet.entities(D)

    # Create the two cells
    cell0 = Cell(mesh, c0)
    cell1 = Cell(mesh, c1)

    # Get the two midpoints
    p0 = cell0.midpoint()
    p1 = cell1.midpoint()

    # Check if the points are inside
    p0_inside = structure.inside(p0, False)
    p1_inside = structure.inside(p1, False)

    # Look for points where exactly one is inside the structure
    if p0_inside and not p1_inside:
        interior_facet_domains[facet.index()] = 1
        facet_orientation[facet.index()] = c1
    elif p1_inside and not p0_inside:
        interior_facet_domains[facet.index()] = 1
        facet_orientation[facet.index()] = c0

#info(interior_facet_domains, True)
#info(facet_orientation, True)

# Create time series for storing primal
primal_u_F = TimeSeries("primal_u_F")
primal_p_F = TimeSeries("primal_p_F")
primal_U_S = TimeSeries("primal_U_S")
primal_P_S = TimeSeries("primal_P_S")
primal_U_M = TimeSeries("primal_U_M")



