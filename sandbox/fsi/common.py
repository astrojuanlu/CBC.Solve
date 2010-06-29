import sys

from dolfin import *
from numpy import array, append, zeros
from cbc.common.utils import *

# Set default parameters
nx = 80
dt = 0.025
T = 1.0
mesh_smooth = 50
tol = 1e-10
maxiter = 100

# Get command-line parameters
for arg in sys.argv[1:]:
    if not "=" in arg: continue
    key, val = arg.split("=")

    if key == "nx":
        nx = int(val)
    elif key == "dt":
        dt = float(val)
    elif key == "T":
        T = float(val)
    elif key == "smooth":
        mesh_smooth = int(val)

# Set resolution
ny = nx/4

# Constants related to the geometry of the channel and the obstruction
channel_length  = 4.0
channel_height  = 1.0
structure_left  = 1.4
structure_right = 1.6
structure_top   = 0.5

# Define area of the structure
structure_area = (structure_right - structure_left)*structure_top

# Create the complete mesh
mesh = Rectangle(0.0, 0.0, channel_length, channel_height, nx, ny)

# Define dimension of mesh
D = mesh.topology().dim()

# Initialize mesh conectivity
mesh.init(D-1, D)

# Define inflow boundary
def inflow(x):
    return x[0] < DOLFIN_EPS and x[1] > DOLFIN_EPS and x[1] < channel_height - DOLFIN_EPS

# Define outflow boundary
def outflow(x):
    return x[0] > channel_length - DOLFIN_EPS and x[1] > DOLFIN_EPS and x[1] < channel_height - DOLFIN_EPS

# Define noslip boundary
def noslip(x, on_boundary):
    return on_boundary and not inflow(x) and not outflow(x)

# Define structure subdomain
class Structure(SubDomain):
    def inside(self, x, on_boundary):
        return \
            x[0] >= structure_left  - DOLFIN_EPS  and \
            x[0] <= structure_right + DOLFIN_EPS  and \
            x[1] <= structure_top   + DOLFIN_EPS

# Structure dirichlet boundaries
def dirichlet_boundaries(x):
    #FIXME: Figure out how to use the constants above in the
    #following boundary definitions
    #bottom ="x[1] == 0.0 && x[0] >= 1.4 && x[0] <= 1.6"
    #return [bottom]
    return x[1] < DOLFIN_EPS and x[0] >= structure_left and x[0] <= structure_right

# Create structure subdomain
structure = Structure()

# Create cell_domain markers (0=fluid,  1=structure)
cell_domains = MeshFunction("uint", mesh, D)
cell_domains.set_all(0)
structure.mark(cell_domains, 1)

# Extract submeshes for fluid and structure
Omega = mesh
Omega_F = SubMesh(mesh, cell_domains, 0)
Omega_S = SubMesh(mesh, cell_domains, 1)
omega_F0 = Mesh(Omega_F)
omega_F1 = Mesh(Omega_F)

# Extract matching indices for fluid and structure
structure_to_fluid = compute_vertex_map(Omega_S, Omega_F)

# Extract matching indices for fluid and structure
fluid_indices = array([i for i in structure_to_fluid.itervalues()])
structure_indices = array([i for i in structure_to_fluid.iterkeys()])

# Extract matching dofs for fluid and structure
fdofs = append(fluid_indices, fluid_indices + Omega_F.num_vertices()) #This is not true for a P2 element!
sdofs = append(structure_indices, structure_indices + Omega_S.num_vertices())

# Create facet marker for the entire mesh
interior_facet_domains = MeshFunction("uint", Omega, D-1)
interior_facet_domains.set_all(0)

# Create facet marker for inflow (used in the Neumann BCs for the dual fluid)
left = compile_subdomains("x[0] == 0.0")
exterior_boundary = MeshFunction("uint", Omega, D-1)
left.mark(exterior_boundary, 2)

# Create facet marker for outflow (used in the Neumann BCs for the dual fluid)
right = compile_subdomains("x[0] == channel_length")
exterior_boundary = MeshFunction("uint", Omega, D-1)
right.mark(exterior_boundary, 3)

# Create facet orientation for the entire mesh
facet_orientation = mesh.data().create_mesh_function("facet orientation", D - 1)
facet_orientation.set_all(0)

# Functions for adding vectors between domains        self.ts = 0 
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

    # Skip facets on the boundary
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
    else:
        # Just set c0, will not be used
        facet_orientation[facet.index()] = c0

# Create time series for storing primal
primal_u_F = TimeSeries("primal_u_F")
primal_p_F = TimeSeries("primal_p_F")
primal_U_S = TimeSeries("primal_U_S")
primal_P_S = TimeSeries("primal_P_S")
primal_U_M = TimeSeries("primal_U_M")

# Create time series for storing dual data
dual_Z_UF = TimeSeries("dual_Z_UF")
dual_Z_PF = TimeSeries("dual_Z_PF")
dual_Z_US = TimeSeries("dual_Z_US")
dual_Z_PS = TimeSeries("dual_Z_PS")
dual_Z_UM = TimeSeries("dual_Z_UM")
dual_Z_PM = TimeSeries("dual_Z_PM")

# Fix time step if needed. Note that this has to be done
# in oder to save the primal data at the correct time
dt, t_range = timestep_range(T, dt)
