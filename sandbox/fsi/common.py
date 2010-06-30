# FIXME: All this should be moved somewhere else

import sys

from dolfin import *
from numpy import array, append, zeros
from cbc.common.utils import *

# Set default parameters
tol = 1e-10

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

# Define area of the structure
structure_area = (structure_right - structure_left)*structure_top

# Initialize mesh conectivity
mesh.init(D-1, D)


omega_F0 = Mesh(Omega_F)
omega_F1 = Mesh(Omega_F)


# Structure dirichlet boundaries
def dirichlet_boundaries(x):
    #FIXME: Figure out how to use the constants above in the
    #following boundary definitions
    #bottom ="x[1] == 0.0 && x[0] >= 1.4 && x[0] <= 1.6"
    #return [bottom]
    return x[1] < DOLFIN_EPS and x[0] >= structure_left and x[0] <= structure_right



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
# Note only one vector
dual_Z = TimeSeries("dual_Z")

# Fix time step if needed. Note that this has to be done
# in oder to save the primal data at the correct time
dt, t_range = timestep_range(T, dt)
