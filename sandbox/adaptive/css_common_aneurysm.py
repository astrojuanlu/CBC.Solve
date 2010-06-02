## Common information for the primal and dual problems

from dolfin import *
import os
from math import ceil
from numpy import linspace

# Load the mesh
mesh = Mesh("aneurysm.xml")

# Some useful fields related to the geometry
n = FacetNormal(mesh)

# Constants
nu = 1.0/8.0
T = 10.0
k = 0.25*mesh.hmin()
N = int(ceil(T / k))

dt = k
t_range = linspace(0, T, N + 1)
k = t_range[1]

# Warn about changing time step
if dt != k:
    warning("Changing time step from %g to %g" % (k, dt))

TOL = 1e-6
REFINE_RATIO = 0.05

# Neumann boundary
# Inflow boundary
def inflowboundary(x):
    return x[0] == -10.0

# Outflow boundary
def outflowboundary(x):
    return x[0] == 10.0

# Dirichlet boundary
def noslipboundary(x, on_boundary):
    return on_boundary and \
        not inflowboundary(x) and \
        not outflowboundary(x)

# Define Epsilon
def epsilon(v):
    return 0.5*(grad(v) + grad(v).T)

# Define Sigma
def sigma(v, q):
    return 2.0*nu*epsilon(v) - q*Identity(v.cell().d)

# Define function spaces
vector = VectorFunctionSpace(mesh, "CG", 2)
scalar = FunctionSpace(mesh, "CG", 1)
system = MixedFunctionSpace([vector, scalar])
scalarDG = FunctionSpace(mesh, "DG", 0)
vectorDG = VectorFunctionSpace(mesh, "DG", 0)

# Create Dirichlet (no-slip) boundary conditions for velocity
g0 = Constant((0.0, 0.0))
bc0 = DirichletBC(vector, g0, noslipboundary)

# Transform Neumann boundary conditions to values for pressure on the
# boundary #FIXME
g1 = Constant(1.0)
bc1 = DirichletBC(scalar, g1, inflowboundary)

# Create outflow boundary condition for pressure
g2 = Constant(0.0)
bc2 = DirichletBC(scalar, g2, outflowboundary)

# Create boundary conditions for psi
g3 = Constant(0.0)
bc3 = DirichletBC(scalar, g3, inflowboundary)
g4 = Constant(0.0)
bc4 = DirichletBC(scalar, g4, outflowboundary)

# Driving force
f = Constant((0.0, 0.0))

# Create folder for storing results
if not os.path.exists("./results/aneurysm"):
    os.makedirs("./results/aneurysm")

# Create files for storing the solution in VTK format
ufile = File("results/aneurysm/primal_velocity.pvd")
pfile = File("results/aneurysm/primal_pressure.pvd")
wfile = File("results/aneurysm/dual_velocity.pvd")
rfile = File("results/aneurysm/dual_pressure.pvd")

# Create time series for storing the solution in binary format
useries = TimeSeries("results/aneurysm/primal_velocity")
pseries = TimeSeries("results/aneurysm/primal_pressure")
wseries = TimeSeries("results/aneurysm/dual_velocity")
rseries = TimeSeries("results/aneurysm/dual_pressure")
