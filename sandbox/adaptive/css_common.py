## Common information for the primal and dual problems

from dolfin import *
import os
from math import ceil
from numpy import linspace

# Create the mesh
mesh = UnitSquare(24, 24)

# Some useful fields related to the geometry
n = FacetNormal(mesh)

# Constants
nu = 1.0/8.0
T = 0.5
k = 0.25*mesh.hmin()
N = int(ceil(T / k))

dt = k
t_range = linspace(0, T, N + 1)
k = t_range[1]

# Warn about changing time step
if dt != k:
    warning("Changing time step from %g to %g" % (k, dt))

TOL = 1e-6
REFINE_RATIO = 0.12

# Dirichlet boundary
def noslipboundary(x):
    return x[1] < DOLFIN_EPS or x[1] > 1.0 - DOLFIN_EPS

# Neumann boundary
# Inflow boundary
def inflowboundary(x):
    return x[0] < DOLFIN_EPS

# Outflow boundary
def outflowboundary(x):
    return x[0] > 1.0 - DOLFIN_EPS

# Define Epsilon
def epsilon(v):
    return 0.5*(grad(v) + grad(v).T)

# Define Sigma
def sigma(v, q):
    return 2.0*nu*epsilon(v) - q*Identity(v.cell().d)

# Define function spaces
vector = VectorFunctionSpace(mesh, "CG", 2)
scalar = FunctionSpace(mesh, "CG", 1)
system = vector + scalar
scalarDG = FunctionSpace(mesh, "DG", 0)
vectorDG = VectorFunctionSpace(mesh, "DG", 0)

# Create folder for storing results
if not os.path.exists("./results/square"):
    os.makedirs("./results/square")

# Create files for storing the solution in VTK format
ufile = File("results/square/css_primal_velocity.pvd")
pfile = File("results/square/css_primal_pressure.pvd")
wfile = File("results/square/css_dual_velocity.pvd")
rfile = File("results/square/css_dual_pressure.pvd")

# Create time series for storing the solution in binary format
useries = TimeSeries("results/square/css_primal_velocity")
pseries = TimeSeries("results/square/css_primal_pressure")
wseries = TimeSeries("results/square/css_dual_velocity")
rseries = TimeSeries("results/square/css_dual_pressure")
