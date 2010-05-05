## The primal problem is solved using the consistent splitting
## scheme, and the velocity and pressure solutions are stored at each
## time step to be later used in the dual problem solution.

from dolfin import *
from math import ceil
import os

# Create the mesh
mesh = UnitSquare(24, 24)

# Some useful fields related to the geometry
n = FacetNormal(mesh)

# Constants
nu = 1.0/8.0
T = 0.5
k = 0.25*mesh.hmin()

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

# Create Dirichlet (no-slip) boundary conditions for velocity
g0 = Constant((0, 0))
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

u0 = Constant((0.0, 0.0))
u0 = project(u0, vector)
p0 = Expression("1.0 - x[0]")
p0 = project(p0, scalar)

bcv = [bc0]
bcp = [bc1, bc2]
bcpsi = [bc3, bc4]

# Test and trial functions
v = TestFunction(vector)
q = TestFunction(scalar)
u = TrialFunction(vector)
p = TrialFunction(scalar)

# Functions
u1  = interpolate(u0, vector)
p1  = interpolate(p0, scalar)
f   = Constant((0.0, 0.0))
psi = Function(scalar)

# Tentative velocity step
F1 =  (1/k)*inner(v, u - u0)*dx + inner(v, grad(u0)*u0)*dx \
    + inner(epsilon(v), sigma(u, p0))*dx - inner(v, f)*dx \
    - nu*inner(v, grad(u).T*n)*ds + inner(v, p0*n)*ds

a1 = lhs(F1)
L1 = rhs(F1)

# Pressure correction
a2 = dot(grad(q), grad(p))*dx
L2 = (1/k)*inner(grad(q), u1 - u0)*dx \
    - (1/k)*inner(q*n, u1 - u0)*ds

# Pressure update
a3 = q*p*dx
L3 = q*p1*dx + q*psi*dx - nu*q*div(u1)*dx
	
# Assemble matrices
A1 = assemble(a1)
A2 = assemble(a2)
A3 = assemble(a3)

# Time loop
t = 0.0
i = 0

# Create folder for storing results
if not os.path.exists("./results/square"):
    os.makedirs("./results/square")

# Create files for storing the solution in VTK format
ufile = File("results/square/css_primal_velocity.pvd")
pfile = File("results/square/css_primal_pressure.pvd")

# Create time series for storing the solution in binary format
useries = TimeSeries("results/square/css_primal_velocity")
pseries = TimeSeries("results/square/css_primal_pressure")

while t < T:

#    plot(p1)
#    plot(u1)
		
    # Propagate values to next time step
    t += k

    # Compute tentative velocity step
    b = assemble(L1)
    [bc.apply(A1, b) for bc in bcv]
    solve(A1, u1.vector(), b, "gmres", "ilu")

    # Compute pressure correction
    b = assemble(L2)
    if len(bcp) == 0: normalize(b)
    [bc.apply(A2, b) for bc in bcpsi]
    solve(A2, psi.vector(), b, "gmres", "amg_hypre")
    if len(bcp) == 0: normalize(psi.vector()) 

    # Compute updated pressure
    b = assemble(L3)
    if len(bcp) == 0: normalize(b)
    [bc.apply(A3, b) for bc in bcp]
    solve(A3, p1.vector(), b, "gmres", "ilu")

    useries.store(u1.vector(), t)
    pseries.store(p1.vector(), t)

    ufile << u1
    pfile << p1

    u0.assign(u1)
    p0.assign(p1)
    
    i = i + 1

    print "Computed the solution at step", i, "where t =", t, "(", t/T*100, "% )"
