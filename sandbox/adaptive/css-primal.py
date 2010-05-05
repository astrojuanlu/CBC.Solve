## The primal problem is solved using the consistent splitting
## scheme, and the velocity and pressure solutions are stored at each
## time step to be later used in the dual problem solution.

from dolfin import *
from numpy import array, vstack, savetxt, loadtxt
from math import ceil
import os

# Create the mesh
mesh = UnitSquare(24, 24)

# Some useful fields related to the geometry
n = FacetNormal(mesh)

# Constants
nu = 1.0/8.0
T = 5.0
k = 0.25*mesh.hmin()

# Dirichlet boundary
class DirichletBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return x[1] < DOLFIN_EPS or x[1] > 1.0 - DOLFIN_EPS

# Neumann boundary
# Inflow boundary
class InflowBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] < DOLFIN_EPS

# Outflow boundary
class OutflowBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] > 1.0 - DOLFIN_EPS

# Neumann boundary condition. This is a vector field that is (1, 0) on
# the left end and (0, 0) on the right end (and everywhere else).
class NeumannBoundaryCondition(Expression):
    def eval(self, values, x):
        if (x[0] == 0 and
            (x[1] > 0 + DOLFIN_EPS or x[1] < 1.0 - DOLFIN_EPS)):
            values[0] = 1.0
            values[1] = 0.0
        elif (x[0] == 1):
            values[0] = 0.0
            values[1] = 0.0
        else:
            values[0] = 0.0
            values[0] = 0.0

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
bc0 = DirichletBC(vector, g0, DirichletBoundary())

# Transform Neumann boundary conditions to values for pressure on the
# boundary #FIXME
g1 = Constant(1.0)
bc1 = DirichletBC(scalar, g1, InflowBoundary())

# Create outflow boundary condition for pressure
g2 = Constant(0.0)
bc2 = DirichletBC(scalar, g2, OutflowBoundary())

# Create boundary conditions for psi
g3 = Constant(0.0)
bc3 = DirichletBC(scalar, g3, InflowBoundary())
g4 = Constant(0.0)
bc4 = DirichletBC(scalar, g4, OutflowBoundary())

u0 = Constant((0.0, 0.0))
u0 = project(u0, vector)
p0 = Expression('1.0 - x[0]')
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

g = NeumannBoundaryCondition(vector, mesh)     # Traction force

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

# Variables to store all the answers
u_store = u0.vector().array()
p_store = p0.vector().array()

# Create files for storing the solution in VTK format
if not os.path.exists("./results/square"):
    os.makedirs("./results/square")

ufile_pvd = File("results/square/css_primal_velocity.pvd")
pfile_pvd = File("results/square/css_primal_pressure.pvd")

while t < T:

#    plot(p1)
#    plot(u1)
		
    # Propagate values to next time step
    t += k

    g1.t = t

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

    ufile_pvd << u1
    pfile_pvd << p1

    u_store = vstack([u_store, u1.vector().array()])
    p_store = vstack([p_store, p1.vector().array()])

    u0.assign(u1)
    p0.assign(p1)
    
    i = i + 1

    print "Computed the solution at step", i, "where t =", t, "(", t/T*100, "% )"

if not os.path.exists("./results/square"):
    os.makedirs("./results/square")
savetxt('results/square/CSS_u_store.txt', u_store, fmt="%12.6G")
savetxt('results/square/CSS_p_store.txt', p_store, fmt="%12.6G")
