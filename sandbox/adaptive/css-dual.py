## The dual problem is solved using the stored velocity and pressure
## solutions of the primal problem.

from dolfin import *
from numpy import array, vstack, savetxt, loadtxt
from math import ceil

# Fix for transposed grad in UFL
from ufl import grad as ufl_grad
def grad(v):
    if v.rank() == 1:
        return ufl_grad(v).T
    else:
        return ufl_grad(v)

# Create the mesh
mesh = UnitSquare(24, 24)

# Some useful fields related to the geometry
h = CellSize(mesh)
n = FacetNormal(mesh)

# Constants
nu = 1.0/8.0
T = 5.0
k = 0.25*h.min()

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

# Define symmetric gradient
def sgrad(v):
    return 0.5*(grad(v) + grad(v).T)

# Define stress
def sigma(v, q):
    return 2.0*nu*sgrad(v) - q*Identity(v.cell().d)

# Define a portion of the linearised dual bilinear form
def a_tilde(uh, ph, v, q, w, r):
    return inner(grad(uh)*v + grad(v)*uh, w)*dx \
        + inner(sigma(v, q), sgrad(w))*dx \
        - nu*inner(sgrad(v)*n, w)*ds \
        + div(v)*r*dx

# Delineate goal functional region
class Cutoff(Function):
    def eval(self, values, x):
        if (x[1] > 1.0 - DOLFIN_EPS and x[0] > 0.0 + DOLFIN_EPS and x[0] < 1.0 - DOLFIN_EPS):
            values[0] = 1.0
        else:
            values[0] = 0.0
    
# Define function spaces
vector = VectorFunctionSpace(mesh, "CG", 2)
scalar = FunctionSpace(mesh, "CG", 1)

system = vector + scalar

# Test and trial functions
(v, q) = TestFunctions(system)
(w, r) = TrialFunctions(system)

# Load primal solutions
uh = Function(vector)
ph = Function(scalar)

# Functions to store solutions
Psi = Constant(mesh, (0.0, 0.0))          # Needs to depend on the goal
w1 = project(Psi, vector)
r1 = project(Constant(mesh, 0.0), scalar) # Needs to be solved initially
w0 = Function(vector)
r0 = Function(scalar)
#gt = Constant(mesh, (-0.0001, 0.0))           # Needs to depend on the goal
cutoff = Cutoff(scalar)
tgt = Function(vector, ("1.0", "0"))

u_store = loadtxt('stored/CSS_u_store.txt')
p_store = loadtxt('stored/CSS_p_store.txt')

# Backward Euler (?) (stable)
a_dual = inner(v, w)*dx + k*a_tilde(uh, ph, v, q, w, r)
L_dual = inner(v, w1)*dx + k*cutoff*inner(sigma(v, q)*n, tgt)*ds # Optimise for shear component
#+ k*inner(v, gt)*dx*cutoff                                              # on top surface

# # Crank-Nicholson (?) (unstable)
# a_dual = inner(v, w)*dx + (k/2.0)*a_tilde(uh, ph, v, q, w, r)
# L_dual = inner(v, w1)*dx + k*inner(v, gt)*dx*cutoff - (k/2.0)*a_tilde(uh, ph, v, q, w1, r1)

# Create Dirichlet (no-slip) boundary conditions for velocity
gd0 = Constant(mesh, (0, 0))
bcd0 = [DirichletBC(vector, gd0, DirichletBoundary())]

# Create inflow boundary condition for pressure
gd1 = Constant(mesh, 0)
bcd1 = [DirichletBC(scalar, gd1, InflowBoundary())]

# Create outflow boundary condition for pressure
gd2 = Constant(mesh, 0)
bcd2 = [DirichletBC(scalar, gd2, OutflowBoundary())]

bcs_dual = bcd0 + bcd1 + bcd2

# If the pressure boundary condition was not set explicitly in the
# primal, but imposed weakly as a term on the rhs of the bilinear
# form, I don't think it must be imposed as a Dirichlet condition in
# the dual.

# Create files for storing the solution in VTK format
wfile_pvd = File("paraview/css/css_dual_velocity.pvd")
rfile_pvd = File("paraview/css/css_dual_pressure.pvd")

# Time loop
t = T
N = int(ceil(T / k))
j = N - 1

# Variables to store the dual solution
w_store = w1.vector().array()
r_store = r1.vector().array()

while t >= 0:

    ph.vector()[:] = p_store[j, :]
    uh.vector()[:] = u_store[j, :]

#    plot(uh, title='Primal velocity')

    pde_dual = VariationalProblem(a_dual, L_dual, bcs_dual)
    (w0, r0) = pde_dual.solve().split(True)

    # FIXME: Match the primal and dual times exactly
    print "Computed the solution at step", j, "where t =", t, "(", t/T*100, "% )"

#     plot(r1)
#     plot(w1)

    # Propagate values to previous time step
    j -= 1
    t -= k

    wfile_pvd << w0
    rfile_pvd << r0

    w_store = vstack([w0.vector().array(), w_store])
    r_store = vstack([r0.vector().array(), r_store])

    w1.assign(w0)
    r1.assign(r0)

savetxt('stored/CSS_w_store.txt', w_store, fmt="%12.6G")
savetxt('stored/CSS_r_store.txt', r_store, fmt="%12.6G")
