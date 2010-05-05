## The dual problem is solved using the stored velocity and pressure
## solutions of the primal problem.

from dolfin import *
from css_common import *
from math import ceil

# Define a portion of the linearised dual bilinear form
def a_tilde(uh, ph, v, q, w, r):
    return inner(grad(uh)*v + grad(v)*uh, w)*dx \
        + inner(sigma(v, q), sym(grad(w)))*dx \
        - nu*inner(sym(grad(v))*n, w)*ds \
        + div(v)*r*dx

# Delineate goal functional region
class Cutoff(Expression):
    def eval(self, values, x):
        if (x[1] > 1.0 - DOLFIN_EPS and x[0] > 0.0 + DOLFIN_EPS and x[0] < 1.0 - DOLFIN_EPS):
            values[0] = 1.0
        else:
            values[0] = 0.0

# Test and trial functions
(v, q) = TestFunctions(system)
(w, r) = TrialFunctions(system)

# Load primal solutions
uh = Function(vector)
ph = Function(scalar)

# Functions to store solutions
Psi = Constant((0.0, 0.0))          # Needs to depend on the goal
w1 = project(Psi, vector)
r1 = project(Constant(0.0), scalar) # Needs to be solved initially
w0 = Function(vector)
r0 = Function(scalar)
#gt = Constant(mesh, (-0.0001, 0.0))           # Needs to depend on the goal
cutoff = Cutoff(scalar)
tgt = Expression(("1.0", "0"))

# Backward Euler (?) (stable)
a_dual = inner(v, w)*dx + k*a_tilde(uh, ph, v, q, w, r)
L_dual = inner(v, w1)*dx + k*cutoff*inner(sigma(v, q)*n, tgt)*ds # Optimise for shear component
#+ k*inner(v, gt)*dx*cutoff                                              # on top surface

# # Crank-Nicholson (?) (unstable)
# a_dual = inner(v, w)*dx + (k/2.0)*a_tilde(uh, ph, v, q, w, r)
# L_dual = inner(v, w1)*dx + k*inner(v, gt)*dx*cutoff - (k/2.0)*a_tilde(uh, ph, v, q, w1, r1)

# Create Dirichlet (no-slip) boundary conditions for velocity
gd0 = Constant((0.0, 0.0))
bcd0 = [DirichletBC(vector, gd0, noslipboundary)]

# Create inflow boundary condition for pressure
gd1 = Constant(0.0)
bcd1 = [DirichletBC(scalar, gd1, inflowboundary)]

# Create outflow boundary condition for pressure
gd2 = Constant(0.0)
bcd2 = [DirichletBC(scalar, gd2, outflowboundary)]

bcs_dual = bcd0 + bcd1 + bcd2

# If the pressure boundary condition was not set explicitly in the
# primal, but imposed weakly as a term on the rhs of the bilinear
# form, I don't think it must be imposed as a Dirichlet condition in
# the dual.

# Time loop
t = T
N = int(ceil(T / k))
j = N - 1

# Variables to store the dual solution
w_store = w1.vector().array()
r_store = r1.vector().array()

while t >= 0:

#    plot(w0)
#    plot(r0)

    useries.retrieve(uh.vector(), t)
    pseries.retrieve(ph.vector(), t)

    pde_dual = VariationalProblem(a_dual, L_dual, bcs_dual)
    (w0, r0) = pde_dual.solve().split(True)

    print "Computed the solution at step", j, "where t =", t, "(", t/T*100, "% )"

    j -= 1
    t -= k

    wseries.store(w0.vector(), t)
    rseries.store(w0.vector(), t)

    wfile << w0
    rfile << r0

    w1.assign(w0)
    r1.assign(r0)

#savetxt('stored/CSS_w_store.txt', w_store, fmt="%12.6G")
#savetxt('stored/CSS_r_store.txt', r_store, fmt="%12.6G")
