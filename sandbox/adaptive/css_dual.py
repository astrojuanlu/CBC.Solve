## The dual problem is solved using the stored velocity and pressure
## solutions of the primal problem.

from dolfin import *
from css_common_aneurysm import *

# Define a portion of the linearised dual bilinear form
def a_tilde(uh, ph, v, q, w, r):
    return inner(grad(uh)*v + grad(v)*uh, w)*dx \
        + inner(sigma(v, q), sym(grad(w)))*dx \
        - nu*inner(sym(grad(v))*n, w)*ds \
        + div(v)*r*dx

# Delineate goal functional region
class Cutoff(Expression):
    def eval(self, values, x):
        if (x[1] > 2.0 - DOLFIN_EPS):
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
cutoff = Cutoff(scalar)
tgt = as_vector([n[1], -n[0]])

# Backward Euler (stable, unlike Crank-Nicholson)
a_dual = inner(v, w)*dx + k*a_tilde(uh, ph, v, q, w, r)
L_dual = inner(v, w1)*dx + k*cutoff*inner(sigma(v, q)*n, tgt)*ds # Optimise for shear component
                                                                 # on the top surface

# Create Dirichlet (no-slip) boundary conditions for velocity
bcd0 = homogenize(bc0)

# Create inflow boundary condition for pressure
bcd1 = homogenize(bc1)

# Create outflow boundary condition for pressure
bcd2 = homogenize(bc2)

bcs_dual = [bcd0, bcd1, bcd2]

# If the pressure boundary condition was not set explicitly in the
# primal, but imposed weakly as a term on the rhs of the bilinear
# form, I don't think it must be imposed as a Dirichlet condition in
# the dual.

for t in reversed(t_range):

    wseries.store(w1.vector(), T - t) # Store at a fake time 
    rseries.store(r1.vector(), T - t) # Store at a fake time

    wfile << w1
    rfile << r1

    plot(w1)
    plot(r1)

    useries.retrieve(uh.vector(), t)
    pseries.retrieve(ph.vector(), t)

    # Compute dual velocity and pressure
    pde_dual = VariationalProblem(a_dual, L_dual, bcs_dual)
    (w0, r0) = pde_dual.solve().split(True)

    print "Computed the solution at t = ", t, "(", t/T*100, "% )"

    w1.assign(w0)
    r1.assign(r0)
