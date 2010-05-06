## The primal problem is solved using the consistent splitting
## scheme, and the velocity and pressure solutions are stored at each
## time step to be later used in the dual problem solution.

from dolfin import *
from css_common import *

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

for t in t_range:

    useries.store(u0.vector(), t)
    pseries.store(p0.vector(), t)

    ufile << u0
    pfile << p0
		
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

    print "Computed the solution at t = ", t, "(", t/T*100, "% )"

    u0.assign(u1)
    p0.assign(p1)
