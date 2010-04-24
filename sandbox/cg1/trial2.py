'''This program is experiments with a cG(1) Newton-Raphson scheme for
the Navier Stokes equations'''

from dolfin import *

# Optimise both the construction and compilation of the forms
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["optimize"] = True

# Define the geometry
mesh = UnitSquare(32, 32)
n = FacetNormal(mesh)

# Define the function spaces
vector = VectorFunctionSpace(mesh, "CG", 2)
scalar = FunctionSpace(mesh, "CG", 1)
real = FunctionSpace(mesh, "R", 0)

mixed_element = MixedFunctionSpace([vector, scalar, real])

# Test, trial and other functions
V = TestFunction(mixed_element)
dU = TrialFunction(mixed_element)
U = Function(mixed_element)
U0 = Function(mixed_element)

xi, eta, zeta = split(V)
v, p, c = split(U)
v0, p0, c0 = split(U0)

# External driver functions
b = Expression(("0.0", "0.0"))
pbar = Expression(("0.0", "0.0"))

# Boundary conditions
noslip = Expression(("0.0", "0.0"))
boundary_noslip = compile_subdomains("x[0] == 0.0 || x[0] == 1.0 || x[1] == 0.0")
bc_noslip = DirichletBC(mixed_element.sub(0), noslip, boundary_noslip)

driver = Expression(("1.0", "0.0"))
boundary_driver = compile_subdomains("x[1] == 1.0")
bc_driver = DirichletBC(mixed_element.sub(0), driver, boundary_driver)

# Select interpolation for time-stepping
alpha = 1.0
beta = 1.0
v_int = (1.0 - alpha)*v0 + alpha*v
p_int = (1.0 - beta)*p0 + beta*p

# Material and time-stepping parameters
mu = Constant(1.0/8.0*1.e-2)
rho = Constant(1.0)

# Time-stepping parameters
dt = Constant(0.01)
t = 0.0
T = 2.5

# Cauchy stress of the fluid
sigma_int = 2.0*mu*sym(grad(v_int)) - p_int*Identity(p_int.cell().d)

# Define the variational problem
L = rho*inner(v - v0, xi)*dx + dt*rho*inner(grad(v_int)*v_int, xi)*dx \
    + dt*inner(sigma_int, sym(grad(xi)))*dx - dt*inner(b, xi)*dx \
    + dt*inner(pbar, xi)*ds - dt*inner(mu*grad(v_int).T*n, xi)*ds \
    + dt*div(v_int)*eta*dx + dt*p*zeta*dx + dt*c*eta*dx
a = derivative(L, U, dU)

problem = VariationalProblem(a, L, [bc_noslip, bc_driver], nonlinear=True)
problem.parameters["newton_solver"]["absolute_tolerance"] = 1e-14
problem.parameters["newton_solver"]["relative_tolerance"] = 1e-12
problem.parameters["newton_solver"]["maximum_iterations"] = 50

# Create files to store the solutions
file_v = File("velocity.pvd")
file_p = File("pressure.pvd")
file_c = File("lagrange.pvd")

while t < T:

    t = t + float(dt)

    problem.solve(U)
    v, p, c = U.split()
    
    file_v << v
    file_p << p
    file_c << c

    U0.assign(U)
