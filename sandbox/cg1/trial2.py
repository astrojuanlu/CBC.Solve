from dolfin import *

parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["optimize"] = True

mesh = UnitSquare(32, 32)
n = FacetNormal(mesh)
vector = VectorFunctionSpace(mesh, "CG", 2)
scalar = FunctionSpace(mesh, "CG", 1)

b = Expression(("0.0", "0.0"))
pbar = Expression(("-1.0*(1.0 - x[0])", "0.0"))

mixed_element = MixedFunctionSpace([vector, scalar])
V = TestFunction(mixed_element)
dU = TrialFunction(mixed_element)
U = Function(mixed_element)
U0 = Function(mixed_element)

xi, eta = split(V)
v, p = split(U)
v0, p0 = split(U0)

noslip = Expression(("0.0", "0.0"))
boundary = compile_subdomains("x[1] == 0.0 || x[1] == 1.0")
bc = DirichletBC(mixed_element.sub(0), noslip, boundary)

v_mid = 0.5*(v0 + v)
p_mid = 0.5*(p0 + p)

mu = Constant(1.0/8.0)
sigma_mid = 2.0*mu*sym(grad(v_mid)) - p_mid*Identity(p_mid.cell().d)

rho = Constant(1.0)
dt = Constant(0.1)

L = rho*inner(v - v0, xi)*dx + dt*rho*inner(grad(v_mid)*v_mid, xi)*dx \
    + dt*inner(sigma_mid, sym(grad(xi)))*dx - dt*inner(b, xi)*dx \
    + dt*inner(pbar, xi)*ds - dt*inner(mu*grad(v_mid).T*n, xi)*ds \
    + dt*div(v_mid)*eta*dx
a = derivative(L, U, dU)

t = 0.0
T = 2.5

problem = VariationalProblem(a, L, bc, nonlinear = True)
problem.parameters["newton_solver"]["absolute_tolerance"] = 1e-14
problem.parameters["newton_solver"]["relative_tolerance"] = 1e-12
problem.parameters["newton_solver"]["maximum_iterations"] = 50


file = File("velocity.pvd")

while t < T:

    t = t + float(dt)

    problem.solve(U)
    v, p = U.split()
    file << v

    U0.assign(U)
