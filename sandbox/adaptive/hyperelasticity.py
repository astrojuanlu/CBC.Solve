from dolfin import *
from numpy import array, loadtxt

parameters["form_compiler"]["cpp_optimize"] = True

mesh = UnitCube(8, 8, 8)
vector = VectorFunctionSpace(mesh, "CG", 1)

B  = Expression(("0.0", "0.0", "0.0"))
T  = Expression(("0.0", "0.0", "0.0"))

mixed_element = MixedFunctionSpace([vector, vector])
V = TestFunction(mixed_element)
dU = TrialFunction(mixed_element)
U = Function(mixed_element)
U0 = Function(mixed_element)

xi, eta = split(V)
u, v = split(U)
u0, v0 = split(U0)

_u0 = loadtxt("twisty.txt")[:]
U0.vector()[0:len(_u0)] = _u0[:]

clamp = Constant((0.0, 0.0, 0.0))
left = compile_subdomains("x[0] == 0")
bcl = DirichletBC(mixed_element.sub(0), clamp, left)

u_mid = 0.5*(u0 + u)
v_mid = 0.5*(v0 + v)

I = Identity(v.cell().d)
F = I + grad(u_mid)
C = F.T*F
E = (C - I)/2
E = variable(E)

mu    = Constant(3.85)
lmbda = Constant(5.77)

psi = lmbda/2*(tr(E)**2) + mu*tr(E*E)
S = diff(psi, E)
P = F*S

rho0 = Constant(1.0)
dt = Constant(0.01)

L = rho0*inner(v - v0, xi)*dx + dt*inner(P, grad(xi))*dx \
    - dt*inner(B, xi)*dx - dt*inner(T, xi)*ds \
    + inner(u - u0, eta)*dx - dt*inner(v_mid, eta)*dx 
a = derivative(L, U, dU)

t = 0.0
T = 2.0

problem = VariationalProblem(a, L, bcl, nonlinear = True)

plot_file = File("displacement.pvd")
displacement_series = TimeSeries("displacement")
velocity_series = TimeSeries("velocity")

# Primal problem

# while t < T:

#     t = t + float(dt)

#     problem.solve(U)
#     u, v = U.split(True)
    
#     plot_file << u
#     displacement_series.store(u.vector(), t)
#     velocity_series.store(v.vector(), t)

#     U0.assign(U)

# Adjoint problem

u_h = Function(vector)
v_h = Function(vector)

right_boundary = compile_subdomains("x[0] == 1.0")
exterior_facet_domains = MeshFunction("uint", mesh, mesh.topology().dim() - 1)
exterior_facet_domains.set_all(0)
right_boundary.mark(exterior_facet_domains, 1)

a_adjoint = adjoint(a)
L_adjoint = dt*inner(grad(u_h), grad(xi))*ds(1)
# FIXME: Replace u with u_h here
problem_adjoint = VariationalProblem(a_adjoint, L_adjoint, homogenize(bcl), exterior_facet_domains=exterior_facet_domains)

while t < T:
    t = t + float(dt)

    displacement_series.retrieve(u_h.vector(), t)
    velocity_series.retrieve(v_h.vector(), t)

    problem_adjoint.solve(U)
    u, v = U.split(True)

    U0.assign(U)

    plot(u, mode='displacement')
#    plot(v_h)

interactive()
