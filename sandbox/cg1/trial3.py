from dolfin import *
from numpy import array, loadtxt

parameters["form_compiler"]["cpp_optimize"] = True
# parameters["form_compiler"]["optimize"] = True

mesh = UnitSquare(8, 8)
vector = VectorFunctionSpace(mesh, "CG", 1)

B  = Expression(("0.0", "0.0"))
# T  = Expression(("t < tc ? tx*(1-cos(t/n))*0.5 : tx ;", "0.0"))
# T.tx = 1e-3
# T.n = 16
# T.tc = T.n*DOLFIN_PI
T = Expression(("0.1","0.0"))

mixed_element = MixedFunctionSpace([vector, vector])
V = TestFunction(mixed_element)
dU = TrialFunction(mixed_element)
U = Function(mixed_element)
U0 = Function(mixed_element)

xi, eta = split(V)
u, v = split(U)
u0, v0 = split(U0)

clamp = Constant((0.0, 0.0))
bottom = compile_subdomains("x[1] == 0")
left = compile_subdomains("x[0] == 0")

boundary = MeshFunction("uint", mesh, 1)
boundary.set_all(0)
left.mark(boundary, 1)

bcl = DirichletBC(mixed_element.sub(0), clamp, bottom)

# u_mid = u
# v_mid = v

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

epsilon = 0.5*(grad(xi) + grad(xi).T)
sigma = 2.0*mu*sym(grad(u_mid)) + lmbda*tr(sym(grad(u_mid)))*Identity(u_mid.cell().d)

rho0 = Constant(1.0)
dt = Constant(0.1)


# L = rho0*inner(v - v0, xi)*dx + dt*inner(sigma, epsilon)*dx \
#     - dt*inner(B, xi)*dx - dt*inner(T, xi)*ds(1) \
#     + inner(u - u0, eta)*dx - dt*inner(v_mid, eta)*dx 
# a = derivative(L, U, dU)


L = rho0*inner(v - v0, xi)*dx + dt*inner(P, grad(xi))*dx \
    - dt*inner(B, xi)*dx - dt*inner(T, xi)*ds(1) \
    + inner(u - u0, eta)*dx - dt*inner(v_mid, eta)*dx 
a = derivative(L, U, dU)

t = 0.0
eT = 30.0

problem = VariationalProblem(a, L, bcl, exterior_facet_domains=boundary, nonlinear = True)

file_u = File("displacement.pvd")
file_v = File("velocity.pvd")

while t < eT:

    t = t + float(dt)
    T.t = t

    problem.solve(U)
    u, v = U.split()
#    plot(u, mode="displacement")
    file_u << u
    file_v << v

    U0.assign(U)
