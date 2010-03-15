from dolfin import *

parameters["form_compiler"]["cpp_optimize"] = True
# parameters["form_compiler"]["optimize"] = True

mesh = UnitCube(8, 8, 8)
V = VectorFunctionSpace(mesh, "CG", 1)

B  = Expression(("0.0", "0.0", "0.0"))
T  = Expression(("0.1", "0.0", "0.0"))

ME = MixedFunctionSpace([V, V])
V = TestFunction(ME)
dU = TrialFunction(ME)
U = Function(ME)
U0 = Function(ME)

xi, eta = split(V)
u, v = split(U)
u0, v0 = split(U0)

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
dt = Constant(0.1)

L = rho0*inner(v - v0, xi)*dx + dt*inner(P, grad(xi))*dx \
    - dt*inner(B, xi)*dx - dt*inner(T, xi)*ds \
    + inner(u - u0, eta)*dx - dt*inner(v_mid, eta)*dx 
a = derivative(L, U, dU)

t = 0.0
T = 1.0

file = File("displacement.pvd");

problem = VariationalProblem(a, L, nonlinear = True)

while t < T:
    t = t + float(dt)

    problem.solve(U)
    plot(u0)

    U0.assign(U)

interactive()
