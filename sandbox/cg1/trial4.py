from dolfin import *
from numpy import array, loadtxt

parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["optimize"] = True

mesh = UnitSquare(64, 64, "crossed")
vector = VectorFunctionSpace(mesh, "CG", 1)

B  = Expression(("0.0", "0.0"))
T  = Expression(("0.0", "0.0"))

mixed_element = MixedFunctionSpace([vector, vector])
V = TestFunction(mixed_element)
dU = TrialFunction(mixed_element)
U = Function(mixed_element)
U0 = Function(mixed_element)

xi, eta = split(V)
u, v = split(U)
u0, v0 = split(U0)

u_plot = v_plot = Function(vector)

# Load initial displacement from file
_u0 = loadtxt("u0.txt")[:]
U0.vector()[0:len(_u0)] = _u0[:]

u_mid = 0.5*(u0 + u)
v_mid = 0.5*(v0 + v)

mu    = Constant(3.85)
lmbda = Constant(5.77)

rho0 = Constant(1.0)
dt = Constant(0.001)

def PE(u):
    return assemble(inner(0.5*epsilon(u), sigma(u))*dx, mesh=mesh)

def KE(v):
    return assemble(0.5*rho0*inner(v, v)*dx, mesh=mesh)

def TE(u, v):
    return PE(u) + KE(v)

def epsilon(v):
    return 0.5*(grad(v) + grad(v).T)

def sigma(v):
    return 2.0*mu*epsilon(v) + lmbda*tr(epsilon(v))*Identity(v.cell().d)

L = rho0*inner(v - v0, xi)*dx + dt*inner(sigma(u_mid), sym(grad(xi)))*dx \
    - dt*inner(B, xi)*dx - dt*inner(T, xi)*ds \
    + inner(u - u0, eta)*dx - dt*inner(v_mid, eta)*dx 
a = derivative(L, U, dU)

t = 0.0
T = 2.0

problem = VariationalProblem(a, L, nonlinear = True)

file = File("displacement.pvd")
u0, v0 = U0.split(True)

print "Energies: ", "0.0\t", PE(u0), "\t", KE(v0)
file << u0

while t < T:

    t = t + float(dt)

    problem.solve(U)
    u, v = U.split()

    u_plot.assign(u)
    plot(u_plot, mode='displacement')
    

    print "Energies: ", t, "\t", PE(u), "\t", KE(v)
    file << u

    U0.assign(U)
