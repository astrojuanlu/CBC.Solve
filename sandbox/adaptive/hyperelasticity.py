from dolfin import *
from numpy import array, loadtxt

parameters["form_compiler"]["cpp_optimize"] = True

mesh = UnitSquare(16, 16)
vector = VectorFunctionSpace(mesh, "CG", 1)

B  = Expression(("0.0", "0.0"))
T  = Expression(("0.01*t", "0.0"))

mixed_element = MixedFunctionSpace([vector, vector])
V = TestFunction(mixed_element)
dU = TrialFunction(mixed_element)
U = Function(mixed_element)
U0 = Function(mixed_element)

ZU = Function(mixed_element)
ZU0 = Function(mixed_element)

xi, eta = split(V)
u, v = split(U)
u0, v0 = split(U0)
zu, zv = split(ZU)
zu0, zv0 = split(ZU0)

clamp = Constant((0.0, 0.0))
bottom = compile_subdomains("x[1] == 0")
bc = DirichletBC(mixed_element.sub(0), clamp, bottom)

u_int = u
v_int = v

mu    = Constant(3.85)
lmbda = Constant(5.77)

def P(u):
    I = Identity(u.cell().d)
    F = I + grad(u)
    C = F.T*F
    E = (C - I)/2
    E = variable(E)
    psi = lmbda/2*(tr(E)**2) + mu*tr(E*E)
    S = diff(psi, E)
    P = F*S
    return P

rho0 = Constant(1.0)
dt = Constant(0.01)

L = rho0*inner(v - v0, xi)*dx + dt*inner(P(u_int), grad(xi))*dx \
    - dt*inner(B, xi)*dx - dt*inner(T, xi)*ds \
    + inner(u - u0, eta)*dx - dt*inner(v_int, eta)*dx
a = derivative(L, U, dU)

t = 0.0
end_time = 10.0

problem = VariationalProblem(a, L, bc, nonlinear = True)

plot_file = File("displacement.pvd")
displacement_series = TimeSeries("displacement")
velocity_series = TimeSeries("velocity")

# # Primal problem

# while t < end_time:

#     t = t + float(dt)
#     T.t = t

#     problem.solve(U)
#     u, v = U.split(True)
    
#     plot_file << u
#     displacement_series.store(u.vector(), t)
#     velocity_series.store(v.vector(), t)

#     U0.assign(U)

# Adjoint problem

U_h = Function(mixed_element)
ZU = Function(mixed_element)
ZU0 = Function(mixed_element)

zu, zv = split(ZU)
zu0, zv0 = split(ZU0)
u_h, v_h = U_h.split(True)

F_adjoint = - rho0*inner(zv - zv0, xi)*dx - inner(zu0 - zu, eta)*dx \
            - dt*inner(zv, eta)*dx
goal = dt*xi[0]*dx

# + inner(grad(zu), P(xi))*dx
a_adjoint = lhs(F_adjoint)
L_adjoint = rhs(F_adjoint) + goal


problem_adjoint = VariationalProblem(a_adjoint, L_adjoint, homogenize(bc))

plot_file_adjoint = File("adjoint_displacement.pvd")

t = end_time

while t  >= 0:

    t = t - float(dt)

    displacement_series.retrieve(u_h.vector(), t)
    velocity_series.retrieve(v_h.vector(), t)

    problem_adjoint.solve(ZU0)
    zu0, zv0 = ZU0.split(True)

    plot_file_adjoint << zu0

    ZU.assign(ZU0)

    plot(zu0, mode='displacement')

interactive()
