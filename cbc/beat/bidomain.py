from dolfin import *
from cell_model import I, F
from conductivities import M_i, M_ie

def Dt(u, u0, dt):
    " Backward Euler"
    dt = Constant(dt)
    return (1.0/dt)*(u - u0)

# Parameters for time
T = 2.0
dt = 0.01

# Parameters for space
n = 32
mesh = UnitSquare(n, n)

# Definition of numerical scheme
V = FunctionSpace(mesh, "CG", 1)
Q = FunctionSpace(mesh, "CG", 1)
S = FunctionSpace(mesh, "DG", 0)
R = FunctionSpace(mesh, "R", 0)
W = MixedFunctionSpace([V, Q, S, R])

y = Function(W)

# Unknowns
(v, u, s, c_u) = split(y)

# Test functions
(w, q, r, d_u) = TestFunctions(W)

# Initial conditions:
v0 = Function(V)
s0 = Function(V)

# Applied (body) current
I_e = Constant(0.0)

# Applied boundary current (M_i grad(v) + M_i grad(u))*n = g
class BoundaryStimulus(Expression):
    def eval(self, values, x):
        if x[0] == 0.0 and x[1] == 0.5 and t > 0.2 and t < 0.4:
            values[0] = 10.0
        else:
            values[0] = 0.0
g = BoundaryStimulus()

# Parabolic equation
F0 = (Dt(v, v0, dt)*w
      + inner(M_i(grad(v)), grad(w)) + inner(M_i(grad(u)), grad(w))
      + I(v, s)*w)*dx \
      + g*w*ds

# Elliptic equation
F1 = (inner(M_i(grad(v)), grad(q)) + inner(M_ie(grad(u)), grad(q))
      + I_e*q)*dx \
      + g*q*ds

# State variables
F2 = (Dt(s, s0, dt)*r - F(v, s)*r)*dx

# Non-singularize (ground system)
F3 = (u*d_u + q*c_u)*dx

# Full system
E = F0 + F1 + F2 + F3

dy = TrialFunction(W)
dE = derivative(E, y, dy)

solution = Function(W)

t = 0.0

file = File("results/membrane.pvd")

while (t <= T):

    # Update sources to new time step
    g.t = t

    # Define (non-linear) variational problem
    pde = VariationalProblem(dE, E, nonlinear=True)

    # Solve
    pde.solve(y)

    # Extract interesting variables
    (v, u, s, c_u) = y.split()

    # Plot/Store solutions
    #plot(v, title="Membrane potential")
    file << v

    # Update solutions at previous time-step
    v0.assign(v)
    s0.assign(s)

    # Update time step
    t += dt

