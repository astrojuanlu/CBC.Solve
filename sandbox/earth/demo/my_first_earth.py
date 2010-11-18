from earth import *

# Define domain
mesh = Mesh("shortening.xml")

# Density
rho = 1.0  # kg/m^3

# Gravitational acceleration
g = 9.81   # m/s^2

# Body force
b = Expression(("0.0", "rho*g"))
b.rho = rho
b.g = g

# Boundary force
h = Expression(("0.0", "0.0"))

# Viscosities
# n = 2.0     # Power-law parameter
# Parameters
#R = 8.3144  # Gas constant (Joule)
#E = 1.0     # Activation energy
#V = 1.0     # Activation volume

# End time
T = 1.0
timestep = 0.01
Delta_t = Constant(timestep)

def tau_prime(tau, v):
    W = spin(v)
    return tau*W - W*tau

V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)
W = V * Q

S = TensorFunctionSpace(mesh, "Quadrature", 2)

solution = Function(W)
(v, p) = (as_vector((solution[0], solution[1])), solution[2])

# Define equations:
nu = Constant(10.0)
mu = Constant(20.0)

# Define effective viscosity
nu_eff = (nu*mu*Delta_t)/(nu + mu*Delta_t)

def D_eff(v, v0, tau0):

    # The strain rate is the sum of three contributions:
    a = strain(v)
    b = (1.0/(2*mu*Delta_t))*tau0
    c = - (1.0/(2*mu))*tau_prime(tau0, v0)

    return a + b + c

# Define relation for tau
def deviatoric_stress(v, v0, tau0):
    return 2.0*nu_eff*D_eff(v, v0, tau0)

# Initial/previous velocity
v0 = Function(V)

# Initial/previous deviatoric stress (or compute from elastic stress)
#tau0 = project(2.0*nu*strain(v0), S)
tau0 = Function(S)

# Define tau as the deviatoric stress as a function of velocity (v) at
# this time and velocity at previous time (v0)
tau = deviatoric_stress(v, v0, tau0)

# Define test functions for variational formulation
(w, q) = TestFunctions(W)

# Define balance of momentum equation
eq1 = (- inner(tau, grad(w)) + p*div(w) - dot(b, w))*dx \
         + dot(h, w)*ds

# Define additional equation (incompressiblity for now).
eq2 = dot(div(v), q)*dx

# Add equations together
F = eq1 + eq2

dw = TrialFunction(W)

uleft = Expression(("0.0", "0.0"))
uright = Expression(("0.0", "0.0"))
bcs = [DirichletBC(W.sub(0), uright, "x[0] == 30.0"),
       DirichletBC(W.sub(0), uleft, "x[0] == 0.0"),
       DirichletBC(W.sub(0).sub(1), 0.0, "x[1] == 0.0")]

# Start at t = \Delta t
t = timestep

file = File("stress_tensor.pvd")

while (t <= T):

    # Update coefficients to new time

    # Define system of equations
    dF = derivative(F, solution, dw)
    pde = VariationalProblem(dF, F, bcs, nonlinear=True)

    # Solve system
    pde.solve(solution)

    # Split computed solution into components
    (v_n, p_n) = solution.split()

    # Compute tau at this timestep
    tau_n = project(tau, S)

    # Update previous velocity and stress
    v0.assign(v_n)
    tau0.assign(tau_n)

    # Write mesh to file
    file << mesh

    # Plot pressure
    plot(p_n)

    # Move mesh
    move(mesh, v_n, timestep)

    # FIXME: Update v0 and tau0 to new mesh
    # Note: v0 is "automatically updated"

    # Update time
    t += timestep


interactive()

