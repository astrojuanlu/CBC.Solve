from dolfin import *
from right_hand_sides_revised import *

cpp_P_M = """
class P_M : public Expression
{
public:

  P_M() : Expression(2), C(0), t(0) {}

  void eval(Array<double>& values, const Array<double>& xx,
            const ufc::cell& cell) const
  {
    const double x = xx[0];
    const double y = xx[1];

    values[0] = 0.0;
    values[1] = C*x*2*sin(pi*t)*cos(pi*t)*pi*(1 - x)*sin(pi*y);
  }

  double C;
  double t;

};
"""

# Parameters
C = 1.0
nu = 1

# Body force
f_F = Expression(cpp_f_F, degree=2)
f_F.C = C

# Exact pressure
p_F = Expression(cpp_p_F, degree=1)
p_F.C = C

# Exact velocity
u_F = Expression(cpp_u_F, degree=2)
u_F.C = C

# Exact mesh movement
P_M = Expression(cpp_P_M, degree=1)
P_M.C = C

# Exact mesh velocity
U_M = Expression(cpp_U_M, degree=1)
U_M.C = C

n = 8
dt = 0.025/4
T = 0.1
mesh = Rectangle(0.0, 0.5, 1.0, 1.0, n, n)

def sigma(u, p):
    I = Identity(mesh.geometry().dim())
    return 2*nu*sym(grad(u)) - p*I

V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)
R = FunctionSpace(mesh, "R", 0)
W = MixedFunctionSpace([V, Q, R])

w0 = Function(W)
w = Function(W)
(u_, p_, r_) = split(w0)
(u, p, r) = split(w)
(v, q, s) = TestFunctions(W)

# Nonlinear forms
k = Constant(dt)

bcs = DirichletBC(W.sub(0), u_F, "on_boundary")

t = dt
while (t < T):

    # Update sources
    f_F.t = t
    U_M.t = t
    P_M.t = t
    u_F.t = t

    F = (1.0/k*inner(u - u_, v)*dx + inner(grad(u)*(u - P_M), v)*dx
         + inner(sigma(u, p), sym(grad(v)))*dx
         + div(u)*q*dx
         - inner(f_F, v)*dx
         + p*s*dx + q*r*dx)

    # Solve problem
    solve(F == 0, w, bcs)

    # Update solutions
    w0.assign(w)

    d = project(u_F, V)
    # Move mesh
    mesh.move(d)

    # Step forward in time
    t += dt

    plot(u, title="Velocity")
    plot(p, title="Pressure")

interactive()
