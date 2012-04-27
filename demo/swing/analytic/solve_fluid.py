import math
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
set_log_level(WARNING)

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

# Exact mesh velocity
P_M = Expression(cpp_P_M, degree=1)
P_M.C = C

# Exact mesh movement
U_M = Expression(cpp_U_M, degree=1)
U_M.C = C

ref = 1
n = 4*2**2
dt = 0.01/2**(ref + 3)
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
v_to_plot = Function(V)
p_to_plot = Function(Q)
while (t < T):

    info_blue("t = %g" % t)
    # Update sources
    f_F.t = t
    U_M.t = t
    P_M.t = t
    u_F.t = t
    p_F.t = t

    alpha = Constant(1.0)
    F = (1.0/k*inner(u - u_, v)*dx + inner(grad(u)*(u - alpha*P_M), v)*dx
         + inner(sigma(u, p), sym(grad(v)))*dx
         + div(u)*q*dx
         - inner(f_F, v)*dx
         + p*s*dx + q*r*dx)

    # Solve problem
    solve(F == 0, w, bcs)

    # Update solutions
    w0.assign(w)

    # Incrementally move mesh
    d = project(U_M, V)
    U_M.t = t - dt
    d_ = project(U_M, V)
    d.vector().axpy(-1, d_.vector())
    mesh.move(d)

    print "||u - u_h|| = ", math.sqrt(assemble(inner(u - u_F, u - u_F)*dx))
    print "||p - p_h|| = ", math.sqrt(assemble(inner(p - p_F, p - p_F)*dx))

    # Step forward in time
    t += dt

v_to_plot.assign(w.split()[0])
p_to_plot.assign(w.split()[1])
plot(v_to_plot, title="Velocity")
plot(p_to_plot, title="Pressure")
plot(p_F, title="Exact pressure", mesh=mesh)
plot(u_F, title="Exact velocity", mesh=mesh)


interactive()
