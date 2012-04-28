import math
from dolfin import *
from right_hand_sides_revised import *
#set_log_level(WARNING)

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
P_M = Expression(cpp_P_M, degree=2)
P_M.C = C

# Exact mesh movement
U_M = Expression(cpp_U_M, degree=2)
U_M.C = C

ref = 0
n = 16
dt = 0.005
T = 0.1
mesh = Rectangle(0.0, 0.5, 1.0, 1.0, 2*n, n)
mesh0 = Rectangle(0.0, 0.5, 1.0, 1.0, 2*n, n)

def sigma(u, p):
    I = Identity(mesh.geometry().dim())
    return 2*nu*sym(grad(u)) - p*I

V_0 = VectorFunctionSpace(mesh0, "CG", 2)

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

t = dt
v_to_plot = Function(V)
p_to_plot = Function(Q)
while (t < T):

    t0 = t - dt
    tmid = 0.5*(t0 + t)

    info_blue("t = %g" % t)

    # Update sources in forms and bcs
    f_F.t = tmid # This makes real difference for the pressure.
    P_M.t = t    # tmid or t same same
    U_M.t = t

    mesh_velocity_on_mesh_0 = project(P_M, V_0)
    mesh_velocity = Function(V)
    mesh_velocity.vector()[:] = mesh_velocity_on_mesh_0.vector()

    #plot(U_M, title="mesh_displacement", mesh=mesh0)
    #plot(mesh_velocity_on_mesh_0, title="mesh_velocity on mesh0", mesh=mesh0)
    #plot(mesh_velocity, title="mesh_velocity", mesh=mesh0)

    u_F.t = t    # This makes real difference for the pressure.

    u_mid = 0.5*(u + u_)
    F = (1.0/k*inner(u - u_, v)*dx
         + inner(grad(u_mid)*(u_mid - mesh_velocity), v)*dx
         + inner(sigma(u_mid, p), sym(grad(v)))*dx
         + div(u_mid)*q*dx
         - inner(f_F, v)*dx
         + p*s*dx + q*r*dx)

    foo = "(near(x[1], 1.0) || near(x[0], 0.0) || near(x[0], 1.0))"

    #bcs = [DirichletBC(W.sub(0), mesh_velocity, "on_boundary && !%s" % foo),
    #       DirichletBC(W.sub(0), u_F, "foo"),
    #       ]

    bcs = [DirichletBC(W.sub(0), u_F, "on_boundary")]

    # Solve problem
    solve(F == 0, w, bcs)

    # Update solutions
    w0.assign(w)

    # Incrementally move mesh
    U_M.t = t
    d0 = project(U_M, V_0)
    d = Function(V)
    d.vector()[:] = d0.vector()

    U_M.t = t - dt
    d_0 = project(U_M, V_0)
    d_ = Function(V)
    d_.vector()[:] = d_0.vector()
    d.vector().axpy(-1, d_.vector())
    mesh.move(d)

    # Compute errors
    p_F.t = t
    print "||u|| = ", math.sqrt(assemble(inner(u, u)*dx, mesh=mesh))
    print "||u - u_h|| = ", math.sqrt(assemble(inner(u - u_F, u - u_F)*dx))
    print "||p - p_h|| = ", math.sqrt(assemble(inner(p - p_F, p - p_F)*dx))

    # Step forward in time
    t += dt

v_to_plot.assign(w.split()[0])
p_to_plot.assign(w.split()[1])
plot(v_to_plot, title="Velocity")
plot(p_to_plot, title="Pressure")
plot(mesh, title="Updated mesh", interactive=True)

#plot(p_F, title="Exact pressure", mesh=mesh)
#plot(u_F, title="Exact velocity", mesh=mesh)


