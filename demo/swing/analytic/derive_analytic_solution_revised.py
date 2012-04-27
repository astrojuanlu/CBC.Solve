"This script checks that the analytical solution satisfies the FSI problem"

__author__ = "Anders Logg"
__copyright__ = "Copyright (C) 2012 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

# Modified by Marie E. Rognes
# Last changed: 2012-04-27

from sympy import *

# Turn this to True to get more diagnostics output
debug = True

def underline(s): print s + "\n" + "-"*len(s)

# Declare symbols
x, y, X, Y, t = symbols("x y X Y t")
C = symbols("C")

# Material parameters
mu = Integer(1)
lmbda = Integer(2)
nu = Integer(1)
rho_S = Integer(100)

# Location of FSI boundary
Y_ = Rational(1, 2)

# Methodology:

# Step 1: Choose U_M = U_S and such that U_S = U_M = 0 on the
# structure boundary, and non-zero on the fsi boundary and 0, and with
# zero time-derivative at t = 0.
U_S = Matrix([0, C*X*(1 - X)*sin(pi*Y)*sin(pi*t)**2])
U_M = U_S

# One can then derive the movement of the fsi boundary from the
# structure displacement:
y_ = Y_ + U_S[1].subs(Y, Y_).subs(X, x)

# Step 2: Define the fluid velocity by u_F(x) = d/dt U_S(X, Y=1/2) =
# P_S(X, Y=1/2)
P_S = Matrix([diff(U_S[0], t), diff(U_S[1], t)])
u_F = P_S.subs(Y, Y_).subs(X, x)
# (This does not have to depend on Y, hence the replacement of Y w/ Y_)

# Step 3: Define your favorite pressure, preferably 0 at t = 0 and
# such that its average value is zero.
#p_F = - 2*C**2*(1 - 2*x)**2*sin(pi*t)**3*(sin(pi*t) + pi*cos(pi*t))
p_F = - C**2*(1 - 2*x)*sin(pi*t)**2
P_F = p_F.subs(x, X).subs(y, Y)

# Normal direction at FSI boundary follows from the definitions
n = Matrix([-simplify(diff(U_S[1], X)), 1]).subs(Y, Y_).subs(X, x)
n = n / sqrt(n[0]**2 + n[1]**2)

# Normal direction on top and reference fsi boundary
N = Matrix([0, 1])

# Print solutions
underline("Analytical solutions")

print "u_F =\n", u_F, "\n"
print "p_F =\n", p_F, "\n"
print "U_S =\n", U_S, "\n"
print "U_M =\n", U_M, "\n"
print

# Compute gradients
grad_u_F = Matrix([[simplify(diff(u_F[0], x)), simplify(diff(u_F[0], y))],
                   [simplify(diff(u_F[1], x)), simplify(diff(u_F[1], y))]])
Grad_U_S = Matrix([[simplify(diff(U_S[0], X)), simplify(diff(U_S[0], Y))],
                   [simplify(diff(U_S[1], X)), simplify(diff(U_S[1], Y))]])
Grad_U_M = Matrix([[simplify(diff(U_M[0], X)), simplify(diff(U_M[0], Y))],
                   [simplify(diff(U_M[1], X)), simplify(diff(U_M[1], Y))]])

# Print gradients
if debug:
    underline("Gradients")
    print "grad(u_F) =\n", grad_u_F, "\n"
    print "Grad(U_S) =\n", Grad_U_S, "\n"
    print "Grad(U_M) =\n", Grad_U_M, "\n"
    print

# Compute Cauchy stress for fluid
I = eye(2)
sigma_F = nu*(grad_u_F + grad_u_F.T) - p_F*I

# Compute reference and Cauchy stress for structure
F_S = I + Grad_U_S
J_S = F_S.det()
E_S = Rational(1, 2)*(F_S.T*F_S - I)
Sigma_S = F_S*(2*mu*E_S + lmbda*E_S.trace()*I)

sigma_S = Sigma_S*F_S.T / J_S
sigma_S = sigma_S.subs(X, x).subs(Y, y)

# Compute reference stress tensor for fluid.
# Note that F_M = F_S by  construction of U_M = U_S
F_M_inv = F_S.inv()
U_F = u_F.subs(x, X).subs(y, Y)
Grad_U_F = Matrix([[simplify(diff(U_F[0], X)), simplify(diff(U_F[0], Y))],
                   [simplify(diff(U_F[1], X)), simplify(diff(U_F[1], Y))]])
Sigma_F = nu*(Grad_U_F * F_M_inv + F_M_inv.T * Grad_U_F.T) - P_F*I

# Compute boundary tractions on fsi boundary in reference frame:
G_F = J_S*Sigma_F*F_M_inv*N
G_F = G_F.subs(Y, Y_)
G_F = Matrix([simplify(G_F[0]), simplify(G_F[1])])
G_S = Sigma_S*N
G_S = G_S.subs(Y, Y_)
G_S = Matrix([simplify(G_S[0]), simplify(G_S[1])])

# Compute top boundary stress for fluid
underline("Deriving top boundary stress for fluid")
sigma_n_top = sigma_F*Matrix([0, 1])
sigma_n_top = sigma_n_top.subs(y, 1.0)
print sigma_n_top
print

# Compute boundary traction in reference frame
underline("Deriving additional reference boundary traction G_0")
G_0 = G_S - G_F
G_0 = Matrix([simplify(G_0[0]), simplify(G_0[1])])
print G_0
print

# # Compute boundary tractions on fsi boundary in physical frame
# g_F = sigma_F*n
# g_F = g_F.subs(y, y_)
# g_S = sigma_S*n
# g_S = g_S.subs(y, y_)

# # Compute boundary traction in physical frame
# underline("Deriving additional boundary traction g_0")
# g_0 = g_S - g_F
# g_0 = Matrix([simplify(g_0[0]), simplify(g_0[1])])
# print g_0
# print

# Derive right-hand side f_F such that the Navier-Stokes equations are
# satisfied
underline("Deriving right-hand side for the Navier--Stokes equations")
div_sigma_F = Matrix([diff(sigma_F[0, 0], x) + diff(sigma_F[0, 1], y),
                      diff(sigma_F[1, 0], x) + diff(sigma_F[1, 1], y)])
dot_u_F = Matrix([diff(u_F[0], t), diff(u_F[1], t)])
grad_u_F_u = grad_u_F*u_F

f_F = dot_u_F + grad_u_F_u - div_sigma_F
f_F = Matrix([simplify(f_F[0]), simplify(f_F[1])])
print f_F
print

# Check that the hyperelastic equation is satisfied
underline("Deriving right-hand side for the hyperelastic equation")
Div_Sigma_S = Matrix([diff(Sigma_S[0, 0], X) + diff(Sigma_S[0, 1], Y),
                      diff(Sigma_S[1, 0], X) + diff(Sigma_S[1, 1], Y)])
ddot_U_S = Matrix([diff(diff(U_S[0], t), t), diff(diff(U_S[1], t), t)])
f_S = rho_S*ddot_U_S - Div_Sigma_S
f_S = Matrix([simplify(f_S[0]), simplify(f_S[1])])
print f_S
print

# Check that the mesh equation is satisfied
underline("Deriving right-hand side for mesh equation")
Sigma_M = mu*(Grad_U_M + Grad_U_M.T) + lmbda*Grad_U_M.trace()*I
Div_Sigma_M = Matrix([diff(Sigma_M[0], X) + diff(Sigma_M[1], Y),
                      diff(Sigma_M[2], X) + diff(Sigma_M[3], Y)])
dot_U_M = Matrix([diff(U_M[0], t), diff(U_M[1], t)])
f_M = dot_U_M - Div_Sigma_M
f_M = Matrix([collect(f_M[0], C*pi*sin(pi*t)**2*cos(pi*Y)),
              collect(f_M[1], C*sin(pi*Y)*sin(pi*t))])
print f_M
print

# Compute reference value of functional: integrated Y-displacement
T = Rational(1, 2)
M = integrate(integrate(integrate(U_S[1], (X, 0, 1)), (Y, 0, Y_)), (t, 0, T))
underline("Value of reference functional")
print "M = ", simplify(M)

if debug:

    # Check continuity of mesh
    underline("Checking continuity of mesh: U_S - U_M")
    print U_S.subs(Y, Y_) - U_M.subs(Y, Y_)
    print

    # Check continuity of velocity
    underline("Checking continuity of velocity: u_F - p_S")
    p_S = U_S.diff(t).subs(Y, Y_).subs(X, x)
    print u_F - p_S
    print

    # Check continuity of boundary traction
    underline("Checking continuity of boundary traction: G_S - G_F - G_0")
    r = G_S - G_F - G_0
    r = Matrix([simplify(r[0]), simplify(r[1])])
    print r
    print

    underline("Checking that u_F is divergence free")
    div_u_F = diff(u_F[0], x) + diff(u_F[1], y)
    print div_u_F
    print

    underline("Average values of pressures in reference and current frame")
    average_P_F = integrate(integrate(P_F, (X, 0, 1)), (Y, Y_, 1))
    average_p_F = integrate(integrate(p_F, (x, 0, 1)), (y, y_, 1))
    print average_P_F
    print average_p_F
    print

    # Print location of FSI boundary
    underline("Location of FSI boundary")
    print "Y_ =", Y_
    print "y_ =", y_
    print

    underline("Normal of FSI boundary")
    print "n = ", n
    print "N = ", N
    print

