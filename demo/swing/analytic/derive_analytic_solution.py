__author__ = "Anders Logg"
__copyright__ = "Copyright (C) 2012 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

#
# Modified by Marie E. Rognes
# Last changed: 2012-05-02

from sympy import *

# Turn this to True to get more diagnostics output
debug = True

def underline(s): print s + "\n" + "-"*len(s)

# Declare symbols
x, y, X, Y, t = symbols("x y X Y t")
C = symbols("C")

func_of_time = t

# Material parameters
mu = Integer(1)
lmbda = Integer(2)
nu = Integer(1)
rho_S = Integer(100)

# Follow these steps to derive your favorite analytical solution ...

# Step 0: Decide where the FSI boundary should be in the reference
# frame
X_ = Integer(1)

# Step 1: Choose U_M, U_S such that is satisfies some nice boundary
# and initial conditions
U_S = Matrix([C*Y*(1 - Y)*(1 - cos(t)), 0])
P_S = Matrix([diff(U_S[0], t), diff(U_S[1], t)])
U_M = Matrix([C*X*Y*(1 - Y)*(1 - cos(t)), 0])
P_M = Matrix([diff(U_M[0], t), diff(U_M[1], t)])

# One can then derive the movement of the fsi boundary from the
# structure displacement:
x_ = X_ + U_S[0].subs(X, X_).subs(Y, y)

# Print location of FSI boundary
if debug:
    underline("Location of FSI boundary")
    print "(X_, Y) =", (X_, Y)
    print "(x_, y) =", (x_, y)
    print

# Step 2: Define the fluid velocity by u_F(x) = d/dt U_S(X=1, Y) =
# P_S(X = 1, Y). The substitution here is ok because P_S does not
# depend on X and y == Y.
u_F = P_S.subs(Y, y)

# Step 2a: Define the reference fluid velocity U_F(X) = U_F(x) = u_F(x),
# this works since the fluid velocity only depends on y, but the fsi boundary
#is only being displaced in the x direction.
U_F = u_F

# Step 3: Define your favorite pressure
p_F = 2*C*nu*(1 - x)*sin(t)
P_F = p_F.subs(y, Y).subs(x, X + U_M[0])

# Print solutions
underline("Analytical solutions")
print "U_F =\n", U_F, "\n"
print "u_F =\n", u_F, "\n"
print "p_F =\n", p_F, "\n"
print "P_F =\n", P_F, "\n"
print "U_S =\n", U_S, "\n"
print "P_S =\n", P_S, "\n"
print "U_M =\n", U_M, "\n"
print "P_M =\n", P_M, "\n"
print

# Normal direction at FSI boundary follows from the definitions
n = Matrix([1, -simplify(diff(U_S[0], Y))]).subs(Y, y)
n = n / sqrt(n[0]**2 + n[1]**2)

# Normal direction on reference fsi boundary
N = Matrix([1, 0])

if debug:
    underline("Normal of FSI boundary")
    print "n = ", n
    print "N = ", N
    print

# Compute gradients
grad_u_F = Matrix([[simplify(diff(u_F[0], x)), simplify(diff(u_F[0], y))],
                   [simplify(diff(u_F[1], x)), simplify(diff(u_F[1], y))]])
grad_U_F = Matrix([[simplify(diff(U_F[0], x)), simplify(diff(U_F[0], y))],
                   [simplify(diff(U_F[1], x)), simplify(diff(U_F[1], y))]])
Grad_U_S = Matrix([[simplify(diff(U_S[0], X)), simplify(diff(U_S[0], Y))],
                   [simplify(diff(U_S[1], X)), simplify(diff(U_S[1], Y))]])
Grad_U_M = Matrix([[simplify(diff(U_M[0], X)), simplify(diff(U_M[0], Y))],
                   [simplify(diff(U_M[1], X)), simplify(diff(U_M[1], Y))]])

if debug:
    underline("Gradients")
    print "grad(u_F) =\n", grad_u_F, "\n"
    print "grad(U_F) =\n", grad_U_F, "\n"
    print "Grad(U_S) =\n", Grad_U_S, "\n"
    print "Grad(U_M) =\n", Grad_U_M, "\n"
    print

# Compute Cauchy stress for fluid
I = eye(2)
sigma_F = nu*(grad_u_F + grad_u_F.T) - p_F*I

# Compute reference stress for structure
F_S = I + Grad_U_S
J_S = F_S.det()
E_S = Rational(1, 2)*(F_S.T*F_S - I)
Sigma_S = F_S*(2*mu*E_S + lmbda*E_S.trace()*I)

# Compute reference stress tensor for fluid.
F_M = I + Grad_U_M
F_M_inv = F_M.inv()
J_M = F_M.det()
U_F = u_F.subs(y, Y)
Grad_U_F = Matrix([[simplify(diff(U_F[0], X)), simplify(diff(U_F[0], Y))],
                   [simplify(diff(U_F[1], X)), simplify(diff(U_F[1], Y))]])
P_F = p_F.subs(y, Y).subs(x, X + U_M[0])
Sigma_F = nu*(Grad_U_F * F_M_inv + F_M_inv.T * Grad_U_F.T) - P_F*I
PSigma_F = J_M*Sigma_F*F_M_inv.T

# Compute boundary tractions on fsi boundary in reference frame:
G_F = J_M*Sigma_F*F_M_inv.T*N 
G_F = Matrix([simplify(G_F[0]), simplify(G_F[1])])
G_S = Sigma_S*N
G_S = Matrix([simplify(G_S[0]), simplify(G_S[1])])

# Compute top boundary stress for fluid
underline("Deriving left traction for fluid")
N_left = Matrix([-1, 0])
traction_left = sigma_F*N_left
print traction_left
print

underline("Deriving left reference traction for fluid")
TRACTION_LEFT = Sigma_F*N_left
print TRACTION_LEFT
print

# Compute boundary traction in reference frame
underline("Deriving additional reference boundary traction G_0")
G_0 = G_S - G_F
G_0 = Matrix([simplify(G_0[0]), simplify(G_0[1])])
print G_0
print

# Print the Fluid Boundary traction in the reference frame
underline("Fluid Boundary traction in the reference frame")
print
print G_F
print

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

#Derive right-hand side F_F such that the Navier Stokes equations are satisfied.
underline("Deriving right-hand side for the Navier--Stokes equations in reference domain")
div_Sigma_F = Matrix([diff(PSigma_F[0, 0], x) + diff(PSigma_F[0, 1], y),
                      diff(PSigma_F[1, 0], x) + diff(PSigma_F[1, 1], y)])
dot_U_F = Matrix([diff(U_F[0], t), diff(U_F[1], t)])
grad_U_F_U = grad_U_F*U_F
F_F = dot_U_F + grad_U_F_U - div_Sigma_F
F_F = Matrix([simplify(F_F[0]), simplify(F_F[1])])
print F_F
print


if debug:
    underline("Divergence of fluid stress tensor")
    print "div sigma_F =", div_sigma_F
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
f_M = Matrix([simplify(f_M[0]), simplify(f_M[1])])
print f_M
print

if debug:

    # Check continuity of mesh
    underline("Checking continuity of mesh: U_S - U_M")
    print U_S.subs(X, X_) - U_M.subs(X, X_)
    print

    # Check continuity of velocity
    underline("Checking continuity of velocity: u_F - p_S")
    p_S = U_S.diff(t).subs(X, X_).subs(Y, y)
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
    
    underline("Checking that U_F is divergence free")
    div_U_F = diff(U_F[0], x) + diff(U_F[1], y)
    print div_U_F
    print
    

    underline("Average values of pressures in reference and current frame")
    average_P_F = integrate(integrate(P_F, (X, 0, 1)), (Y, 0, 1))
    average_p_F = integrate(integrate(p_F, (x, x_, 1)), (y, 0, 1))
    print average_P_F
    print average_p_F
    print

    underline("Goal functional")
    T = symbols("T")
    goal_functional = integrate(integrate(integrate(U_S[0], (X, 1, 2)), (Y, 0, 1)), (t, 0, T))
    print goal_functional
