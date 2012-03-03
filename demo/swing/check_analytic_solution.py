"This script checks that the analytical solution satisfies the FSI problem"

__author__ = "Anders Logg"
__copyright__ = "Copyright (C) 2012 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

# Last changed: 2012-03-03

from sympy import *

def underline(s): print s + "\n" + "-"*len(s)

# Declare symbols
x, y, X, Y, t = symbols("x y X Y t")
C = symbols("C")

# Material parameters
mu = 1
lmbda = 2
nu = 1

# Analytic solutions
u_F = Matrix((0, 2*pi*C*x*(1 - x)*sin(pi*t)*cos(pi*t)))
p_F = -2*C**2*(1 - 2*x)**2*sin(pi*t)**3*(sin(pi*t) + pi*cos(pi*t))
U_S = Matrix((0, C*X*(1 - X)*sin(pi*Y)*sin(pi*t)**2))
U_M = Matrix((0, C*X*(1 - X)*sin(pi*Y)*sin(pi*t)**2))

# Additional boundary traction for structure
g_0 = Matrix((C*(1 - 2*x)*sin(pi*t)*((1 - p_F)*sin(pi*t) - 2*pi*cos(pi*t)) \
              / sqrt(1 + C**2*(1 - 2*x)**2*sin(pi*t)**4), 0))

# Print solutions
underline("Analytical solutions")
print "u_F =\n", u_F, "\n"
print "p_F =\n", p_F, "\n"
print "U_S =\n", U_S, "\n"
print "U_M =\n", U_M, "\n"
print "g_0 =\n", g_0, "\n"

# Location of FSI boundary
Y_ = Rational(1, 2)
y_ = Y_ + U_S[1].subs(Y, Y_).subs(X, x)

# Print location of FSI boundary
underline("Location of FSI boundary")
print "Y_ =", Y_
print "y_ =", y_
print

# Normal direction at FSI boundary
n = Matrix((-simplify(diff(U_S[1], X)), 1)).subs(Y, Y_).subs(X, x)
n = n / sqrt(n[0]**2 + n[1]**2)

# Compute gradients
grad_u_F = Matrix([[simplify(diff(u_F[0], x)), simplify(diff(u_F[0], y))],
                   [simplify(diff(u_F[1], x)), simplify(diff(u_F[1], y))]])
Grad_U_S = Matrix([[simplify(diff(U_S[0], X)), simplify(diff(U_S[0], Y))],
                   [simplify(diff(U_S[1], X)), simplify(diff(U_S[1], Y))]])

# Symbolic gradients used for debugging
#fx, gx = symbols("fx gx")
#grad_u_F = Matrix(((0, 0), (fx, 0)))
#Grad_U_S = Matrix(((0, 0), (gx, 0)))
#n = Matrix((-gx, 1)) / sqrt(1 + gx**2)
#p_F = -gx*(fx + 2*gx)
#g_0 = Matrix(((gx - fx + fx*gx**2 + 2*gx**3) / sqrt(1 + gx**2), 0))

# Print gradients
underline("Gradients")
print "grad(u_F) =\n", grad_u_F, "\n"
print "Grad(U_S) =\n", Grad_U_S, "\n"
print

# Compute Cauchy stress for fluid
I = eye(2)
sigma_F = nu*(grad_u_F + grad_u_F.T) - p_F*I

# Compute Cauchy stress for structure
F_S = I + Grad_U_S
J_S = F_S.det()
E_S = Rational(1, 2)*(F_S.T*F_S - I)
Sigma_S = F_S*(2*mu*E_S + lmbda*E_S.trace()*I)
sigma_S = Sigma_S*F_S.T / J_S

# Compute boundary traction
g_F = sigma_F*n
g_S = sigma_S*n

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
underline("Checking continuity of boundary traction: g_S - g_F - g_0")
r = g_S - g_F - g_0
r = r.subs(Y, Y_).subs(y, y_).subs(X, x)
r = Matrix((simplify(r[0]), simplify(r[1])))
print r
