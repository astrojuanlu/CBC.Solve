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

# Simplification not handled by SymPy
cos2pit = cos(pi*t)**2 - sin(pi*t)**2
cos2piY = cos(pi*Y)**2 - sin(pi*Y)**2

# Analytical solutions
u_F = Matrix([0, 2*pi*C*x*(1 - x)*sin(pi*t)*cos(pi*t)])
p_F = -2*C**2*(1 - 2*x)**2*sin(pi*t)**3*(sin(pi*t) + pi*cos(pi*t))
U_S = Matrix([0, C*X*(1 - X)*sin(pi*Y)*sin(pi*t)**2])
U_M = Matrix([0, C*X*(1 - X)*sin(pi*Y)*sin(pi*t)**2])

# Right-hand side for fluid problem
f_F = Matrix([8*C**2*(1 - 2*x)*sin(pi*t)**3*(sin(pi*t) + pi*cos(pi*t)),
              2*pi**2*C*x*(1 - x)*cos2pit \
                  + 4*pi*C*sin(pi*t)*cos(pi*t)])

# Right-hand side for structure problem
f_S = Matrix([C*sin(pi*t)**2*(3*pi*cos(pi*Y)*(2*X - 1) \
              + C*sin(pi*t)**2*( \
              sin(pi*Y)**2*(2*pi**2*X**3 - 3*pi**2*X**2 - (16 - pi**2)*X + 8) \
              - 3*pi**2*X*cos(pi*Y)**2*(2*X**2 - 3*X + 1))),
              C*sin(pi*t)**2*(2*sin(pi*Y) + pi*(2*X - 1)*cos(pi*Y) \
              - C*pi*sin(pi*t)**2*( \
              cos(pi*Y)*sin(pi*Y)*(6*X**2 - 6*X + 1) \
              + pi*X*cos2piY*(2*X**2 - 3*X + 1)))])

# Right-hand side for structure problem
f_M = Matrix([3*C*pi*cos(pi*Y)*sin(pi*t)**2*(2*X - 1),
              C*sin(pi*t)*( \
              2*sin(pi*Y)*sin(pi*t) \
              + pi*cos(pi*Y)*sin(pi*t)*(2*X - 1) \
              - 2*X*pi*sin(pi*Y)*cos(pi*t)*(X - 1))])

# Additional boundary traction for structure
g_0 = Matrix([C*(1 - 2*x)*sin(pi*t)*((1 - p_F)*sin(pi*t) - 2*pi*cos(pi*t)) \
              / sqrt(1 + C**2*(1 - 2*x)**2*sin(pi*t)**4), 0])

# Print solutions
underline("Analytical solutions")
print "u_F =\n", u_F, "\n"
print "p_F =\n", p_F, "\n"
print "U_S =\n", U_S, "\n"
print "U_M =\n", U_M, "\n"
print

# Print right-hand sides
underline("Right-hand sides")
print "f_F =\n", f_F, "\n"
print "f_S =\n", f_S, "\n"
print "f_M =\n", f_M, "\n"
print "g_0 =\n", g_0, "\n"
print

# Location of FSI boundary
Y_ = Rational(1, 2)
y_ = Y_ + U_S[1].subs(Y, Y_).subs(X, x)

# Print location of FSI boundary
underline("Location of FSI boundary")
print "Y_ =", Y_
print "y_ =", y_
print

# Normal direction at FSI boundary
n = Matrix([-simplify(diff(U_S[1], X)), 1]).subs(Y, Y_).subs(X, x)
n = n / sqrt(n[0]**2 + n[1]**2)

# Compute gradients
grad_u_F = Matrix([[simplify(diff(u_F[0], x)), simplify(diff(u_F[0], y))],
                   [simplify(diff(u_F[1], x)), simplify(diff(u_F[1], y))]])
Grad_U_S = Matrix([[simplify(diff(U_S[0], X)), simplify(diff(U_S[0], Y))],
                   [simplify(diff(U_S[1], X)), simplify(diff(U_S[1], Y))]])
Grad_U_M = Matrix([[simplify(diff(U_M[0], X)), simplify(diff(U_M[0], Y))],
                   [simplify(diff(U_M[1], X)), simplify(diff(U_M[1], Y))]])

# Symbolic gradients used for debugging
#fx, gx, gy = symbols("fx gx gy")
#grad_u_F = Matrix([[0, 0], [fx, 0]])
#Grad_U_S = Matrix([[0, 0], [gx, gy]])
#n = Matrix([-gx, 1]) / sqrt(1 + gx**2)
#p_F = -gx*(fx + 2*gx)
#g_0 = Matrix([(gx - fx + fx*gx**2 + 2*gx**3) / sqrt(1 + gx**2), 0])

# Print gradients
underline("Gradients")
print "grad(u_F) =\n", grad_u_F, "\n"
print "Grad(U_S) =\n", Grad_U_S, "\n"
print "Grad(U_M) =\n", Grad_U_M, "\n"
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
r = Matrix([simplify(r[0]), simplify(r[1])])
print r
print

# Check that the Navier-Stokes equations are satisfied
underline("Checking that the Navier-Stokes equations are satisfied")
div_sigma_F = Matrix([diff(sigma_F[0], x) + diff(sigma_F[1], y),
                      diff(sigma_F[1], x) + diff(sigma_F[1], y)])
dot_u_F = Matrix([diff(u_F[0], t), diff(u_F[1], t)])
grad_u_F_u = grad_u_F*u_F
r = dot_u_F + grad_u_F_u - div_sigma_F - f_F
r = Matrix([simplify(r[0]), simplify(r[1])])
print r
div_u_F = diff(u_F[0], x) + diff(u_F[1], y)
print div_u_F
print


# Check that the hyperelastic equation is satisfied
underline("Checking that hyperelastic equation is satisfied")
Div_Sigma_S = Matrix([diff(Sigma_S[0], X) + diff(Sigma_S[1], Y),
                      diff(Sigma_S[1], X) + diff(Sigma_S[1], Y)])
ddot_U_S = Matrix([diff(diff(U_S[0], t), t), diff(diff(U_S[0], t), t)])
r = ddot_U_S - Div_Sigma_S - f_S
r = Matrix([simplify(r[0]), simplify(r[1])])
print r
print

# Check that the mesh equation is satisfied
underline("Checking that mesh equation is satisfied")
Sigma_M = mu*(Grad_U_M + Grad_U_M.T) + lmbda*Grad_U_M.trace()*I
Div_Sigma_M = Matrix([diff(Sigma_M[0], X) + diff(Sigma_M[1], Y),
                      diff(Sigma_M[1], X) + diff(Sigma_M[1], Y)])
dot_U_M = Matrix([diff(U_M[0], t), diff(U_M[1], t)])
r = dot_U_M - Div_Sigma_M - f_M
r = Matrix([simplify(r[0]), simplify(r[1])])
print r
print

# Compute reference value of functional: integrated Y-displacement
T = Rational(1, 2)
M = integrate(integrate(integrate(U_S[1], (X, 0, 1)), (Y, 0, Y_)), (t, 0, T))
underline("Value of reference functional")
print "M =", M
