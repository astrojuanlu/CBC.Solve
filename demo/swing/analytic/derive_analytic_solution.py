"This script checks that the analytical solution satisfies the FSI problem"

__author__ = "Anders Logg"
__copyright__ = "Copyright (C) 2012 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

# Last changed: 2012-03-09

from sympy import *

def underline(s): print s + "\n" + "-"*len(s)

# Declare symbols
x, y, X, Y, t = symbols("x y X Y t")
C = symbols("C")

# Material parameters
mu = 1
lmbda = 2
nu = 1
rho_S = 100

# Simplification not handled by SymPy
sin2pit = 2*sin(pi*t)*cos(pi*t)
cos2pit = cos(pi*t)**2 - sin(pi*t)**2
cos2piY = cos(pi*Y)**2 - sin(pi*Y)**2

# Analytical solutions
u_F = Matrix([0, C*pi*x*(1 - x)*sin2pit])
p_F = -2*C**2*(1 - 2*x)**2*sin(pi*t)**3*(sin(pi*t) + pi*cos(pi*t))
P_F = -2*C**2*(1 - 2*X)**2*sin(pi*t)**3*(sin(pi*t) + pi*cos(pi*t))
U_S = Matrix([0, C*X*(1 - X)*sin(pi*Y)*sin(pi*t)**2])
U_M = Matrix([0, C*X*(1 - X)*sin(pi*Y)*sin(pi*t)**2])

# Right-hand side for fluid problem
f_F = Matrix([8*C**2*(1 - 2*x)*sin(pi*t)**3*(sin(pi*t) + pi*cos(pi*t)),
              2*C*pi**2*x*(1 - x)*cos2pit + 2*C*pi*sin2pit])

# Right-hand side for structure problem
f_S = Matrix([C*sin(pi*t)**2*(3*pi*cos(pi*Y)*(2*X - 1) \
              + C*sin(pi*t)**2*( \
              sin(pi*Y)**2*(2*pi**2*X**3 - 3*pi**2*X**2 - (16 - pi**2)*X + 8) \
              - 3*pi**2*X*cos(pi*Y)**2*(2*X**2 - 3*X + 1))), \
              2*C*sin(pi*t)**2*sin(pi*Y) - C*pi*sin(pi*t)**2*cos(pi*Y) \
              - C**2*pi*sin(pi*t)**4*cos(pi*Y)*sin(pi*Y) \
              + 2*pi*C*X*sin(pi*t)**2*cos(pi*Y)
              - 200*C*pi**2*(X**2 - X)*(cos(pi*t)**2 - sin(pi*t)**2)*sin(pi*Y) \
              - 6*C**2*pi*(X**2 - X)*sin(pi*t)**4*cos(pi*Y)*sin(pi*Y) \
              + C**2*pi**2*(3*X**2 - 2*X**3 - X)*sin(pi*t)**4*(cos(pi*Y)**2 - sin(pi*Y)**2)
])

# Right-hand side for structure problem
f_M = Matrix([3*C*pi*cos(pi*Y)*sin(pi*t)**2*(2*X - 1),
              C*sin(pi*t)*( \
              2*sin(pi*Y)*sin(pi*t) \
              + pi*cos(pi*Y)*sin(pi*t)*(2*X - 1) \
              - 2*X*pi*sin(pi*Y)*cos(pi*t)*(X - 1))])

# Additional boundary traction for structure
g_0 = Matrix([C*(1 - 2*x)*sin(pi*t)*((1 - p_F)*sin(pi*t) - 2*pi*cos(pi*t)) \
              / sqrt(1 + C**2*(1 - 2*x)**2*sin(pi*t)**4), 0])
G_0 = Matrix([C*(1 - 2*X)*sin(pi*t)*((1 - P_F)*sin(pi*t) - 2*pi*cos(pi*t)) \
              / (1 + C**2*(1 - 2*X)**2*sin(pi*t)**4), 0])

# C++ friendly versions of analytical formulas

def pow(expression, power):
    return expression**power

def eval_f_F():
    A = Integer(1)
    B = Integer(2)
    D = Integer(4)
    E = Integer(8)
    a = sin(pi*t)
    b = cos(pi*t)
    fx = E*pow(C, 2)*(A - B*x)*pow(a, 3)*(a + pi*b)
    fy = B*pow(pi, 2)*C*x*(A - x)*(pow(b, 2) - pow(a, 2)) + D*pi*C*a*b
    return Matrix([fx, fy])

def eval_f_S():
    A = Integer(1)
    B = Integer(2)
    D = Integer(3)
    E = Integer(6)
    F = Integer(8)
    G = Integer(16)
    H = Integer(200)
    a = sin(pi*t)
    b = cos(pi*t)
    c = sin(pi*Y)
    d = cos(pi*Y)
    e = pow(a, 2)
    f = pow(pi, 2)
    g = pow(X, 2)
    h = pow(c, 2)
    i = pow(d, 2)
    j = pow(b, 2)
    k = pow(a, 4)
    l = pow(C, 2)
    m = pow(X, 3)
    fx = C*e*(D*pi*d*(B*X - A) + C*e*(h*(B*f*X*g - D*f*g \
         - (G - f)*X + F) - D*f*X*i*(B*g - D*X + A)))
    fy = B*C*e*c - C*pi*e*d - l*pi*k*d*c + B*pi*C*X*e*d - H*C*f*(g - X)*(j - e)*c \
         - E*l*pi*(g - X)*k*d*c + l*f*(D*g - B*m - X)*k*(i - h)
    return Matrix([fx, fy])

def eval_f_M():
    A = Integer(1)
    B = Integer(2)
    D = Integer(3)
    a = sin(pi*t)
    b = cos(pi*t)
    c = sin(pi*Y)
    d = cos(pi*Y)
    fx = D*C*pi*d*pow(a, 2)*(B*X - A)
    fy = C*a*(B*c*a + pi*d*a*(B*X - A) - B*X*pi*c*b*(X - A))
    return Matrix([fx, fy])

def eval_g_0():
    A = Integer(1)
    B = Integer(2)
    a = sin(pi*t)
    b = cos(pi*t)
    p = -B*pow(C, 2)*pow(A - B*x, 2)*pow(a, 3)*(a + pi*b)
    gx = C*(A - B*x)*a*((A - p)*a - B*pi*b) / sqrt(A + pow(C, 2)*pow(A - B*x, 2)*pow(a, 4))
    gy = 0
    return Matrix([gx, gy])

def eval_G_0():
    A = Integer(1)
    B = Integer(2)
    a = sin(pi*t)
    b = cos(pi*t)
    p = -B*pow(C, 2)*pow(A - B*X, 2)*pow(a, 3)*(a + pi*b)
    Gx = C*(A - B*X)*a*((A - p)*a - B*pi*b) / (A + pow(C, 2)*pow(A - B*X, 2)*pow(a, 4))
    Gy = 0
    return Matrix([Gx, Gy])

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
print "G_0 =\n", g_0, "\n"
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
ddot_U_S = Matrix([diff(diff(U_S[0], t), t), diff(diff(U_S[1], t), t)])
r = rho_S*ddot_U_S - Div_Sigma_S - f_S
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

# Check that C++ style right-hand sides evaluate correctly
underline("Checking evaluation of C++ style right-hand sides")
print eval_f_F() - f_F
print eval_f_S() - f_S
print eval_f_M() - f_M
print eval_g_0() - g_0
print eval_G_0() - G_0
print

# Compute reference value of functional: integrated Y-displacement
T = Rational(1, 2)
M = integrate(integrate(integrate(U_S[1], (X, 0, 1)), (Y, 0, Y_)), (t, 0, T))
underline("Value of reference functional")
print "M =", M
