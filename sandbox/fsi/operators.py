" Defines operators and constants needed in dual.py and residual.py"
from dolfin import *

# Define constants 
rho_F = 1.0
mu_F = 0.002
rho_S = 15.0
mu_S =  75.0
lmbda_S =  125.0
mu_M =  3.8461
lmbda_M =  5.76
alpha = 1.0

# Identity matrix
I = variable(Identity(2))

# Deformation gradient
def F(u):
    F = (I + grad(u))
    return F

# Transpose of deformation gradient
def F_T(u):
    F_T  = ((I + grad(u))).T
    return F_T

# Inverse deformation gradient
def F_inv(u):
    F_inv  = inv((I + grad(u)))
    return F_inv

# Transpose of inverse deformation gradient
def F_invT(u):
    F_invT  = (inv((I + grad(u)))).T
    return F_invT

# Determinant of the deformation gradient
def J(u):
    J = det(I + grad(u))
    return J

# DJ(u,w) is J(u) linearized around w (w = test function)
# (u is always U_M)
def DJ(u,w):
    DJ = w[0].dx(0)*(1 - u[1].dx(1)) - w[0].dx(1)*u[1].dx(0) \
        -w[1].dx(0)*u[0].dx(1) + w[1].dx(1)*(1 + u[0].dx(0))
    return DJ

# Green - Lagrange tensor
def E(u):
    return 0.5*(F_T(u)*F(u) - I)

# Sym gradient FIXME: This should be removed ones the UFL bug for rhs/lhs is fixed
def sym_gradient(u):
    sym_gradient = 0.5*(grad(u)+ grad(u).T)
    return sym_gradient

# Fluid stress in the reference domain
def Sigma_F(u,p,v):
    return  mu_F*(grad(u)*F_inv(v) + F_invT(v)*grad(u).T) - p*I

# Structure stress in the reference domain
def Sigma_S(u):
    return F(u)*(2*mu_S*E(u) + lmbda_S*tr(E(u))*I)

# Mesh stress in the reference domain
def Sigma_M(u):
    return 2.0*mu_M*sym_gradient(u) + lmbda_M*tr(sym_gradient(u))*I

# Define Quasi-Euler time derivative for the fluid
def D_t(U_F, U_F0, kn, U_M, rho_F):
    return rho_F*J(U_M)*((1/kn)*(U_F - U_F0) + dot(dot(grad(U_F), F_inv(U_M)), U_F))

