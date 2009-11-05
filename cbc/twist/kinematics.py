__author__ = "Harish Narayanan"
__copyright__ = "Copyright (C) 2009"
__license__  = "GNU GPL Version 3 or any later version"

from dolfin import *

# Fix for transposed gradient in UFL
# This should be removed
from ufl import grad as ufl_grad
def Grad(v):
    if v.rank() == 1:
        return ufl_grad(v).T
    else:
        return ufl_grad(v)

# Infinitesimal strain tensor
def InfinitesimalStrain(u):
    return variable(0.5*(Grad(u) + Grad(u).T))

# Second order identity tensor
def SecondOrderIdentity(u):
    return variable(Identity(u.cell().d))

# Deformation gradient
def DeformationGradient(u):
    I = SecondOrderIdentity(u)
    return variable(I + Grad(u))

# Determinant of the deformation gradient
def Jacobian(u):
    F = DeformationGradient(u)
    return variable(det(F))

# Right Cauchy-Green tensor
def RightCauchyGreen(u):
    F = DeformationGradient(u)
    return variable(F.T*F)

# Green-Lagrange strain tensor
def GreenLagrangeStrain(u):
    I = SecondOrderIdentity(u)
    C = RightCauchyGreen(u)
    return variable(0.5*(C - I))

# Left Cauchy-Green tensor
def LeftCauchyGreen(u):
    F = DeformationGradient(u)
    return variable(F*F.T)

# Euler-Almansi strain tensor
def EulerAlmansiStrain(u):
    I = SecondOrderIdentity(u)
    b = LeftCauchyGreen(u)
    return variable(0.5*(I - inv(b)))

# Invariants of an arbitrary tensor, A
def Invariants(A):
    I1 = tr(A)
    I2 = 0.5*(tr(A)**2 - tr(A*A))
    I3 = det(A)
    return [I1, I2, I3]

# Invariants of the (right/left) Cauchy-Green tensor
def CauchyGreenInvariants(u):
    C = RightCauchyGreen(u)
    [I1, I2, I3] = Invariants(C)
    return [variable(I1), variable(I2), variable(I3)]

# Isochoric part of the deformation gradient
def IsochoricDeformationGradient(u):
    F = DeformationGradient(u)
    J = Jacobian(u)
    return variable(J**(-1.0/3.0)*F)

# Isochoric part of the right Cauchy-Green tensor
def IsochoricRightCauchyGreen(u):
    C = RightCauchyGreen(u)
    J = Jacobian(u)
    return variable(J**(-2.0/3.0)*C)

# Invariants of the ischoric part of the (right/left) Cauchy-Green
# tensor. Note that I3bar = 1 by definition.
def IsochoricCauchyGreenInvariants(u):
    Cbar = IsochoricRightCauchyGreen(u)
    [I1bar, I2bar, I3bar] = Invariants(Cbar)
    return [variable(I1bar), variable(I2bar)]
