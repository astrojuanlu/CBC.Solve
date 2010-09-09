"This module defines special operators for the dual problem and residuals."

__author__ = "Kristoffer Selim and Anders Logg"
__copyright__ = "Copyright (C) 2010 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

# Last changed: 2010-09-09

from dolfin import *

# Define identity matrix (2D)
I = Identity(2)

def F(v):
    "Return deformation gradient"
    return I + grad(v)

def J(v):
    "Return determinant of deformation gradient"
    return det(F(v))

def Sigma_F(U_F, P_F, U_M, mu_F):
    "Return fluid stress in reference domain"
    return mu_F*(grad(U_F)*inv(F(U_M)) + inv(F(U_M)).T*grad(U_F).T) - P_F*I

def Sigma_S(U_S, mu_S, lmbda_S):
    "Return structure stress in reference domain"
    E_S = 0.5*(dot(F(U_S).T, F(U_S) - I)
    return dot(F(U_S), 2*mu_S*E_S + lmbda_S*tr(E_S)*I)

def Sigma_M(U_M, mu_M, lmbda_M):
    "Return mesh stress in reference domain"
    return 2*mu_M*sym(grad(U_M)) + lmbda_M*tr(sym(grad(U_M)))*I

def Dt(U_F0, U_F1, U_M0, U_M1, rho_F, kn):
    "Return time derivative for fluid problem in reference domain"

    # Midpoint values
    U_F = 0.5 * (U_F0 + U_F1)
    U_M = 0.5 * (U_M0 + U_M1)

    # Time derivatives
    dU_F = (1/kn) * (U_F1 - U_F0)
    dU_M = (1/kn) * (U_M1 - U_M0)

    return rho_F * J(U_M) * (dU_F + dot(grad(U_F), dot(inv(F(U_M)), U_F - dU_M)))
