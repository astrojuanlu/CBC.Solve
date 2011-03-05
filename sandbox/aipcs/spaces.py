"This module creates function spaces and functions."

__author__ = "Kristoffer Selim and Anders Logg"
__copyright__ = "Copyright (C) 2010 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

from dolfin import *

def create_primal_functions(Omega, parameters):
    "Return primal variables on the full domain initialized to zero"

    # Create function spaces
    V_F = VectorFunctionSpace(Omega, "CG", 2)
    Q_F = FunctionSpace(Omega, "CG", 1)

    # Create primal functions
    U_F = Function(V_F)
    P_F = Function(Q_F)

    return U_F, P_F

def create_dual_space(Omega, parameters):
    "Return dual function space on the full domain"

    # Create function spaces
    V_F = VectorFunctionSpace(Omega, "CG", 2)
    Q_F = FunctionSpace(Omega, "CG", 1)

    # Create mixed function space
    W = MixedFunctionSpace([V_F, Q_F])

    return W

def create_dual_functions(Omega, parameters):
    "Return dual variables on the full domain initialized to zero"
    W = create_dual_space(Omega, parameters)
    Z = Function(W)
    return Z, Z.split(False)
