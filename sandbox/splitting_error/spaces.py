"This module creates function spaces and functions."

__author__ = "Kristoffer Selim and Anders Logg"
__copyright__ = "Copyright (C) 2010 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

from dolfin import *

def create_primal_functions(Omega):
    "Return primal velocity"

    # Create function spaces
    V = VectorFunctionSpace(Omega, "CG", 2)
    Q = FunctionSpace(Omega, "CG", 1)

    # Create primal functions
    uh = Function(V)
    ph = Function(Q)

    return uh, ph

def create_dual_space(Omega):
    "Return dual function space"

    # Create function spaces
    V = VectorFunctionSpace(Omega, "CG", 2)
    Q = FunctionSpace(Omega, "CG", 1)

    # Create mixed function space
    W = MixedFunctionSpace([V, Q])
    
    return W

def create_dual_functions(Omega):
    "Return dual variables on the full domain initialized to zero"

    W = create_dual_space(Omega)
    dual_sol = Function(W)
    return dual_sol, dual_sol.split()
