"This module creates function spaces and functions."

__author__ = "Kristoffer Selim and Anders Logg"
__copyright__ = "Copyright (C) 2010 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

from dolfin import *

def create_primal_functions(Omega, parameters):
    "Return primal variables on the full domain initialized to zero"

    # Create function spaces
    structure_element_degree = parameters["structure_element_degree"]
    V_F = VectorFunctionSpace(Omega, "CG", 2)
    Q_F = FunctionSpace(Omega, "CG", 1)
    V_S = VectorFunctionSpace(Omega, "CG", structure_element_degree)
    Q_S = VectorFunctionSpace(Omega, "CG", structure_element_degree)
    V_M = VectorFunctionSpace(Omega, "CG", 1)
    Q_M = VectorFunctionSpace(Omega, "CG", 1)

    # Create primal functions
    U_F = Function(V_F)
    P_F = Function(Q_F)
    U_S = Function(V_S)
    P_S = Function(Q_S)
    U_M = Function(V_M)

    return U_F, P_F, U_S, P_S, U_M

def create_dual_space(Omega, parameters):
    "Return dual function space on the full domain"

    # Create function spaces
    structure_element_degree = parameters["structure_element_degree"]
    V_F = VectorFunctionSpace(Omega, "CG", 2)
    Q_F = FunctionSpace(Omega, "CG", 1)
    S_F = VectorFunctionSpace(Omega, "CG", 1)
    V_S = VectorFunctionSpace(Omega, "CG", structure_element_degree)
    Q_S = VectorFunctionSpace(Omega, "CG", structure_element_degree)
    V_M = VectorFunctionSpace(Omega, "CG", 1)
    Q_M = VectorFunctionSpace(Omega, "CG", 1)

    # Create mixed function space
    W = MixedFunctionSpace([V_F, Q_F, S_F, V_S, Q_S, V_M, Q_M])

    # Print some info
    offset = 0
    info("Created dual spaces: dim = %d" % W.dim())
    info("  num_vertices = %d" % Omega.num_vertices())
    info("  num_edges    = %d" % Omega.num_edges())
    info("  V_F: dim = %d offset = %d" % (V_F.dim(), offset)); offset += V_F.dim()
    info("  Q_F: dim = %d offset = %d" % (Q_F.dim(), offset)); offset += Q_F.dim()
    info("  S_F: dim = %d offset = %d" % (S_F.dim(), offset)); offset += S_F.dim()
    info("  V_S: dim = %d offset = %d" % (V_S.dim(), offset)); offset += V_S.dim()
    info("  Q_S: dim = %d offset = %d" % (Q_S.dim(), offset)); offset += Q_S.dim()
    info("  V_M: dim = %d offset = %d" % (V_M.dim(), offset)); offset += V_M.dim()
    info("  Q_M: dim = %d offset = %d" % (Q_M.dim(), offset)); offset += Q_M.dim()

    return W

def create_dual_functions(Omega, parameters):
    "Return dual variables on the full domain initialized to zero"
    W = create_dual_space(Omega, parameters)
    Z = Function(W)
    return Z, Z.split(False)
