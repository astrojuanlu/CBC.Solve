"This module specifies the variational forms for the dual FSI problem."

__author__ = "Kristoffer Selim and Anders Logg"
__copyright__ = "Copyright (C) 2010 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

# Last changed: 2012-05-03

from dolfin import *
from operators import *
import new_jacobianforms as jfor

def create_dual_forms(Omega_F, Omega_S, k, problem,
                      v_F,  q_F,  s_F,  v_S,  q_S,  v_M,  q_M,
                      Z_F,  Y_F,  X_F,  Z_S,  Y_S,  Z_M,  Y_M,
                      Z_F0, Y_F0, X_F0, Z_S0, Y_S0, Z_M0, Y_M0,
                      U_F0, P_F0, U_S0, P_S0, U_M0,
                      U_F1, P_F1, U_S1, P_S1, U_M1):
    """
    Return bilinear and linear forms for a time step
    method - FE is forward Euler, BE is backward Euler, CG1 is midpoint rule
    """

    # Choose method here
    method = "FE"

    info_blue("Creating dual forms")

    # Repackage the parameters and functions
    Iulist    = [v_F,  q_F,  s_F,  v_S,  q_S,  v_M,  q_M]

    #Primal function discretization is the same for all methods
    U1list    = [U_F1, P_F1, None,U_S1, P_S1, U_M1,None]
    Umidlist  = U1list
    Udotlist  = U1list
    #Here I assume Z0 is the last computed time step.
    #and U1 is the primal solution at the time at which we are trying to compute the dual
    #solution, ie the reverse of what it denoted in the primal problem.

    Zlist = [Z_F,Y_F,X_F,Z_S,Y_S,Z_M,Y_M]
    Z0list = [Z_F0, Y_F0, X_F0, Z_S0, Y_S0, Z_M0, Y_M0]
    #There is a minus here due to IBP
    dotVlist  = [(z0 - z)/k for z0,z in zip(Z0list,Zlist)]

    if method == "BE":
        Vlist     = Zlist
    elif method == "FE":
        Vlist     =  Z0list
    elif method == "CG1":
        Vlist = [(z0 + z)*0.5 for z0,z in zip(Z0list,Zlist)]
    else:
        raise Exception("Only FE,BE,and CG1 methods are possible")


    #Material Parameters Dictionary
    matparams = {"mu_F":Constant(problem.fluid_viscosity()),
                 "rho_F":Constant(problem.fluid_density()),
                 "mu_S":Constant(problem.structure_mu()),
                 "lmbda_S":Constant(problem.structure_lmbda()),
                 "rho_S":Constant(problem.structure_density()),
                 "mu_M":Constant(problem.mesh_mu()),
                 "lmbda_M":Constant(problem.mesh_lmbda()) }

    #Normals Dictionary
    normals = {"N_F":FacetNormal(Omega_F), \
               "N_S":FacetNormal(Omega_S)}

    #Measures dictionary
    measures = {"dxF":dx(0),\
                "dxM":dx(0),\
                "dxS":dx(1),\
                "dsF":ds(0),\
                "dsS":None,\
                "dFSI":dS(2)}

    #Forces dictionary
    try:
        g_F = problem.fluid_boundary_force()
    except:
        g_F = None

    forces = {"F_F":None,
              "F_S":None,
              "F_M":None,
              "G_S":None,
              "g_F":g_F}

    #Todo
    #at the moment the other schemes are derived from
    #the perspective of the CG1 scheme. An abstract form
    #should be used to derive all three schemes.

    A_system = jfor.fsi_jacobian(Iulist = Iulist,
                                 Iudotlist = Iulist,
                                 Iumidlist = Iulist,
                                 U1list = U1list,
                                 Umidlist = Umidlist,
                                 Udotlist = Udotlist,
                                 Vlist = Vlist,
                                 dotVlist = dotVlist,

                                 matparams = matparams,
                                 measures = measures,
                                 forces = forces,
                                 normals = normals)

                              # Define goal funtional
    goal_functional = problem.evaluate_functional(v_F, q_F, v_S, q_S, v_M,
                                                  measures["dxF"], measures["dxS"],
                                                  measures["dxM"])

    # Define the dual rhs and lhs
    A = lhs(A_system)
    L = rhs(A_system) + goal_functional

    info_blue("Dual forms created")
    return A, L

def create_dual_bcs(problem, W):
    "Create boundary conditions for dual problem"

    bcs = []

    # Boundary conditions for dual velocity
    for boundary in problem.fluid_velocity_dirichlet_boundaries():
        bcs += [DirichletBC(W.sub(0), (0, 0), boundary)]
    bcs += [DirichletBC(W.sub(0), (0, 0), problem.fsi_boundary, 1)]

    # Boundary conditions for dual pressure
    for boundary in problem.fluid_pressure_dirichlet_boundaries():
        bcs += [DirichletBC(W.sub(1), 0, boundary)]

    # Boundary conditions for dual structure displacement and velocity
    for boundary in problem.structure_dirichlet_boundaries():
        bcs += [DirichletBC(W.sub(3), (0, 0), boundary)]
        bcs += [DirichletBC(W.sub(4), (0, 0), boundary)]

    # Boundary conditions for dual mesh displacement
    try:
        for bound in problem.mesh_dirichlet_boundaries():
            print bound
            bcs += [DirichletBC(W.sub(5), (0, 0),bound)]
            print "GB New Boundary found"
            adoih()
    except:
        bcs += [DirichletBC(W.sub(5), (0, 0), DomainBoundary())]
    return bcs
