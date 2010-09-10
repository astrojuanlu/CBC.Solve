"This module implements functionality for adaptivity."

__author__ = "Kristoffer Selim and Anders Logg"
__copyright__ = "Copyright (C) 2010 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

# Last changed: 2010-09-10

from dolfin import info

from operators import Sigma_F as _Sigma_F
from operators import Sigma_S as _Sigma_S
from operators import Sigma_M as _Sigma_M
from operators import F, J
from storage import *
from utils import *

def estimate_error(problem):
    "Estimate error and compute error indicators"

    # Compute error indicators
    eta_K = compute_error_indicators_h(problem)

    # FIXME: Need to include E_k and E_c

    # Compute error estimate
    E_h = sum(eta_K)
    E_k = 0.0
    E_c = 0.0
    E = E_h + E_k + E_c

    return E, eta_K

def refine_mesh(mesh, indicators):
    "Refine mesh based on error indicators"

    # Set cell markers using Dorfler marking
    fraction = 0.5
    indices = list(argsort(indicators))
    indices.reverse()
    sub_sum = 0.0
    total_sum = sum(indicators)
    for i in indices:
        sub_sum += indicators[i]
        markers[i] = True
        if sub_sum >= fraction*total_sum:
            break

    # Refine mesh
    mesh = refine(mesh, markers)

    plot(mesh, "Refined mesh")

    return mesh

def compute_error_indicators_h(problem):
    "Compute error indicators for space discretization error E_h"

    # Get problem parameters
    Omega   = problem.mesh()
    Omega_F = problem.fluid_mesh()
    Omega_S = problem.structure_mesh()
    rho_F   = problem.fluid_density()
    mu_F    = problem.fluid_viscosity()
    rho_S   = problem.structure_density()
    mu_S    = problem.structure_mu()
    lmbda_S = problem.structure_lmbda()
    alpha_M = problem.mesh_alpha()
    mu_M    = problem.mesh_mu()
    lmbda_M = problem.mesh_lmbda()

    # Define projection space (piecewise constants)
    W = FunctionSpace(Omega, "DG", 0)
    w = TestFunction(W)

    # Initialize primal functions
    U_F0, P_F0, U_S0, P_S0, U_M0 = init_primal_data(Omega)
    U_F1, P_F1, U_S1, P_S1, U_M1 = init_primal_data(Omega)

    # Initialize dual functions
    Z, (Z_F, Y_F, Z_S, Y_S, Z_M, Y_M) = init_dual_data(Omega)

    # Define function spaces for extrapolation
    V2 = VectorFunctionSpace(Omega, "CG", 2)
    V3 = VectorFunctionSpace(Omega, "CG", 3)
    Q2 = FunctionSpace(Omega, "CG", 2)

    # Define functions for extrapolation
    ZZ_F = Function(V3)
    YY_F = Function(Q2)
    ZZ_S = Function(V2)
    YY_S = Function(V2)
    ZZ_M = Function(V2)
    YY_M = Function(V2)

    # Define time step (value set in each time step)
    kn = Constant(0.0)

    # Define normals
    N = FacetNormal(Omega)
    N_F = N
    N_S = N

    # Define midpoint values
    U_F = 0.5 * (U_F0 + U_F1)
    P_F = 0.5 * (P_F0 + P_F1)
    U_S = 0.5 * (U_S0 + U_S1)
    P_S = 0.5 * (P_S0 + P_S1)
    U_M = 0.5 * (U_M0 + U_M1)

    # Define time derivatives
    dt_U_F = (1/kn) * (U_F1 - U_F0)
    dt_U_M = (1/kn) * (U_M1 - U_M0)
    Dt_U_F = rho_F * J(U_M) * (dt_U_F + dot(grad(U_F), dot(inv(F(U_M)), U_F - dt_U_M)))
    Dt_U_S = (1/kn) * (U_S1 - U_S0)
    Dt_P_S = rho_S * (1/kn) * (P_S1 - P_S0)
    Dt_U_M = alpha_M * (1/kn) * (U_M1 - U_M0)

    # Define stresses
    Sigma_F = J(U_M)*dot(_Sigma_F(U_F, P_F, U_M, mu_F), inv(F(U_M)).T)
    Sigma_S = _Sigma_S(U_S, mu_S, lmbda_S)
    Sigma_M = _Sigma_M(U_M, mu_M, lmbda_M)

    # Fluid residual contributions
    e_F1 = w*inner(ZZ_F - Z_F, Dt_U_F - div(Sigma_F))*dx
    e_F2 = avg(w)*inner(ZZ_F('+') - Z_F('+'), jump(dot(Sigma_F, N_F)))*dS
    e_F3 = w*inner(YY_F - Y_F, div(J(U_M)*dot(inv(F(U_M)), U_F)))*dx

    # Structure residual contributions
    e_S1 = w*inner(ZZ_S - Z_S, Dt_P_S - div(Sigma_S))*dx
    e_S2 = avg(w)*inner(ZZ_S('+') - Z_S('+'), jump(dot(Sigma_S, N_S)))*dS
    e_S3 = avg(w)*inner(ZZ_S - Z_S, dot(Sigma_S - Sigma_F, N_S))('+')*dS(1)
    e_S4 = w*inner(YY_S - Y_S, Dt_U_S - P_S)*dx

    # Mesh residual contributions
    e_M1 = w*inner(ZZ_M - Z_M, Dt_U_M - div(Sigma_M))*dx
    e_M2 = avg(w)*inner(ZZ_M('+') - Z_M('+'), jump(dot(Sigma_M, N_F)))*dS
    e_M3 = avg(w)*inner(YY_M - Y_M, U_M - U_S)('+')*dS(1)

    # Collect residuals
    e_F = e_F1 + e_F2 + e_F3
    e_S = e_S1 + e_S2 + e_S3 + e_S4
    e_M = e_M1 + e_M2 + e_M3

    # Reset vectors for assembly of residuals
    eta_F = Vector(Omega.num_cells())
    eta_S = Vector(Omega.num_cells())
    eta_M = Vector(Omega.num_cells())

    # Sum residuals over time intervals
    timestep_range = read_timestep_range(problem)
    #for i in range(1, len(timestep_range)):
    # FIXME: Temporary while testing
    for i in range(len(timestep_range) / 2, len(timestep_range) / 2 + 1):

        # Get current time and time step
        t0 = timestep_range[i - 1]
        t1 = timestep_range[i]
        T  = problem.end_time()
        dt = t1 - t0
        kn.assign(dt)

        # Display progress
        info("")
        info("-"*80)
        begin("* Evaluating residuals on new time step")
        info_blue("  * t = %g (T = %g, dt = %g)" % (t0, T, dt))

        # Read primal data
        read_primal_data(U_F0, P_F0, U_S0, P_S0, U_M0, t0, Omega, Omega_F, Omega_S)
        read_primal_data(U_F1, P_F1, U_S1, P_S1, U_M1, t1, Omega, Omega_F, Omega_S)

        # Read dual data (pick value at right-hand side of interval)
        read_dual_data(Z, t1)

        # Extrapolate dual data
        #ZZ_F.extrapolate(Z_F)
        #YY_F.extrapolate(Y_F)
        #ZZ_S.extrapolate(Z_S)
        #YY_S.extrapolate(Y_S)
        #ZZ_M.extrapolate(Z_M)
        #YY_M.extrapolate(Y_M)

        # Assemble residuals
        eta_F.axpy(dt, assemble(e_F, interior_facet_domains=problem.fsi_boundary))
        eta_S.axpy(dt, assemble(e_S, interior_facet_domains=problem.fsi_boundary))
        eta_M.axpy(dt, assemble(e_M, interior_facet_domains=problem.fsi_boundary))

        end()

    # Compute sum of error indicators
    eta_K = Vector(Omega.num_cells())
    eta_K += eta_F
    eta_K += eta_S
    eta_K += eta_M

    # Plot residuals
    plot(vector_to_meshfunction(eta_F, Omega), title="Fluid error indicators")
    plot(vector_to_meshfunction(eta_S, Omega), title="Structure error indicators")
    plot(vector_to_meshfunction(eta_M, Omega), title="Mesh error indicators")
    plot(vector_to_meshfunction(eta_K, Omega), title="Total error indicators")
    interactive()

    return eta_K.array()

def compute_timestep(R, S, TOL, dt, t, T):
    """Compute new time step based on residual R, stability factor S,
    tolerance TOL, and the previous time step dt. The time step is
    adjusted so that we will not step beyond the given end time."""

    # Parameters for adaptive time-stepping
    C = 1.0               # interpolation constant
    safety_factor = 0.9   # safety factor for time step selection
    snap = 0.9            # snapping to end time when close
    conservation = 1.0    # time step conservation (high value means small change)

    # Compute new time step
    dt_new = safety_factor * TOL / (C*S*R)

    # FIXME: Temporary until we get real input
    dt_new  = dt

    # Modify time step to avoid oscillations
    dt_new = (1.0 + conservation) * dt * dt_new / (dt + conservation * dt_new)

    # Modify time step so we don't step beoynd end time
    at_end = False
    if dt_new > snap * (T - t):
        info("Close to t = T, snapping time step to end time: %g --> %g" % (dt_new, T - t))
        dt_new = T - t
        at_end = True

    info("Changing time step: %g --> %g" % (dt, dt_new))

    return dt_new, at_end
