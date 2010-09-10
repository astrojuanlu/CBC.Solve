"""This module implements all residuals used for adaptivity. Note that
time residuals need to be initialized before they are first used to
enable reuse of forms between time steps."""

__author__ = "Kristoffer Selim and Anders Logg"
__copyright__ = "Copyright (C) 2010 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

from operators import Sigma_F as _Sigma_F
from operators import Sigma_S as _Sigma_S
from operators import Sigma_M as _Sigma_M
from operators import F, J
from storage import *

def evaluate_space_residuals(problem):
    "Evaluate residuals in space"

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

    # Fluid residuals
    Rh_F1 = w*inner(ZZ_F - Z_F, Dt_U_F - div(Sigma_F))*dx
    Rh_F2 = avg(w)*inner(ZZ_F('+') - Z_F('+'), jump(dot(Sigma_F, N_F)))*dS
    Rh_F3 = w*inner(YY_F - Y_F, div(J(U_M)*dot(inv(F(U_M)), U_F)))*dx

    # Structure residuals
    Rh_S1 = w*inner(ZZ_S - Z_S, Dt_P_S - div(Sigma_S))*dx
    Rh_S2 = avg(w)*inner(ZZ_S('+') - Z_S('+'), jump(dot(Sigma_S, N_S)))*dS
    Rh_S3 = avg(w)*inner(ZZ_S - Z_S, dot(Sigma_S - Sigma_F, N_S))('+')*dS(1)
    Rh_S4 = w*inner(YY_S - Y_S, Dt_U_S - P_S)*dx

    # Mesh residuals
    Rh_M1 = w*inner(ZZ_M - Z_M, Dt_U_M - div(Sigma_M))*dx
    Rh_M2 = avg(w)*inner(ZZ_M('+') - Z_M('+'), jump(dot(Sigma_M, N_F)))*dS
    Rh_M3 = avg(w)*inner(YY_M - Y_M, U_M - U_S)('+')*dS(1)

    # Collect residuals
    Rh_F = Rh_F1 + Rh_F2 + Rh_F3
    Rh_S = Rh_S1 + Rh_S2 + Rh_S3 + Rh_S4
    Rh_M = Rh_M1 + Rh_M2 + Rh_M3

    # Reset vectors for assembly of residuals
    rh_F = Vector(Omega.num_cells())
    rh_S = Vector(Omega.num_cells())
    rh_M = Vector(Omega.num_cells())

    # Sum residuals over time intervals
    timestep_range = read_timestep_range(problem)
    for i in range(1, len(timestep_range)):

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
        ZZ_F.extrapolate(Z_F)
        YY_F.extrapolate(Y_F)
        ZZ_S.extrapolate(Z_S)
        YY_S.extrapolate(Y_S)
        ZZ_M.extrapolate(Z_M)
        YY_M.extrapolate(Y_M)

        # Assemble residuals
        rh_F.axpy(dt, assemble(Rh_F, interior_facet_domains=problem.fsi_boundary))
        rh_S.axpy(dt, assemble(Rh_S, interior_facet_domains=problem.fsi_boundary))
        rh_M.axpy(dt, assemble(Rh_M, interior_facet_domains=problem.fsi_boundary))

        end()

    # Compute sum of error indicators
    rh = Vector(Omega.num_cells())
    rh += rh_F
    rh += rh_S
    rh += rh_M

    # Convert to mesh functions
    rh_F = vector_to_meshfunction(rh_F, Omega)
    rh_S = vector_to_meshfunction(rh_S, Omega)
    rh_M = vector_to_meshfunction(rh_M, Omega)
    rh   = vector_to_meshfunction(rh, Omega)

    # Plot residuals
    plot(rh_F, title="Fluid residuals")
    plot(rh_S, title="Structure residuals")
    plot(rh_M, title="Mesh residuals")

    return rh

def evaluate_time_residuals():
    "Evaluate residuals in time"

    pass

def evaluate_computational_residuals():
    "Evaluate computational residuals"

    pass

def init_time_residuals():
    "Initialize time residuals"

    pass

def vector_to_meshfunction(x, Omega):
    "Convert vector x to cell function on Omega"
    f = CellFunction("double", Omega)
    if not f.size() == x.size():
        raise RuntimeError, "Size of vector does not match number of cells."
    xx = x.array()
    for i in range(x.size()):
        f[i] = xx[i]
    return f
