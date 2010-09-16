"This module implements functionality for adaptivity."

__author__ = "Kristoffer Selim and Anders Logg"
__copyright__ = "Copyright (C) 2010 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

# Last changed: 2010-09-16

from dolfin import info
from numpy import zeros, argsort, linalg

from residuals import *
from storage import *
from utils import *

# Variables for time residual
U_F0 = P_F0 = U_S0 = P_S0 = U_M0 = None
U_F1 = P_F1 = U_S1 = P_S1 = U_M1 = None
v_F = q_F = v_S = q_S = v_M = q_M = None

# Variables for storing adaptive data
refinement_level = 0

def estimate_error(problem):
    "Estimate error and compute error indicators"

    # Get meshes
    Omega = problem.mesh()
    Omega_F = problem.fluid_mesh()
    Omega_S = problem.structure_mesh()

    # Define projection space (piecewise constants)
    DG = FunctionSpace(Omega, "DG", 0)
    dg = TestFunction(DG)

    # Create dual function space and test functions
    W = init_dual_space(Omega)
    v_F, q_F, v_S, q_S, v_M, q_M = TestFunctions(W)

    # Initialize primal functions
    U_F0, P_F0, U_S0, P_S0, U_M0 = init_primal_data(Omega)
    U_F1, P_F1, U_S1, P_S1, U_M1 = init_primal_data(Omega)

    # Initialize dual functions
    Z0, (Z_F0, Y_F0, Z_S0, Y_S0, Z_M0, Y_M0) = init_dual_data(Omega)
    Z1, (Z_F1, Y_F1, Z_S1, Y_S1, Z_M1, Y_M1) = init_dual_data(Omega)

    # Define midpoint values for primal functions
    U_F = 0.5 * (U_F0 + U_F1)
    P_F = 0.5 * (P_F0 + P_F1)
    U_S = 0.5 * (U_S0 + U_S1)
    P_S = 0.5 * (P_S0 + P_S1)
    U_M = 0.5 * (U_M0 + U_M1)

    # Define midpoint values for dual functions
    Z_F = 0.5 * (Z_F0 + Z_F1)
    Y_F = 0.5 * (Y_F0 + Y_F1)
    Z_S = 0.5 * (Z_S0 + Z_S1)
    Y_S = 0.5 * (Y_S0 + Y_S1)
    Z_M = 0.5 * (Z_M0 + Z_M1)
    Y_M = 0.5 * (Y_M0 + Y_M1)

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

    # Get strong residuals for E_h
    Rh_F, Rh_S, Rh_M = strong_residuals(U_F0, P_F0, U_S0, P_S0, U_M0,
                                        U_F1, P_F1, U_S1, P_S1, U_M1,
                                        U_F,  P_F,  U_S,  P_S,  U_M,
                                        Z_F1, Y_F1, Z_S1, Y_S1, Z_M1, Y_M1,
                                        ZZ_F, YY_F, ZZ_S, YY_S, ZZ_M, YY_M,
                                        dg, kn, problem)

    # Get weak residuals for E_k
    Rk_F, Rk_S, Rk_M = weak_residuals(U_F0, P_F0, U_S0, P_S0, U_M0,
                                      U_F1, P_F1, U_S1, P_S1, U_M1,
                                      U_F1, P_F1, U_S1, P_S1, U_M1,
                                      v_F, q_F, v_S, q_S, v_M, q_M,
                                      kn, problem)

    # Get weak residuals for E_c
    Rc_F, Rc_S, Rc_M = weak_residuals(U_F0, P_F0, U_S0, P_S0, U_M0,
                                      U_F1, P_F1, U_S1, P_S1, U_M1,
                                      U_F,  P_F,  U_S,  P_S,  U_M,
                                      Z_F, Y_F, Z_S, Y_S, Z_M, Y_M,
                                      kn, problem)

    # Reset vectors for assembly of residuals
    eta_F = zeros(Omega.num_cells())
    eta_S = zeros(Omega.num_cells())
    eta_M = zeros(Omega.num_cells())

    # Reset variables
    E_k = 0.0
    E_c = 0.0
    ST = 0.0

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

        # Read dual data
        read_dual_data(Z0, t0)
        read_dual_data(Z1, t1)

        # Extrapolate dual data
        info("Extrapolating dual solution")
        ZZ_F.extrapolate(Z_F1)
        YY_F.extrapolate(Y_F1)
        ZZ_S.extrapolate(Z_S1)
        YY_S.extrapolate(Y_S1)
        ZZ_M.extrapolate(Z_M1)
        YY_M.extrapolate(Y_M1)

        # Assemble strong residuals for space discretization error
        info("Assembling error contributions")
        e_F = [assemble(Rh_Fi, interior_facet_domains=problem.fsi_boundary) for Rh_Fi in Rh_F]
        e_S = [assemble(Rh_Si, interior_facet_domains=problem.fsi_boundary) for Rh_Si in Rh_S]
        e_M = [assemble(Rh_Mi, interior_facet_domains=problem.fsi_boundary) for Rh_Mi in Rh_M]

        # Assemble weak residuals for time discretization error
        Rk = norm(assemble(Rk_F + Rk_S + Rk_M, interior_facet_domains=problem.fsi_boundary))

        # Assemble weak residuals for computational error
        Rc = assemble(Rc_F + Rc_S + Rc_M, mesh=Omega, interior_facet_domains=problem.fsi_boundary)

        # Estimate interpolation error (local weight)
        s = 0.5 * linalg.norm(Z0.vector().array() - Z1.vector().array(), 2) / dt

        # Add to error indicators
        eta_F += dt * sum(abs(e.array()) for e in e_F)
        eta_S += dt * sum(abs(e.array()) for e in e_S)
        eta_M += dt * sum(abs(e.array()) for e in e_M)

        # Add to E_k
        E_k += dt * s * dt * Rk

        # Add to E_c
        E_c += dt * Rc

        # Add to stability factor
        ST += dt * s

        end()

    # Compute sum of error indicators
    eta_K = eta_F + eta_S + eta_M

    # Compute space discretization error
    E_h = sum(eta_K)

    # Compute total error
    E = E_h + E_k + abs(E_c)

    # Report results
    save_errors(E, E_h, E_k, E_c, ST)
    save_indicators(eta_F, eta_S, eta_M, eta_K)

    return E, eta_K, ST

def compute_time_residual(time_series, t0, t1, problem):
    "Compute size of time residual"

    info("Computing time residual")

    # Get meshes
    Omega = problem.mesh()
    Omega_F = problem.fluid_mesh()
    Omega_S = problem.structure_mesh()

    # Initialize solution variables (only first time)
    global U_F0, P_F0, U_S0, P_S0, U_M0
    global U_F1, P_F1, U_S1, P_S1, U_M1
    global v_F, q_F, v_S, q_S, v_M, q_M
    if t0 == 0.0:

        # Create primal variables
        info("Initializing primal variables for time residual")
        U_F0, P_F0, U_S0, P_S0, U_M0 = init_primal_data(Omega)
        U_F1, P_F1, U_S1, P_S1, U_M1 = init_primal_data(Omega)

        # Create dual function space and test functions
        W = init_dual_space(Omega)
        v_F, q_F, v_S, q_S, v_M, q_M = TestFunctions(W)

    # Read solution data
    read_primal_data(U_F0, P_F0, U_S0, P_S0, U_M0, t0, Omega, Omega_F, Omega_S)
    read_primal_data(U_F1, P_F1, U_S1, P_S1, U_M1, t1, Omega, Omega_F, Omega_S)

    # Set time step
    kn = Constant(t1 - t0)

    # Get weak residuals
    r_F, r_S, r_M = weak_residuals(U_F0, P_F0, U_S0, P_S0, U_M0,
                                   U_F1, P_F1, U_S1, P_S1, U_M1,
                                   U_F1, P_F1, U_S1, P_S1, U_M1,
                                   v_F, q_F, v_S, q_S, v_M, q_M,
                                   kn, problem)

    # Assemble residual
    r = assemble(r_F + r_S + r_M, interior_facet_domains=problem.fsi_boundary)

    # Compute l^2 norm
    Rk = norm(r, "l2")

    info("Time residual is Rk = %g" % Rk)

    return Rk

def refine_mesh(mesh, indicators):
    "Refine mesh based on error indicators"

    # Set cell markers using Dorfler marking
    fraction = 0.5
    indices = list(argsort(indicators))
    indices.reverse()
    sub_sum = 0.0
    total_sum = sum(indicators)
    markers = CellFunction("bool", mesh)
    markers.set_all(False)
    for i in indices:
        sub_sum += indicators[i]
        markers[int(i)] = True
        if sub_sum >= fraction*total_sum:
            break

    # Plot markers (convert to uint so it can be plotted)
    plot_markers = CellFunction("uint", mesh)
    plot_markers.set_all(0)
    for i in range(plot_markers.size()):
        if markers[i]:
            plot_markers[i] = True
    plot(plot_markers, title="Markers")

    # Refine mesh
    refined_mesh = refine(mesh, markers)

    # Save mesh to file
    save_mesh(mesh, refined_mesh)

    return refined_mesh

def compute_timestep(Rk, ST, TOL, dt, t1, T):
    """Compute new time step based on residual R, stability factor S,
    tolerance TOL, and the previous time step dt. The time step is
    adjusted so that we will not step beyond the given end time."""

    # Parameters for adaptive time-stepping
    safety_factor = 0.9   # safety factor for time step selection
    snap = 0.9            # snapping to end time when close
    conservation = 1.0    # time step conservation (high value means small change)

    # Compute new time step
    dt_new = safety_factor * TOL / (ST * Rk)

    # Modify time step to avoid oscillations
    dt_new = (1.0 + conservation) * dt * dt_new / (dt + conservation * dt_new)

    # Modify time step so we don't step beoynd end time
    at_end = False
    if dt_new > snap * (T - t1):
        info("Close to t = T, snapping time step to end time: %g --> %g" % (dt_new, T - t1))
        dt_new = T - t1
        at_end = True

    # Save time step
    save_timestep(t1, Rk, dt)

    info("Changing time step: %g --> %g" % (dt, dt_new))

    return dt_new, at_end

def save_mesh(mesh, refined_mesh):
    "Save mesh to file"

    global refinement_level

    # Save initial mesh first time
    if refinement_level == 0:
        file = File("adaptivity/mesh_0.xml")
        file << mesh

    # Increase refinement level
    refinement_level += 1

    # Save refined mesh
    file = File("adaptivity/mesh_%d.xml" % refinement_level)
    file << refined_mesh

def save_errors(E, E_h, E_k, E_c, ST):
    "Save errors to file"

    global refinement_level

    # Summarize errors
    summary = """

Estimating error
----------------
level = %d
E_h   = %g
E_k   = %g
E_c   = %g
E     = E_h + E_k + E_c = %g
S(T)  = %g

""" % (refinement_level, E_h, E_k, E_c, E, ST)

    # Print summary
    info(summary)

    # Save to file
    f = open("adaptivity/adaptivity.log", "a")
    f.write(summary)
    f.close()

def save_indicators(eta_F, eta_S, eta_M, eta_K):
    "Save indicators to file"

    global refinement_level

    save_array(eta_F, "eta_F_%d.xml" % refinement_level)
    save_array(eta_S, "eta_S_%d.xml" % refinement_level)
    save_array(eta_M, "eta_M_%d.xml" % refinement_level)
    save_array(eta_K, "eta_K_%d.xml" % refinement_level)

def save_timestep(t1, Rk, dt):
    "Save time step to file"

    global refinement_level

    f = open("adaptivity/timesteps.txt", "a")
    f.write("%d %g %g %g\n" % (refinement_level, t1, dt, Rk))
    f.close()

def save_array(x, filename):
    "Save array to file"
    f = open(filename, "w")
    f.write(" ".join(str(xx) for xx in x))
    f.close()
