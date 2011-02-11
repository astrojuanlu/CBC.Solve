"This module implements functionality for adaptivity."

__author__ = "Kristoffer Selim and Anders Logg"
__copyright__ = "Copyright (C) 2010 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

# Last changed: 2011-02-11

from dolfin import info
from numpy import zeros, argsort, linalg

from residuals import *
from storage import *
from spaces import *
from utils import *
from sys import exit

# Variables for time residual
U0 = U1 = w = None

# Variables for storing adaptive data
_refinement_level = -1
min_timestep = None

# Create files for plotting error indicators
indicator_files = None

def estimate_error(problem, parameters):
    "Estimate error and compute error indicators"

    # Get meshes
    Omega = problem.mesh()
    Omega_F = problem.fluid_mesh()
    Omega_S = problem.structure_mesh()

    # Define projection space (piecewise constants)
    DG = FunctionSpace(Omega, "DG", 0)
    dg = TestFunction(DG)

    # Create dual function space and test functions
    W = create_dual_space(Omega)
    w = TestFunctions(W)

    # Create time series
    primal_series = create_primal_series(parameters)
    dual_series = create_dual_series(parameters)

    # Create primal functions
    U0 = create_primal_functions(Omega)
    U1 = create_primal_functions(Omega)

    # Create dual functions
    ZZ0, Z0 = create_dual_functions(Omega)
    ZZ1, Z1 = create_dual_functions(Omega)

    # Define function spaces for extrapolation
    V2 = VectorFunctionSpace(Omega, "CG", 2)
    V3 = VectorFunctionSpace(Omega, "CG", 3)
    Q2 = FunctionSpace(Omega, "CG", 2)

    # Define functions for extrapolation
    EZ0 = [Function(EV) for EV in (V3, Q2, V2, V2, V2, V2)]
    EZ1 = [Function(EV) for EV in (V3, Q2, V2, V2, V2, V2)]

    # Define midpoint values for primal and dual functions
    U  = [0.5 * (U0[i]  + U1[i])  for i in range(5)]
    Z  = [0.5 * (Z0[i]  + Z1[i])  for i in range(6)]
    EZ = [0.5 * (EZ0[i] + EZ1[i]) for i in range(6)]

    # Define time step (value set in each time step)
    kn = Constant(0.0)

    # Get strong residuals for E_h
    Rh_F, Rh_S, Rh_M = strong_residuals(U0, U1, U, Z, EZ, dg, kn, problem)

    # Get weak residuals for E_k
    Rk_F, Rk_S, Rk_M = weak_residuals(U0, U1, U1, w, kn, problem)

    # Get weak residuals for E_c
    Rc_F, Rc_S, Rc_M = weak_residuals(U0, U1, U, Z, kn, problem)

    # Reset vectors for assembly of residuals
    eta_F = None
    eta_S = None
    eta_M = None

    # Reset variables
    E_k   = 0.0
    E_c   = 0.0
    E_c_F = 0.0
    E_c_S = 0.0
    E_c_M = 0.0
    ST    = 0.0

    # Sum residuals over time intervals
    timestep_range = read_timestep_range(problem.end_time(), primal_series)
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
        read_primal_data(U0, t0, Omega, Omega_F, Omega_S, primal_series)
        read_primal_data(U1, t1, Omega, Omega_F, Omega_S, primal_series)

        # Read dual data
        read_dual_data(ZZ0, t0, dual_series)
        read_dual_data(ZZ1, t1, dual_series)

        # Extrapolate dual data
        [EZ0[j].extrapolate(Z0[j]) for j in range(6)]
        [EZ1[j].extrapolate(Z1[j]) for j in range(6)]

        # Assemble strong residuals for space discretization error
        info("Assembling error contributions")
        e_F = [assemble(Rh_Fi, interior_facet_domains=problem.fsi_boundary, cell_domains=problem.cell_domains) for Rh_Fi in Rh_F]
        e_S = [assemble(Rh_Si, interior_facet_domains=problem.fsi_boundary, cell_domains=problem.cell_domains) for Rh_Si in Rh_S]
        e_M = [assemble(Rh_Mi, interior_facet_domains=problem.fsi_boundary, cell_domains=problem.cell_domains) for Rh_Mi in Rh_M]

        # Assemble weak residuals for time discretization error
        Rk = norm(assemble(Rk_F + Rk_S + Rk_M, interior_facet_domains=problem.fsi_boundary, cell_domains=problem.cell_domains))

        # Assemble weak residuals for computational error
        RcF = assemble(Rc_F, mesh=Omega, interior_facet_domains=problem.fsi_boundary, cell_domains=problem.cell_domains)
        RcS = assemble(Rc_S, mesh=Omega, interior_facet_domains=problem.fsi_boundary, cell_domains=problem.cell_domains)
        RcM = assemble(Rc_M, mesh=Omega, interior_facet_domains=problem.fsi_boundary, cell_domains=problem.cell_domains)

        # Estimate interpolation error (local weight)
        s = 0.5 * linalg.norm(ZZ0.vector().array() - ZZ1.vector().array(), 2) / dt

        # Reset vectors for assembly of residuals
        eta_F = [zeros(Omega.num_cells()) for i in range(len(e_F))]
        eta_S = [zeros(Omega.num_cells()) for i in range(len(e_S))]
        eta_M = [zeros(Omega.num_cells()) for i in range(len(e_M))]

        # Add to error indicators
        for i in range(len(e_F)):
            eta_F[i] += dt * abs(e_F[i].array())
        for i in range(len(e_S)):
            eta_S[i] += dt * abs(e_S[i].array())
        for i in range(len(e_M)):
            eta_M[i] += dt * abs(e_M[i].array())

        # Add to E_k
        E_k += dt * s * dt * Rk

        # Add to E_c's
        E_c_F += dt * RcF
        E_c_S += dt * RcS
        E_c_M += dt * RcM

        # Sum total computational error
        E_c = E_c_F + E_c_S + E_c_M

        # Add to stability factor
        ST += dt * s

        end()

    # Compute sum of error indicators
    eta_K = sum(eta_F) + sum(eta_S) + sum(eta_M)

    # Compute space discretization error
    E_h = sum(eta_K)

    # Compute total error
    E = E_h + E_k + abs(E_c)

    # Report results
    save_errors(E, E_h, E_k, E_c, E_c_F, E_c_S, E_c_M, ST, parameters)
#    save_computational_errors(E_c_F, E_c_S, E_c_M, parameters)
    save_indicators(eta_F, eta_S, eta_M, eta_K, Omega, parameters)
    save_stability_factor(T, ST, parameters)

    return E, eta_K, ST, E_h

def compute_time_residual(primal_series, t0, t1, problem):
    "Compute size of time residual"

    info("Computing time residual")

    # Get meshes
    Omega = problem.mesh()
    Omega_F = problem.fluid_mesh()
    Omega_S = problem.structure_mesh()

    # Initialize solution variables (only first time)
    global U0, U1, w
    if t0 == 0.0:

        # Create primal variables
        info("Initializing primal variables for time residual")
        U0 = create_primal_functions(Omega)
        U1 = create_primal_functions(Omega)

        # Create dual function space and test functions
        W = create_dual_space(Omega)
        w = TestFunctions(W)

    # Read solution data
    read_primal_data(U0, t0, Omega, Omega_F, Omega_S, primal_series)
    read_primal_data(U1, t1, Omega, Omega_F, Omega_S, primal_series)

    # Set time step
    kn = Constant(t1 - t0)

    # Get weak residuals
    r_F, r_S, r_M = weak_residuals(U0, U1, U1, w, kn, problem)

    # Assemble residual
    r = assemble(r_F + r_S + r_M, interior_facet_domains=problem.fsi_boundary, cell_domains=problem.cell_domains)

    # Compute l^2 norm
    Rk = norm(r, "l2")

    info("Time residual is Rk = %g" % Rk)

    return Rk

def refine_mesh(problem, mesh, indicators, parameters):
    "Refine mesh based on error indicators"

    # Get fraction of elements for refinement
    fraction = parameters["marking_fraction"]

    # Create lists and markers for refinement
    indices = list(argsort(indicators))
    indices.reverse()
    markers = MeshFunction("bool", mesh, mesh.topology().dim())
    markers.set_all(False)

    # Dorfler marking strategy
    if parameters["dorfler_marking"]:
        info_blue("Refining using Dorfler fraction = %g" %fraction)
        total_sum = sum(indicators)
        sub_sum = 0.0
        for i in indices:
            sub_sum += indicators[i]
            markers[int(i)] = True
            if sub_sum >= fraction * total_sum:
                  break

    # "Standard" marking strategy
    else:
        info_blue("Refining with fraction = %g" %fraction)
        stopping_criteria = int(round(len(indices) * fraction))
        counter = 0
        for i in indices:
            counter += 1
            markers[int(i)] = True
            if counter == stopping_criteria:
                break

    # Save marked cells (for plotting)
    save_refinement_markers(mesh, markers)

    # Refine mesh
    refined_mesh = refine(mesh, markers)

    return refined_mesh

def compute_time_step(problem, Rk, ST, TOL, dt, t1, T, w_k, parameters):
    """Compute new time step based on residual R, stability factor S,
    tolerance TOL, and the previous time step dt. The time step is
    adjusted so that we will not step beyond the given end time."""

    # Parameters for adaptive time-stepping
    safety_factor = 0.9   # safety factor for time step selection
    snap = 0.9            # snapping to end time when close
    conservation = 1.0    # time step conservation (high value means small change)

    # Compute new time step
    dt_new = safety_factor * TOL * w_k / (ST * Rk)

    # Modify time step to avoid oscillations
    dt_new = (1.0 + conservation) * dt * dt_new / (dt + conservation * dt_new)

    # Modify time step so we don't step beoynd end time
    at_end = False
    if dt_new > snap * (T - t1):
        info("Close to t = T, snapping time step to end time: %g --> %g" % (dt_new, T - t1))
        dt_new = T - t1
        at_end = True

    # Store minimum time step
    global min_timestep
    if min_timestep is None or dt_new < min_timestep:
        min_timestep = dt_new

    # Save time step
    save_timestep(t1, Rk, dt, parameters)
    if at_end:
        save_timestep(T, Rk, dt_new, parameters)

    info("Changing time step: %g --> %g" % (dt, dt_new))

    return dt_new, at_end

def initial_timestep(problem, parameters):
    "Return initial time step"

    global min_timestep

    # Get initial time step from parameters
    dt = parameters["initial_timestep"]

    # Use the smallest time step so far
    if (not min_timestep is None) and min_timestep < dt:
        dt = min_timestep

    return dt

def compute_itertol(problem, w_c, TOL, dt, t1, parameters):
    "Compute tolerance for FSI iterations"

    if parameters["uniform_timestep"]:
        tol = parameters["fixedpoint_tolerance"]
        info("")
        info_blue("  * Tolerance for (f)-(S)-(M) iteration is fixed to %g" % tol)
        end()

    else:
        S_c = 1.0 # not computed
        tol = w_c * TOL * dt / S_c
        info("")
        info_blue("  * Changing tolerance for (f)-(S)-(M) iteration to %g" % tol)
        end()

    # Save FSI iteration tolerance to file
    save_itertol(t1, tol, parameters)

    return tol

def save_mesh(mesh, parameters):
    "Save mesh to file"

    global _refinement_level

    # Increase refinement level
    _refinement_level += 1

    # Save refined mesh
    file = File("%s/mesh_%d.xml" % (parameters["output_directory"], _refinement_level))
    file << mesh

def save_errors(E, E_h, E_k, E_c, E_c_F, E_c_S, E_c_M, ST, parameters):
    "Save errors to file"

    global _refinement_level

    # Summarize errors
    summary = """

Estimating error
-------------------------
Adaptive loop no. = %d
-------------------------

E_h  = %g
E_k  = %g
E_c  = %g

E_tot = %g
S(T)  = %g

""" % (_refinement_level, E_h, E_k, abs(E_c), E, ST)

    # Print summary
    info(summary)

    # Save to log file
    f = open("%s/adaptivity.log" % parameters["output_directory"], "a")
    f.write(summary)
    f.close()

    # Save to file (for plotting)
    g = open("%s/error_estimates.txt" % parameters["output_directory"], "a")
    g.write("%d %g %g %g %g %g %g %g %g \n" %(_refinement_level, E, E_h, E_k, abs(E_c), E_c_F, E_c_S, E_c_M, ST))
    g.close()

def save_timestep(t1, Rk, dt, parameters):
    "Save time step to file"

    global _refinement_level

    f = open("%s/timesteps.txt" % parameters["output_directory"], "a")
    f.write("%d %g %g %g\n" % (_refinement_level, t1, dt, Rk))
    f.close()

def save_stability_factor(T, ST, parameters):
    "Save Galerkin stability factor"

    global _refinement_level

    f = open("%s/stability_factor.txt" % parameters["output_directory"], "a")
    f.write("%g %g\n" % (T, ST))
    f.close()

def save_goal_functional(t1, goal_functional, integrated_goal_functional, parameters):
    "Saving goal functional at t = t1"

    global _refinement_level

    info("Value of goal functional at t = %g: %g" % (t1, goal_functional))
    f = open("%s/goal_functional.txt" % parameters["output_directory"], "a")
    f.write("%d %.16g %.16g %.16g\n" % (_refinement_level, t1, goal_functional, integrated_goal_functional))
    f.close()

def save_goal_functional_final(goal_functional, integrated_goal_functional, parameters):
    "Saving goal functional at final time"

    global _refinement_level

    info("Value of goal functional at T: %g" % goal_functional)
    f = open("%s/goal_functional_final.txt" % parameters["output_directory"], "a")
    f.write("%d %.16g %.16g\n" % (_refinement_level, goal_functional, integrated_goal_functional))
    f.close()

def save_itertol(t1, tol, parameters):
    "Save FSI iteration tolerance"

    global refinment_level

    f = open("%s/fsi_tolerance.txt" % parameters["output_directory"], "a")
    f.write("%d %g %g \n" % (_refinement_level, t1, tol))
    f.close()

def save_no_FSI_iter(t1, no, parameters):
    "Save number of FSI iterations"

    global _refinement_level

    f = open("%s/no_iterations.txt" % parameters["output_directory"], "a")
    f.write("%d %g %g \n" % (_refinement_level, t1, no))
    f.close()

def save_dofs(num_dofs_FSM, timestep_counter, parameters):
    "Save number of total number of dofs"

    global _refinement_level

    # Calculate total number of dofs
    space_dofs = num_dofs_FSM
    time_dofs  = timestep_counter
    dofs       = space_dofs * time_dofs

    f = open("%s/num_dofs.txt" % parameters["output_directory"], "a")
    f.write("%d %g %g %g \n" %(_refinement_level, dofs, space_dofs, time_dofs))
    f.close()

def save_indicators(eta_F, eta_S, eta_M, eta_K, Omega, parameters):
    "Save mesh function for visualization"

    global indicator_files
    global _refinement_level

    # Create mesh functions
    plot_markers_F = [MeshFunction("double", Omega, Omega.topology().dim()) for i in range(len(eta_F))]
    plot_markers_S = [MeshFunction("double", Omega, Omega.topology().dim()) for i in range(len(eta_S))]
    plot_markers_M = [MeshFunction("double", Omega, Omega.topology().dim()) for i in range(len(eta_M))]
    plot_markers_K =  MeshFunction("double", Omega, Omega.topology().dim())

    # Extract error indicators
    for i in range(Omega.num_cells()):
        for j in range(len(eta_F)): plot_markers_F[j][i] = eta_F[j][i]
        for j in range(len(eta_S)): plot_markers_S[j][i] = eta_S[j][i]
        for j in range(len(eta_M)): plot_markers_M[j][i] = eta_M[j][i]
        plot_markers_K[i] = eta_K[i]

    # Sum markers
    plot_markers = plot_markers_F + plot_markers_S + plot_markers_M + [plot_markers_K]

    # Create indicator files (including file for refinement markers not used here)
    if indicator_files is None:
        indicator_files = \
            [File("%s/pvd/level_%d/eta_F_%d.pvd" % (parameters["output_directory"], _refinement_level, i)) for i in range(len(eta_F))] + \
            [File("%s/pvd/level_%d/eta_S_%d.pvd" % (parameters["output_directory"], _refinement_level, i)) for i in range(len(eta_S))] + \
            [File("%s/pvd/level_%d/eta_M_%d.pvd" % (parameters["output_directory"], _refinement_level, i)) for i in range(len(eta_M))] + \
            [File("%s/pvd/level_%d/eta_K.pvd" % (parameters["output_directory"], _refinement_level))] + \
            [File("%s/pvd/level_%d/refinement_markers.pvd" % (parameters["output_directory"], _refinement_level))]

    # Save error indicators
    for i in range(len(indicator_files) - 1):
        indicator_files[i] << plot_markers[i]

def save_refinement_markers(mesh, markers):
    "Save refinement markers for visualization"

    # Create mesh functions
    refinement_markers = MeshFunction("uint", mesh, mesh.topology().dim())

    # Reset plot markers
    refinement_markers.set_all(0)

    # Extract error indicators
    for i in range(mesh.num_cells()):
        if markers[i]:
            refinement_markers[i] = True

    # Save markers
    indicator_files[4] << refinement_markers

def refinement_level():
    "Return current refinement level"
    global _refinement_level
    return _refinement_level
