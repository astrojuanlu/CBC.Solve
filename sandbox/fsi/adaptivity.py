"This module implements functionality for adaptivity."

__author__ = "Kristoffer Selim and Anders Logg"
__copyright__ = "Copyright (C) 2010 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

# Last changed: 2010-10-03

from dolfin import info
from numpy import zeros, argsort, linalg

from residuals import *
from storage import *
from spaces import *
from utils import *

# Variables for time residual
U0 = U1 = w = None

# Variables for storing adaptive data
refinement_level = 0
min_timestep = None

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
    W = create_dual_space(Omega)
    w = TestFunctions(W)

    # Create time series
    primal_series = create_primal_series()
    dual_series = create_dual_series()

    # Create primal functions
    U0 = create_primal_functions(Omega)
    U1 = create_primal_functions(Omega)

    # Create dual functions
    ZZ0, Z0 = create_dual_functions(Omega)
    ZZ1, Z1 = create_dual_functions(Omega)

    # Define midpoint values for primal and dual functions
    U = [0.5 * (U0[i] + U1[i]) for i in range(5)]
    Z = [0.5 * (Z0[i] + Z1[i]) for i in range(6)]

    # Define function spaces for extrapolation
    V2 = VectorFunctionSpace(Omega, "CG", 2)
    V3 = VectorFunctionSpace(Omega, "CG", 3)
    Q2 = FunctionSpace(Omega, "CG", 2)

    # Define functions for extrapolation
    EZ = [Function(EV) for EV in (V3, Q2, V2, V2, V2, V2)]

    # Define time step (value set in each time step)
    kn = Constant(0.0)

    # Get strong residuals for E_h
    Rh_F, Rh_S, Rh_M = strong_residuals(U0, U1, U, Z, EZ, dg, kn, problem)

    # Get weak residuals for E_k
    Rk_F, Rk_S, Rk_M = weak_residuals(U0, U1, U1, w, kn, problem)

    # Get weak residuals for E_c
    Rc_F, Rc_S, Rc_M = weak_residuals(U0, U1, U, Z, kn, problem)

    # Reset vectors for assembly of residuals
    eta_F = zeros(Omega.num_cells())
    eta_S = zeros(Omega.num_cells())
    eta_M = zeros(Omega.num_cells())

    # Reset variables
    E_k = 0.0
    E_c = 0.0
    ST = 0.0

    # Sum residuals over time intervals
    timestep_range = read_timestep_range(problem.end_time(), primal_series)
    for i in range(1, len(timestep_range)):
    # FIXME: Temporary while testing
    #for i in range(len(timestep_range) / 2, len(timestep_range) / 2 + 1):

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
        info("Extrapolating dual solution")
        [EZ[i].extrapolate(Z1[i]) for i in range(6)]

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
        s = 0.5 * linalg.norm(ZZ0.vector().array() - ZZ1.vector().array(), 2) / dt

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

    # Retrieve weights
    W_h = problem.space_error_weight()
    W_k = problem.time_error_weight()
    W_c = problem.non_galerkin_error_weight() 

    # Compute total weigted error
    E = E_h * W_h + E_k * W_k + E_c * W_c
        
    # Report results
    save_errors(E, E_h, E_k, E_c, ST, W_h, W_k, W_c)
    save_indicators(eta_F, eta_S, eta_M, eta_K)

    return E, eta_K, ST

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
    r = assemble(r_F + r_S + r_M, interior_facet_domains=problem.fsi_boundary)

    # Compute l^2 norm
    Rk = norm(r, "l2")

    info("Time residual is Rk = %g" % Rk)

    return Rk

def refine_mesh(problem, mesh, indicators):
    "Refine mesh based on error indicators"

    # Set cell markers using Dorfler marking
    fraction = problem.dorfler_fraction()
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
    #plot(plot_markers, title="Markers")

    # Refine mesh
    refined_mesh = refine(mesh, markers)

    # Save mesh to file
    save_mesh(mesh, refined_mesh)

    return refined_mesh

def compute_time_step(problem, Rk, ST, TOL, dt, t1, T):
    """Compute new time step based on residual R, stability factor S,
    tolerance TOL, and the previous time step dt. The time step is
    adjusted so that we will not step beyond the given end time."""

    # Parameters for adaptive time-stepping
    safety_factor = 0.9   # safety factor for time step selection
    snap = 0.9            # snapping to end time when close
    conservation = 1.0    # time step conservation (high value means small change)

    # Compute new time step
    dt_new = safety_factor * TOL * problem.time_error_weight() / (ST * Rk)

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
    save_timestep(t1, Rk, dt)

    info("Changing time step: %g --> %g" % (dt, dt_new))

    return dt_new, at_end

def initial_timestep(problem):
    "Return initial time step"

    global min_timestep

    # Get initial timestep for problem
    dt = problem.initial_timestep()

    # Use the smallest time step so far
    if (not min_timestep is None) and min_timestep < dt:
        dt = min_timestep

    return dt

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

def save_errors(E, E_h, E_k, E_c, ST, W_h, W_k, W_c):
    "Save errors to file"

    global refinement_level
    
    # Summarize errors
    summary = """

Estimating weighted error
-------------------------
Adaptive loop no. = %d
-------------------------

E_h * %g = %g
E_k * %g = %g  
E_c * %g = %g
  
E_tot = %g 
S(T)  = %g

""" % (refinement_level, W_h, E_h, W_k, E_k, W_c, E_c, E, ST)

    # Print summary
    info(summary)

    # Save to file
    f = open("adaptivity/adaptivity.log", "a")
    f.write(summary)
    f.close()

def save_indicators(eta_F, eta_S, eta_M, eta_K):
    "Save indicators to file"

    global refinement_level

    save_array(eta_F, "adaptivity/eta_F_%d.xml" % refinement_level)
    save_array(eta_S, "adaptivity/eta_S_%d.xml" % refinement_level)
    save_array(eta_M, "adaptivity/eta_M_%d.xml" % refinement_level)
    save_array(eta_K, "adaptivity/eta_K_%d.xml" % refinement_level)

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
