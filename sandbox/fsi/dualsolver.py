"This module implements the dual FSI solver."

__author__ = "Kristoffer Selim and Anders Logg"
__copyright__ = "Copyright (C) 2010 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

# Last changed: 2010-08-11

from dolfin import *
from dualproblem import dual_forms

class DualSolver:
    "Dual FSI solver"

    def __init__(self, problem, solver_parameters):
        "Create dual FSI solver"

        # Get solver parameters
        self.plot_solution = solver_parameters["plot_solution"]
        self.save_solution = solver_parameters["save_solution"]
        self.save_series = solver_parameters["save_series"]

        # Create files for saving to VTK
        if self.save_solution:
            Z_F_file = File("pvd/Z_F.pvd")
            Y_F_file = File("pvd/Y_F.pvd")
            Z_S_file = File("pvd/Z_S.pvd")
            Y_S_file = File("pvd/Y_S.pvd")
            Z_M_file = File("pvd/Z_M.pvd")
            Y_M_file = File("pvd/Y_M.pvd")

        # Create time series for storing solution
        if self.save_series:
            Z_F_series = TimeSeries("bin/Z_F")
            Y_F_series = TimeSeries("bin/Y_F")
            Z_S_series = TimeSeries("bin/Z_S")
            Y_S_series = TimeSeries("bin/Y_S")
            Z_M_series = TimeSeries("bin/Z_M")
            Y_M_series = TimeSeries("bin/Y_M")

        # Open time series for primal solution
        self.u_F_series = TimeSeries("bin/u_F")
        self.p_F_series = TimeSeries("bin/p_F")
        self.U_S_series = TimeSeries("bin/U_S")
        self.P_S_series = TimeSeries("bin/P_S")
        self.U_M_series = TimeSeries("bin/U_M")

        # Get nodal points for primal time series
        t = self.u_F_series.vector_times()
        T = problem.end_time()
        # FIXME: Test disable due to bug in DOLFIN Array typemap
        #if not (len(t) > 1 and t[0] == 0.0 and t[-1] == T):
        #    print "Nodal points for primal time series:", times
        #    raise RuntimeError, "Missing primal data, unable to solve dual problem."
        self.timestep_range = t

        # Store problem
        self.problem = problem

    def solve(self):
        "Solve the dual FSI problem"

        # Get problem parameters
        T = self.problem.end_time()
        Omega = self.problem.mesh()
        Omega_F = self.problem.fluid_mesh()
        Omega_S = self.problem.structure_mesh()

        # Define function spaces defined on the whole domain
        V_F1 = VectorFunctionSpace(Omega, "CG", 1)
        V_F2 = VectorFunctionSpace(Omega, "CG", 2)
        Q_F  = FunctionSpace(Omega, "CG", 1)
        V_S  = VectorFunctionSpace(Omega, "CG", 1)
        Q_S  = VectorFunctionSpace(Omega, "CG", 1)
        V_M  = VectorFunctionSpace(Omega, "CG", 1)
        Q_M  = VectorFunctionSpace(Omega, "CG", 1)

        # Create mixed function space
        W = MixedFunctionSpace([V_F2, Q_F, V_S, Q_S, V_M, Q_M])

        # Create test and trial functions
        (v_F, q_F, v_S, q_S, v_M, q_M) = TestFunctions(W)
        (Z_F, Y_F, Z_S, Y_S, Z_M, Y_M) = TrialFunctions(W)

        # Create solution functions
        Z0 = Function(W)
        Z1 = Function(W)
        (Z_F0, Y_F0, Z_S0, Y_S0, Z_M0, Y_M0) = Z0.split()
        (Z_F1, Y_F1, Z_S1, Y_S1, Z_M1, Y_M1) = Z1.split()

        # Create primal functions
        U_F0 = Function(V_F1); U_F1 = Function(V_F1)
        P_F0 = Function(Q_F);  P_F1 = Function(Q_F)
        U_S0 = Function(V_S);  U_S1 = Function(V_S)
        P_S0 = Function(Q_S);  P_S1 = Function(Q_S)
        U_M0 = Function(V_M);  U_M1 = Function(V_M)

        # Create time step (value set in each time step)
        k = Constant(0.0)

        # Create variational forms for dual problem
        A, L = dual_forms(Omega_F, Omega_S, k, self.problem,
                          v_F,  q_F,  v_S,  q_S,  v_M,  q_M,
                          Z_F,  Y_F,  Z_S,  Y_S,  Z_M,  Y_M,
                          Z_F0, Y_F0, Z_S0, Y_S0, Z_M0, Y_M0,
                          U_F0, P_F0, U_S0, P_S0, U_M0,
                          U_F1, P_F1, U_S1, P_S1, U_M1)

        # Define BCs (define the dual trial space = homo. Dirichlet BCs)
        bc_U_F0 = DirichletBC(W.sub(0), Constant((0.0, 0.0)), noslip)
        bc_U_F1 = DirichletBC(W.sub(0), Constant((0.0, 0.0)), interior_facet_domains, 1 )
        bc_P_F0 = DirichletBC(W.sub(1), Constant(0.0), inflow)
        bc_P_F1 = DirichletBC(W.sub(1), Constant(0.0), outflow)
        bc_P_F2 = DirichletBC(W.sub(1), Constant(0.0), interior_facet_domains, 1 )
        bc_U_S  = DirichletBC(W.sub(2), Constant((0.0, 0.0)), dirichlet_boundaries)
        bc_P_S  = DirichletBC(W.sub(3), Constant((0.0, 0.0)), dirichlet_boundaries)
        bc_U_M1 = DirichletBC(W.sub(4), Constant((0.0, 0.0)), DomainBoundary())
        bc_U_M2 = DirichletBC(W.sub(4), Constant((0.0, 0.0)), interior_facet_domains, 1)
        bc_P_M1 = DirichletBC(W.sub(5), Constant((0.0, 0.0)), DomainBoundary())            # FIXME: Correct BC?
        bc_P_M2 = DirichletBC(W.sub(5), Constant((0.0, 0.0)), interior_facet_domains, 1)

        # Collect bcs
        bcs = [bc_U_F0, bc_U_F1, bc_P_F0, bc_P_F1, bc_P_F2, bc_U_S, bc_P_S, bc_U_M1, bc_U_M2, bc_P_M1, bc_P_M2]

        # Time-stepping
        p = Progress("Time-stepping")
        for t in t_range:

            print "*******************************************"
            print "-------------------------------------------"
            print "Solving the DUAL problem at t =", str(T - t)
            print "-------------------------------------------"
            print "*******************************************"

            # Get primal data
            _get_primal_data(t)

            # Assemble
            dual_matrix = assemble(A, cell_domains = cell_domains, interior_facet_domains = interior_facet_domains, exterior_facet_domains = exterior_boundary)
            dual_vector = assemble(L, cell_domains = cell_domains, interior_facet_domains = interior_facet_domains, exterior_facet_domains = exterior_boundary)

            # Apply bcs
            for bc in bcs:
                bc.apply(dual_matrix, dual_vector)

            # Remove inactive dofs
            dual_matrix.ident_zeros()

            # Compute dual solution
            solve(dual_matrix, Z.vector(), dual_vector)

            # Copy solution from previous interval
            # Dual varibles
            Z_F0.assign(Z_F)
            Y_F0.assign(Y_F)
            Z_S0.assign(Z_S)
            Y_S0.assign(Y_S)
            Z_M0.assign(Z_M)
            #Y_M0.assign(Y_M)

            # Primal varibles
            U_M0.assign(U_M)
            U_F0.assign(U_F)

            # Plot solutions
            if plot_solution:
                plot(Z_F, title="Dual fluid velocity")
                plot(Y_F, title="Dual fluid pressure")
                plot(Z_S, title="Dual displacement")
                plot(Y_S, title="Dual structure velocity")
                plot(Z_M, title="Dual mesh displacement")
                plot(Y_M, title="Dual mesh Lagrange Multiplier")

            # Store dual bin files. NOTE: we save dual solutions at t!
            if store_bin_files:
                dual_Z.store(Z.vector(), t)

            # Store vtu files
            if store_vtu_files:
                file_Z_F << Z_F
                file_Y_F << Y_F
                file_Z_S << Z_S
                file_Y_S << Y_S
                file_Z_M << Z_M
                file_Y_M << Y_M

            # Move to next time interval
            t += kn

# Retrieve primal data
def _get_primal_data(t):

    # Initialize edges on fluid sub mesh (needed for P2 elements)
    Omega_F.init(1)

    # Define number of vertices and edges
    Nv = Omega.num_vertices()
    Nv_F = Omega_F.num_vertices()
    Ne_F = Omega_F.num_edges()

    # Create vectors for primal dofs
    u_F_subdofs = Vector()
    p_F_subdofs = Vector()
    U_S_subdofs = Vector()
    P_S_subdofs = Vector()
    U_M_subdofs = Vector()

    # Get primal data (note the "dual time shift")
    primal_u_F.retrieve(u_F_subdofs, T - t)
    primal_p_F.retrieve(p_F_subdofs, T - t)
    primal_U_S.retrieve(U_S_subdofs, T - t)
    primal_P_S.retrieve(P_S_subdofs, T - t)
    primal_U_M.retrieve(U_M_subdofs, T - t)

    # Create mapping from Omega_(F,S,M) to Omega (extend primal vectors with zeros)
    global_vertex_indices_F = Omega_F.data().mesh_function("global vertex indices")
    global_vertex_indices_S = Omega_S.data().mesh_function("global vertex indices")
    global_vertex_indices_M = Omega_F.data().mesh_function("global vertex indices")

    # Create lists
    F_global_index = zeros([global_vertex_indices_F.size()], "uint")
    M_global_index = zeros([global_vertex_indices_M.size()], "uint")
    S_global_index = zeros([global_vertex_indices_S.size()], "uint")

    # Extract global vertices
    for j in range(global_vertex_indices_F.size()):
        F_global_index[j] = global_vertex_indices_F[j]
    for j in range(global_vertex_indices_M.size()):
        M_global_index[j] = global_vertex_indices_M[j]
    for j in range(global_vertex_indices_S.size()):
        S_global_index[j] = global_vertex_indices_S[j]

    # Get global dofs
    U_F_global_dofs = append(F_global_index, F_global_index + Nv)
    P_F_global_dofs = F_global_index
    U_S_global_dofs = append(S_global_index, S_global_index + Nv)
    P_S_global_dofs = append(S_global_index, S_global_index + Nv)
    U_M_global_dofs = append(M_global_index, M_global_index + Nv)

    # Get rid of P2 dofs for u_F and create a P1 function
    u_F_subdofs = append(u_F_subdofs[:Nv_F], u_F_subdofs[Nv_F + Ne_F: 2*Nv_F + Ne_F])

    # Transfer the stored primal solutions on the dual mesh Omega
    U_F.vector()[U_F_global_dofs] = u_F_subdofs
    P_F.vector()[P_F_global_dofs] = p_F_subdofs
    U_S.vector()[U_S_global_dofs] = U_S_subdofs
    P_S.vector()[U_S_global_dofs] = P_S_subdofs
    U_M.vector()[U_M_global_dofs] = U_M_subdofs

    return U_F, P_F, U_S, P_S, U_M
