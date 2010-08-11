"This module implements the dual FSI solver."

__author__ = "Kristoffer Selim and Anders Logg"
__copyright__ = "Copyright (C) 2010 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

# Last changed: 2010-08-11

from dolfin import *

def solve_dual(problem, solver_parameters):
    "Solve the dual FSI problem"

    # Get problem parameters
    T = problem.end_time()

    # Get solver parameters
    plot_solution = solver_parameters["plot_solution"]
    save_solution = solver_parameters["save_solution"]
    save_series = solver_parameters["save_series"]

    # Create files for saving to VTK
    if save_solution:
        Z_F_file = File("pvd/Z_F.pvd")
        Y_F_file = File("pvd/Y_F.pvd")
        Z_S_file = File("pvd/Z_S.pvd")
        Y_S_file = File("pvd/Y_S.pvd")
        Z_M_file = File("pvd/Z_M.pvd")
        Y_M_file = File("pvd/Y_M.pvd")

    # Create time series for storing solution
    if save_series:
        Z_F_series = TimeSeries("bin/Z_F")
        Y_F_series = TimeSeries("bin/Y_F")
        Z_S_series = TimeSeries("bin/Z_S")
        Y_S_series = TimeSeries("bin/Y_S")
        Z_M_series = TimeSeries("bin/Z_M")
        Y_M_series = TimeSeries("bin/Y_M")

    # Define function spaces defined on the whole domain
    V_F1 = VectorFunctionSpace(Omega, "CG", 1)
    V_F2 = VectorFunctionSpace(Omega, "CG", 2)
    Q_F  = FunctionSpace(Omega, "CG", 1)
    V_S  = VectorFunctionSpace(Omega, "CG", 1)
    Q_S  = VectorFunctionSpace(Omega, "CG", 1)
    V_M  = VectorFunctionSpace(Omega, "CG", 1)
    Q_M  = VectorFunctionSpace(Omega, "CG", 1)

    # Create mixed function space
    mixed_space = (V_F2, Q_F, V_S, Q_S, V_M, Q_M)
    W = MixedFunctionSpace(mixed_space)

    # Create test functions
    (v_F, q_F, v_S, q_S, v_M, q_M) = TestFunctions(W)

    # Define trial functions
    (Z_UF, Z_PF, Z_US, Z_PS, Z_UM, Z_PM) = TrialFunctions(W)

    # Create primal functions on Omega
    U_F = Function(V_F1)
    P_F = Function(Q_F)
    U_S = Function(V_S)
    P_S = Function(Q_S)
    U_M = Function(V_M)
    P_M = Function(Q_M) #FIXME: Check for sure that Im not needed in the primal!

    # Create functions for time-stepping/initial conditions
    # Dual varibles
    Z_UF0 = Function(V_F2)
    Z_PF0 = Function(Q_F)
    Z_US0 = Function(V_S)
    Z_PS0 = Function(Q_S)
    Z_UM0 = Function(V_M)
    Z_PM0 = Function(Q_M)

    # Primal varibles
    U_M0  = Function(V_M)
    U_F0  = Function(V_F1)
    U_S0  = Function(V_S)
    P_S0  = Function(Q_S)

    # Fix time step if needed. Note that this has to be done
    # in oder to retrieve primal data at the correct time
    dt, t_range = timestep_range(T, dt) #FIXME: Only the default values in common.py are set!!!
    kn = Constant(dt)

    # Define normals
    N_S =  FacetNormal(Omega_S)
    N =  N_S('+')
    N_F = FacetNormal(Omega_F)

    # Fluid eq. linearized around fluid variables
    A_FF01 = -(1/kn)*inner((Z_UF0 - Z_UF), rho_F*J(U_M)*v_F)*dx(0)
    A_FF02 =  inner(Z_UF, rho_F*J(U_M)*dot(dot(grad(v_F),F_inv(U_M)), (U_F - (U_M0 - U_M)*(1/kn))))*dx(0)
    A_FF03 =  inner(Z_UF, rho_F*J(U_M)*dot(grad(U_F) , dot(F_inv(U_M), v_F)))*dx(0)
    A_FF04 =  inner(grad(Z_UF), J(U_M)*mu_F*dot(grad(v_F) , dot(F_inv(U_M), F_invT(U_M))))*dx(0)
    A_FF05 =  inner(grad(Z_UF), J(U_M)*mu_F*dot(F_invT(U_M) , dot(grad(v_F).T, F_invT(U_M))))*dx(0)
    A_FF06 = -inner(grad(Z_UF), J(U_M)*q_F*F_invT(U_M))*dx(0)
    A_FF07 =  inner(Z_PF, div(J(U_M)*dot(F_inv(U_M),v_F)))*dx(0)

    # Boundary terms (Neumann condition G_N_F, dS(2) = in, dS(3) = out)
    # Note that we assume an inifinte long channel -> grad(U_F) = 0
    G_FF_in_1  = -inner(Z_US('+'), dot(J(U_M)('+')*mu_F*dot(F_invT(U_M)('+') , dot(grad(v_F('+')).T, F_invT(U_M)('+'))), N_F('+')))*dS(2)
    G_FF_in_2  =  inner(Z_US('+'), dot(J(U_M)('+')*q_F('+')*F_invT(U_M)('+'), N_F('+')))*dS(2)
    G_FF_out_1 = -inner(Z_US('+'), dot(J(U_M)('+')*mu_F*dot(F_invT(U_M)('+') , dot(grad(v_F('+')).T, F_invT(U_M)('+'))), N_F('+')))*dS(3)
    G_FF_out_2 =  inner(Z_US('+'), dot(J(U_M)('+')*q_F('+')*F_invT(U_M)('+'), N_F('+')))*dS(3)

    # Collect boundary terms
    G_FF = G_FF_in_1 + G_FF_in_2 + G_FF_out_1 + G_FF_out_2

    # Collect A_FF form
    A_FF = A_FF01 + A_FF02 + A_FF03 + A_FF04 + A_FF05 + A_FF06 + A_FF07 + G_FF

    # Fluid eq. linearized around mesh variable
    A_FM01 =  (1/kn)*inner(Z_UF, rho_F*DJ(U_M, v_M)*(U_F0 - U_F))*dx(0)
    A_FM02 =  inner(Z_UF, rho_F*DJ(U_M, v_M)*dot(grad(U_F), dot(F_inv(U_M), (U_M - U_M0)*(1/kn))))*dx(0)
    A_FM03 = -inner(Z_UF,  rho_F*J(U_M)*dot((dot(grad(U_F), dot(F_inv(U_M), dot(grad(v_M),F_inv(U_M))))),(U_F - (U_M0 - U_M)/kn)))*dx(0)
    A_FM04 =  (1/kn)*inner((Z_UF0 - Z_UF), rho_F*J(U_M)*dot(grad(U_F), dot(F_inv(U_M) ,v_M )))*dx(0)
    A_FM05 =  inner(grad(Z_UF), DJ(U_M, v_M)*dot(Sigma_F(U_F, P_F, U_M),F_invT(U_M)))*dx(0)
    A_FM06 = -inner(grad(Z_UF), J(U_M)*dot(mu_F*(dot(grad(U_F), dot(F_inv(U_M), dot(grad(v_M), F_inv(U_M))))), F_invT(U_M)))*dx(0)
    A_FM07 = -inner(grad(Z_UF), J(U_M)*dot(mu_F*(dot(F_invT(U_M), dot(grad(v_M).T, dot(F_invT(U_M), grad(U_F).T )))), F_invT(U_M)))*dx(0)
    A_FM08 = -inner(grad(Z_UF), J(U_M)*dot(mu_F*(dot(grad(U_F), dot(F_inv(U_M), dot(F_invT(U_M), grad(v_M).T )))), F_invT(U_M)))*dx(0)
    A_FM09 = -inner(grad(Z_UF), J(U_M)*dot(mu_F*(dot(F_invT(U_M), dot(grad(U_F).T, dot(F_invT(U_M), grad(v_M).T )))), F_invT(U_M)))*dx(0)
    A_FM10 =  inner(grad(Z_UF), J(U_M)*dot(dot( P_F*I,F_invT(U_M)) ,  dot(grad(v_M).T ,F_invT(U_M) )))*dx(0)
    A_FM11 =  inner(Z_PF, div(DJ(U_M,v_M)*dot(F_inv(U_M), U_F)))*dx(0)
    A_FM12 = -inner(Z_PF, div(J(U_M)*dot(dot(F_inv(U_M),grad(v_M)), dot(F_inv(U_M) ,U_F))))*dx(0)

    # Boundary terms (Neumann conditions G_N_F, dS(2) = in, dS(3) = out)
    # Note that we assume an inifinte long channel -> grad(U_F) = 0
    G_FM_in_1 = -inner(Z_UF('+'), DJ(U_M, v_M)('+')*mu_F*dot(dot(F_invT(U_M)('+'),grad(U_F('+')).T), dot(F_invT(U_M)('+'), N_F('+'))))*dS(2)
    G_FM_in_2 =  inner(Z_UF('+'), DJ(U_M, v_M)('+')*dot(P_F('+')*I('+'), N_F('+')))*dS(2)
    G_FM_in_3 =  inner(Z_UF('+'), J(U_M)('+')*mu_F*dot(dot(F_invT(U_M)('+'), dot(grad(v_M('+')).T, F_invT(U_M)('+'))), dot(grad(U_F('+')).T, dot(F_invT(U_M)('+'), N_F('+') ))))*dS(2)
    G_FM_in_4 =  inner(Z_UF('+'), J(U_M)('+')*mu_F*dot(dot(F_invT(U_M)('+'), dot(grad(U_F('+')).T, F_invT(U_M)('+'))), dot(grad(v_M('+')).T , dot(F_invT(U_M)('+'),N_F('+')))))*dS(2)
    G_FM_out_1 = -inner(Z_UF('+'), DJ(U_M, v_M)('+')*mu_F*dot(dot(F_invT(U_M)('+'),grad(U_F('+')).T), dot(F_invT(U_M)('+'), N_F('+'))))*dS(3)
    G_FM_out_2 =  inner(Z_UF('+'), DJ(U_M, v_M)('+')*dot(P_F('+')*I('+'), N_F('+')))*dS(3)
    G_FM_out_3 =  inner(Z_UF('+'), J(U_M)('+')*mu_F*dot(dot(F_invT(U_M)('+'), dot(grad(v_M('+')).T, F_invT(U_M)('+'))), dot(grad(U_F('+')).T, dot(F_invT(U_M)('+'), N_F('+') ))))*dS(3)
    G_FM_out_4 =  inner(Z_UF('+'), J(U_M)('+')*mu_F*dot(dot(F_invT(U_M)('+'), dot(grad(U_F('+')).T, F_invT(U_M)('+'))), dot(grad(v_M('+')).T , dot(F_invT(U_M)('+'),N_F('+')))))*dS(3)

    # Collect boundary terms
    G_FM = G_FM_in_1 + G_FM_in_2 + G_FM_in_3 + G_FM_in_4 + G_FM_out_1 + G_FM_out_2 + G_FM_out_3 + G_FM_out_4

    # Collect A_FM form
    A_FM =  A_FM01 + A_FM02 + A_FM03 + A_FM04 + A_FM05 + A_FM06 + A_FM07 + A_FM08 + A_FM09 + A_FM10 + A_FM11 + A_FM12 + G_FM

    # Structure eq. linearized around the fluid variables
    A_SF01 = -inner(Z_US('+'), mu_F*J(U_M)('+')*dot(dot(grad(v_F('+')), F_inv(U_M)('+')), dot(F_invT(U_M)('+'), N)))*dS(1)
    A_SF02 = -inner(Z_US('+'), mu_F*J(U_M)('+')*dot(dot(F_invT(U_M)('+'), grad(v_F('+')).T), dot(F_invT(U_M)('+'), N)))*dS(1)
    A_SF03 =  inner(Z_US('+'), mu_F*J(U_M)('+')*q_F('+')*dot(I('+'), dot(F_invT(U_M)('+'), N)))*dS(1)

    # Collect A_SF form
    A_SF = A_SF01 + A_SF02 + A_SF03

    # Operators for A_SS
    Fu = F(U_S)
    Eu = Fu*Fu.T - I
    Ev = grad(v_S)*Fu.T + Fu*grad(v_S).T
    Sv = grad(v_S)*(2*mu_S*Eu + lmbda_S*tr(Eu)*I) + Fu*(2*mu_S*Ev + lmbda_S*tr(Ev)*I)

    # Structure eq. linearized around structure variable
    A_SS = - (1/kn)*inner(Z_US0 - Z_US, rho_S*q_S)*dx(1) + inner(grad(Z_US), Sv)*dx(1) \
        - (1/kn)*inner(Z_PS0 - Z_PS, v_S)*dx(1) - inner(Z_PS, q_S)*dx(1)

    # Structure eq. linearized around mesh variable
    A_SM01 = -inner(Z_US('+'), DJ(U_M,v_M)('+')*mu_F*dot(dot(grad(U_F('+')), F_inv(U_F)('+')), dot(F_invT(U_M)('+'), N)))*dS(1) # FIXME: Replace with Sigma_F
    A_SM02 = -inner(Z_US('+'), DJ(U_M,v_M)('+')*mu_F*dot(dot(F_invT(U_F)('+'), grad(U_F('+')).T), dot(F_invT(U_M)('+'), N)))*dS(1)# FIXME: Replace with Sigma_F
    A_SM03 =  inner(Z_US('+'), DJ(U_M,v_M)('+')*dot(P_F('+')*I('+'), dot(F_invT(U_M)('+'),N)))*dS(1)# FIXME: Replace with Sigma_F
    A_SM04 =  inner(Z_US('+'), J(U_M)('+')*mu_F*dot(dot(grad(U_F('+')), dot(F_inv(U_M)('+'),grad(v_M('+')))), dot(F_inv(U_M)('+'), dot(F_invT(U_M)('+'), N))))*dS(1)
    A_SM05 =  inner(Z_US('+'), J(U_M)('+')*mu_F*dot(dot(grad(U_F('+')).T, dot(F_invT(U_M)('+'), grad(v_M('+')).T)), dot(F_invT(U_M)('+'), dot(F_invT(U_M)('+'),N))))*dS(1)
    A_SM06 =  inner(Z_US('+'), J(U_M)('+')*mu_F*dot(dot(grad(U_F('+')),F_inv(U_M)('+')),dot(F_invT(U_M)('+'), dot(grad(v_M('+')).T, dot(F_invT(U_M)('+'),N)))))*dS(1)
    A_SM07 =  inner(Z_US('+'), J(U_M)('+')*mu_F*dot(dot(F_invT(U_M)('+'),grad(U_M('+')).T),dot(F_invT(U_M)('+'), dot(grad(v_M('+')).T, dot(F_invT(U_M)('+'),N)))))*dS(1)
    A_SM08 = -inner(Z_US('+'), J(U_M)('+')*dot(dot(P_F('+')*I('+'),F_invT(U_M)('+')), dot(grad(v_M('+')).T, dot(F_invT(U_M)('+'), N))))*dS(1)

    # Collect A_SM form
    A_SM = A_SM01 + A_SM02 + A_SM03 + A_SM04 + A_SM05 + A_SM06 + A_SM07 + A_SM08

    # Mesh eq. linearized around mesh variable
    A_MM01 = -(1/kn)*inner(v_M, Z_UM0 - Z_UM)*dx(0) + inner(sym_gradient(Z_UM), Sigma_M(v_M))*dx(0)
    A_MM02 = inner(Z_UM('+'),v_M('+'))*dS(1)
    A_MM03 = inner(Z_PM('+'),q_M('+'))*dS(1)

    # Collect A_MM form
    A_MM = A_MM01 + A_MM02 + A_MM03

    # Mesh eq. linearized around structure variable
    A_MS = - inner(Z_PM('+'), q_S('+'))*dS(1)

    # Define goal funtionals
    n_F = FacetNormal(Omega_F)
    #goal_F = 0.03*inner(v_F('+'), N)*ds(1)
    area = 0.2*0.5
    goal_functional = (1/T)*(1.0/area)*v_S[0]*dx(1)

    # Define the dual rhs and lhs
    A_system = A_FF + A_FM + A_SS + A_SF + A_SM + A_MM + A_MS
    A = lhs(A_system)
    L = rhs(A_system)  + goal_functional

    # Define BCs (define the dual trial space = homo. Dirichlet BCs)
    bc_U_F0  = DirichletBC(W.sub(0), Constant((0.0, 0.0)), noslip)
    bc_U_F1  = DirichletBC(W.sub(0), Constant((0.0, 0.0)), interior_facet_domains, 1 )
    bc_P_F0  = DirichletBC(W.sub(1), Constant(0.0), inflow)
    bc_P_F1  = DirichletBC(W.sub(1), Constant(0.0), outflow)
    bc_P_F2  = DirichletBC(W.sub(1), Constant(0.0), interior_facet_domains, 1 )
    bc_U_S   = DirichletBC(W.sub(2), Constant((0.0, 0.0)), dirichlet_boundaries)
    bc_P_S   = DirichletBC(W.sub(3), Constant((0.0, 0.0)), dirichlet_boundaries)
    bc_U_M1  = DirichletBC(W.sub(4), Constant((0.0, 0.0)), DomainBoundary())
    bc_U_M2  = DirichletBC(W.sub(4), Constant((0.0, 0.0)), interior_facet_domains, 1)
    bc_P_M1  = DirichletBC(W.sub(5), Constant((0.0, 0.0)), DomainBoundary())            # FIXME: Correct BC?
    bc_P_M2  = DirichletBC(W.sub(5), Constant((0.0, 0.0)), interior_facet_domains, 1)

    # Collect bcs
    bcs = [bc_U_F0, bc_U_F1, bc_P_F0, bc_P_F1, bc_P_F2, bc_U_S, bc_P_S, bc_U_M1, bc_U_M2, bc_P_M1, bc_P_M2]

    # Create solution functions
    Z = Function(W)
    (Z_UF, Z_PF, Z_US, Z_PS, Z_UM, Z_PM) = Z.split()

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
        Z_UF0.assign(Z_UF)
        Z_PF0.assign(Z_PF)
        Z_US0.assign(Z_US)
        Z_PS0.assign(Z_PS)
        Z_UM0.assign(Z_UM)
        #Z_PM0.assign(Z_PM)

        # Primal varibles
        U_M0.assign(U_M)
        U_F0.assign(U_F)

        # Plot solutions
        if plot_solution:
            plot(Z_UF, title="Dual fluid velocity")
            plot(Z_PF, title="Dual fluid pressure")
            plot(Z_US, title="Dual displacement")
            plot(Z_PS, title="Dual structure velocity")
            plot(Z_UM, title="Dual mesh displacement")
            plot(Z_PM, title="Dual mesh Lagrange Multiplier")

        # Store dual bin files. NOTE: we save dual solutions at t!
        if store_bin_files:
            dual_Z.store(Z.vector(), t)

        # Store vtu files
        if store_vtu_files:
            file_Z_UF << Z_UF
            file_Z_PF << Z_PF
            file_Z_US << Z_US
            file_Z_PS << Z_PS
            file_Z_UM << Z_UM
            file_Z_PM << Z_PM

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
