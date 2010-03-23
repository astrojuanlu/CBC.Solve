from common import *
from cbc.common import CBCSolver
from cbc.twist.kinematics import SecondOrderIdentity
from numpy import array, append, zeros
from dualoperators import *

# Optimize form compiler
parameters["form_compiler"]["cpp_optimize"] = True

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

# Define test functions
(v_F, q_F, v_S, q_S, v_M, q_M) = TestFunctions(W)

# Define trial functions
(Z_UF, Z_PF, Z_US, Z_PS, Z_UM, Z_PM) = TrialFunctions(W)

# Create primal functions on Omega
U_F = Function(V_F1)
P_F = Function(Q_F)
U_S = Function(V_S)
#P_S = Funtion()     FIXME: Need this from the primal solver that solves the sys. of eq.
U_M = Function(V_M)
#P_M = Function(Q_M) FIXME: Check for sure that Im not needed!
#P_S = Function(V_S) FIXME: Check for sure that Im not needed! 

# Create functions for time-stepping
Z_UF0 = Function(V_F2)
Z_PF0 = Function(Q_F)
Z_US0 = Function(V_S)
Z_PS0 = Function(Q_S)
Z_UM0 = Function(V_M)
Z_PM0 = Function(Q_M)
U_M0  = Function(V_M) # FIXME: should be taken care of when retrieve the primal data
U_F0  = Function(V_F1)                      # or no?t

# # Define cG(1) evaluation of non-time derivatives (mid-point)
# Z_UF_ip = 0.5*(Z_UF + Z_UF0)
# Z_PF_ip = 0.5*(Z_PF + Z_PF0)
# Z_US_ip = 0.5*(Z_US + Z_US0)
# Z_PS_ip = 0.5*(Z_PS + Z_PS0)
# Z_UM_ip = 0.5*(Z_UM + Z_UM0) # FIXME: should be taken care of when retrieve the primal data
# Z_PM_ip = 0.5*(Z_PM + Z_PM0) # or not?

# Define dG(0) (aka Backward Euler) evaluation of non-time derivatives
Z_UF_ip = Z_UF 
Z_PF_ip = Z_PF
Z_US_ip = Z_US
Z_PS_ip = Z_PS 
Z_UM_ip = Z_UM 
Z_PM_ip = Z_PM

# Define time-step
kn = dt

# Define FSI normal 
N_S =  FacetNormal(Omega_S)
N =  N_S('+')

# Retrieve primal data 
def get_primal_data(t):

   # Initialize edges on fluid sub mesh (needed for P2 elements)
    Omega_F.init(1)

    # Define number of vertices and edges
    Nv = Omega.num_vertices()
    Nv_F = Omega_F.num_vertices()
    Ne_F = Omega_F.num_edges()
    
    # Create vectors for primal dofs
    u_F_subdofs = Vector()
    p_F_subdofs = Vector()
    U_M_subdofs = Vector()
    U_S_subdofs = Vector()

    # Get primal data (note the "dual time shift")
    # FIXME: Should interpolate:  T - (t - kn/2)
    primal_u_F.retrieve(u_F_subdofs, T - t)
    primal_p_F.retrieve(p_F_subdofs, T - t)
    primal_U_M.retrieve(U_M_subdofs, T - t)
    primal_U_S.retrieve(U_S_subdofs, T - t)

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
    U_M_global_dofs = append(M_global_index, M_global_index + Nv)

    # Get rid of P2 dofs for u_F and create a P1 function
    u_F_subdofs = append(u_F_subdofs[:Nv_F], u_F_subdofs[Nv_F + Ne_F: 2*Nv_F + Ne_F])

    # Transfer the stored primal solutions on the dual mesh Omega
    U_F.vector()[U_F_global_dofs] = u_F_subdofs
    P_F.vector()[P_F_global_dofs] = p_F_subdofs
    U_S.vector()[U_S_global_dofs] = U_S_subdofs
    U_M.vector()[U_M_global_dofs] = U_M_subdofs

    # FIXME: Need primal data U(kn), U(kn-1), and U_ip (like for Z_UF_ip etc.)
    return U_F, P_F, U_S, U_M

# Fluid eq. linearized around fluid variables
A_FF01 = -(1/kn)*inner((Z_UF - Z_UF0), rho_F*J(U_M)*v_F)*dx(0)                           
A_FF02 =  (1/kn)*inner(Z_UF_ip, J(U_M)*rho_F*dot(dot(grad(v_F),F_inv(U_M)), (U_F - (U_M - U_M0))))*dx(0) # FIXME: Check time-derivative on U_M.
A_FF03 =  inner(Z_UF_ip, J(U_M)*dot(grad(U_F) , dot(F_inv(U_M), v_F)))*dx(0)
A_FF04 =  inner(grad(Z_UF_ip), J(U_M)*mu_F*dot(grad(v_F) , dot(F_inv(U_M), F_invT(U_M))))*dx(0)
A_FF05 =  inner(grad(Z_UF_ip), J(U_M)*mu_F*dot(F_invT(U_M) , dot(grad(v_F).T, F_invT(U_M))))*dx(0)
A_FF06 = -inner(grad(Z_UF_ip), J(U_M)*q_F*F_invT(U_M))*dx(0)
A_FF07 =  inner(Z_PF_ip, div(J(U_M)*dot(F_inv(U_M),v_F)))*dx(0)

# Collect A_FF form
A_FF_lhs = lhs(A_FF01 + A_FF03 + A_FF04 + A_FF05 + A_FF06 + A_FF07)
A_FF_rhs = rhs(A_FF01 + A_FF03 + A_FF04 + A_FF05 + A_FF06 + A_FF07)

# Fluid eq. linearized around mesh variable
A_FM01 =  (1/kn)*inner(Z_UF_ip, rho_F*DJ(U_M, v_M)*(U_F - U_F0))*dx(0)
A_FM02 =  (1/kn)*inner(Z_UF_ip, rho_F*DJ(U_M, v_M)*dot(grad(U_F), dot(F_inv(U_M), (U_M - U_M0))))*dx(0) 
A_FM03 = -(1/kn)*inner(Z_UF,  rho_F*J(U_M)*dot((dot(grad(U_F), dot(F_inv(U_M), dot(grad(v_M),F_inv(U_M))))),(U_F - (U_M - U_M0))))*dx(0) 
A_FM04 =  (1/kn)*inner((Z_UF - Z_UF0), rho_F*J(U_M)*dot(grad(U_F), dot(F_inv(U_M) ,v_M )))*dx(0)
A_FM05 =  inner(grad(Z_UF_ip), DJ(U_M, v_M)*dot(Sigma_F(U_F, P_F, U_M),F_invT(U_M)))*dx(0)
A_FM06 = -inner(grad(Z_UF_ip), J(U_M)*dot(mu_F*(dot(grad(U_F), dot(F_inv(U_M), dot(grad(v_M).T, F_inv(U_M))))), F_invT(U_M)))*dx(0)
A_FM07 = -inner(grad(Z_UF_ip), J(U_M)*dot(mu_F*(dot(F_invT(U_M), dot(grad(v_M).T, dot(F_invT(U_M), grad(U_F).T )))), F_invT(U_M)))*dx(0)
A_FM08 = -inner(grad(Z_UF_ip), J(U_M)*dot(mu_F*(dot(grad(U_F), dot(F_inv(U_M), dot(F_invT(U_M), grad(v_M).T )))), F_invT(U_M)))*dx(0)
A_FM09 = -inner(grad(Z_UF_ip), J(U_M)*dot(mu_F*(dot(F_invT(U_M), dot(grad(U_F).T, dot(F_invT(U_M), grad(v_M).T )))), F_invT(U_M)))*dx(0)
A_FM10 =  inner(grad(Z_UF_ip), J(U_M)*dot(dot( P_F*I(U_F),F_invT(U_M)) ,  dot(grad(v_M).T ,F_invT(U_M) )))*dx(0)
A_FM11 =  inner(Z_PF_ip, div(DJ(U_M,q_M)*dot(F_inv(U_M), U_F)))*dx(0)
A_FM12 = -inner(Z_PF_ip, div(J(U_M)*dot(dot(F_inv(U_M),grad(q_M)), dot(F_inv(U_M) ,U_F))))*dx(0)

#A_FM11 = -inner(Z_PF_ip, inner( dot(F_invT(U_M), grad(v_M).T)  , dot( I(U_M), grad(U_F))))*dx(0) # FIXME: Is this the right way to write it????

# Collect A_FM form
A_FM_lhs =  lhs(A_FM01 + A_FM02 + A_FM03 + A_FM04 + A_FM05 + A_FM06 + A_FM07 + A_FM08 + A_FM09 + A_FM10 + A_FM11 + A_FM12)
A_FM_rhs =  rhs(A_FM01 + A_FM02 + A_FM03 + A_FM04 + A_FM05 + A_FM06 + A_FM07 + A_FM08 + A_FM09 + A_FM10 + A_FM11 + A_FM12)

# # Structure eq. linearized around the fluid variables
# # LHS
# A_SF01_lhs = -0.5*inner(Z_US('+'), mu_F*J(U_M)('+')*dot(dot(grad(v_M('+')), F_inv(U_M)('+')), dot(F_invT(U_M)('+'), N)))*dS(1)
# A_SF02_lhs = -0.5*inner(Z_US('+'), mu_F*J(U_M)('+')*dot(dot(F_invT(U_M)('+'), grad(v_M('+')).T), dot(F_invT(U_M)('+'), N)))*dS(1)
# A_SF03_lhs =  0.5*inner(Z_US('+'), mu_F*J(U_M)('+')*q_F('+')*dot(I(U_M)('+'), dot(F_invT(U_M)('+'), N)))*dS(1)
# # RHS
# A_SF01_rhs =  0.5*inner(Z_US0('+'), mu_F*J(U_M)('+')*dot(dot(grad(v_M('+')), F_inv(U_M)('+')), dot(F_invT(U_M)('+'), N)))*dS(1)
# A_SF02_rhs =  0.5*inner(Z_US0('+'), mu_F*J(U_M)('+')*dot(dot(F_invT(U_M)('+'), grad(v_M('+')).T), dot(F_invT(U_M)('+'), N)))*dS(1)
# A_SF03_rhs = -0.5*inner(Z_US0('+'), mu_F*J(U_M)('+')*q_F('+')*dot(I(U_M)('+'), dot(F_invT(U_M)('+'), N)))*dS(1)

# # Collect A_SF form
# A_SF_lhs = A_SF01_lhs + A_SF02_lhs + A_SF03_lhs
# A_SF_rhs = A_SF01_rhs + A_SF02_rhs + A_SF03_rhs

# FIXME: Can't use lhs/rhs to work with forms containing restrictions
A_SF01_lhs = -inner(Z_US('+'), mu_F*J(U_M)('+')*dot(dot(grad(v_M('+')), F_inv(U_M)('+')), dot(F_invT(U_M)('+'), N)))*dS(1)
A_SF02_lhs = -inner(Z_US('+'), mu_F*J(U_M)('+')*dot(dot(F_invT(U_M)('+'), grad(v_M('+')).T), dot(F_invT(U_M)('+'), N)))*dS(1)
A_SF03_lhs =  inner(Z_US('+'), mu_F*J(U_M)('+')*q_F('+')*dot(I(U_M)('+'), dot(F_invT(U_M)('+'), N)))*dS(1)
# RHS
A_SF01_rhs =  0.0*inner(Z_US0('+'), mu_F*J(U_M)('+')*dot(dot(grad(v_M('+')), F_inv(U_M)('+')), dot(F_invT(U_M)('+'), N)))*dS(1)
A_SF02_rhs =  0.0*inner(Z_US0('+'), mu_F*J(U_M)('+')*dot(dot(F_invT(U_M)('+'), grad(v_M('+')).T), dot(F_invT(U_M)('+'), N)))*dS(1)
A_SF03_rhs = -0.0*inner(Z_US0('+'), mu_F*J(U_M)('+')*q_F('+')*dot(I(U_M)('+'), dot(F_invT(U_M)('+'), N)))*dS(1)

# Collect A_SF form
A_SF_lhs = A_SF01_lhs + A_SF02_lhs + A_SF03_lhs
A_SF_rhs = A_SF01_rhs + A_SF02_rhs + A_SF03_rhs

# Structure eq. linearized around the structure variable
# Note that we solve the srtucture as a first order system in time
# FIXME: for this to be consistent, we need to change the PRIMAL solver as well
# FIXME: Add A_SS08 term
A_SS01 = -(1/kn)*inner((Z_PS - Z_PS0), rho_S*v_S)*dx(1) 
A_SS02 =  inner(grad(Z_US_ip), mu_S*dot(grad(v_S), dot(F_T(U_S), F(U_S)) - I(U_S)))*dx(1)
A_SS03 =  inner(grad(Z_US_ip), mu_S*dot(F(U_S), dot(grad(v_S).T, F(U_S)) - I(U_S)))*dx(1)
A_SS04 =  inner(grad(Z_US_ip), mu_S*dot(F(U_S), dot(F_T(U_S), grad(v_S)) - I(U_S)))*dx(1)
A_SS05 =  inner(grad(Z_US_ip), 0.5*lamb_S*dot(grad(v_S), tr(dot(F(U_S),F_T(U_S)))*I(U_S)))*dx(1)
A_SS06 =  inner(grad(Z_US_ip), 0.5*lamb_S*dot(F(U_S), tr(dot(grad(v_S),F_T(U_S)))*I(U_S)))*dx(1)
A_SS07 =  inner(grad(Z_US_ip), 0.5*lamb_S*dot(F(U_S), tr(dot(F(U_S), grad(v_S).T))*I(U_S)))*dx(1)
# A_SS08 = ...

# Collect A_SS form
A_SS_lhs =  lhs(A_SS02 + A_SS02 + A_SS04 + A_SS05 + A_SS06 + A_SS07)
A_SS_rhs =  rhs(A_SS02 + A_SS02 + A_SS04 + A_SS05 + A_SS06 + A_SS07)

# # Structure eq. linearized around mesh variable
# # LHS
# A_SM01_lhs = -0.5*inner(Z_US('+'), DJ(U_M,v_M)('+')*mu_F*dot(dot(grad(U_F('+')), F_inv(U_F)('+')), dot(F_invT(U_M)('+'), N)))*dS(1) # FIXME: Replace with Sigma_F
# A_SM02_lhs = -0.5*inner(Z_US('+'), DJ(U_M,v_M)('+')*mu_F*dot(dot(F_invT(U_F)('+'), grad(U_F('+')).T), dot(F_invT(U_M)('+'), N)))*dS(1)# FIXME: Replace with Sigma_F
# A_SM03_lhs =  0.5*inner(Z_US('+'), DJ(U_M,v_M)('+')*dot(P_F('+')*I(U_F)('+'), dot(F_invT(U_M)('+'),N)))*dS(1)# FIXME: Replace with Sigma_F
# A_SM04_lhs =  0.5*inner(Z_US('+'), J(U_M)('+')*mu_F*dot(dot(grad(U_F('+')), dot(F_inv(U_M)('+'),grad(v_M('+')))), dot(F_inv(U_M)('+'), dot(F_invT(U_M)('+'), N))))*dS(1) 
# A_SM05_lhs =  0.5*inner(Z_US('+'), J(U_M)('+')*mu_F*dot(dot(grad(U_F('+')).T, dot(F_invT(U_M)('+'), grad(v_M('+')).T)), dot(F_invT(U_M)('+'), dot(F_invT(U_M)('+'),N))))*dS(1)
# A_SM06_lhs =  0.5*inner(Z_US('+'), J(U_M)('+')*mu_F*dot(dot(grad(U_F('+')),F_inv(U_M)('+')),dot(F_invT(U_M)('+'), dot(grad(v_M('+')).T, dot(F_invT(U_M)('+'),N)))))*dS(1)
# A_SM07_lhs =  0.5*inner(Z_US('+'), J(U_M)('+')*mu_F*dot(dot(F_invT(U_M)('+'),grad(U_M('+')).T),dot(F_invT(U_M)('+'), dot(grad(v_M('+')).T, dot(F_invT(U_M)('+'),N)))))*dS(1)
# A_SM08_lhs = -0.5*inner(Z_US('+'), J(U_M)('+')*dot(dot(P_F('+')*I(U_F)('+'),F_invT(U_M)('+')), dot(grad(v_M('+')).T, dot(F_invT(U_M)('+'), N))))*dS(1)
# # RHS
# A_SM01_rhs =  0.5*inner(Z_US0('+'), DJ(U_M,v_M)('+')*mu_F*dot(dot(grad(U_F('+')), F_inv(U_F)('+')), dot(F_invT(U_M)('+'), N)))*dS(1)# FIXME: Replace with Sigma_F
# A_SM02_rhs =  0.5*inner(Z_US0('+'), DJ(U_M,v_M)('+')*mu_F*dot(dot(F_invT(U_F)('+'), grad(U_F('+')).T), dot(F_invT(U_M)('+'), N)))*dS(1)# FIXME: Replace with Sigma_F
# A_SM03_rhs = -0.5*inner(Z_US0('+'), DJ(U_M,v_M)('+')*dot(P_F('+')*I(U_F)('+'), dot(F_invT(U_M)('+'),N)))*dS(1)# FIXME: Replace with Sigma_F
# A_SM04_rhs = -0.5*inner(Z_US0('+'), J(U_M)('+')*mu_F*dot(dot(grad(U_F('+')), dot(F_inv(U_M)('+'),grad(v_M('+')))), dot(F_inv(U_M)('+'), dot(F_invT(U_M)('+'), N))))*dS(1) 
# A_SM05_rhs = -0.5*inner(Z_US0('+'), J(U_M)('+')*mu_F*dot(dot(grad(U_F('+')).T, dot(F_invT(U_M)('+'), grad(v_M('+')).T)), dot(F_invT(U_M)('+'), dot(F_invT(U_M)('+'),N))))*dS(1)
# A_SM06_rhs = -0.5*inner(Z_US0('+'), J(U_M)('+')*mu_F*dot(dot(grad(U_F('+')),F_inv(U_M)('+')),dot(F_invT(U_M)('+'), dot(grad(v_M('+')).T, dot(F_invT(U_M)('+'),N)))))*dS(1)
# A_SM07_rhs = -0.5*inner(Z_US0('+'), J(U_M)('+')*mu_F*dot(dot(F_invT(U_M)('+'),grad(U_M('+')).T),dot(F_invT(U_M)('+'), dot(grad(v_M('+')).T, dot(F_invT(U_M)('+'),N)))))*dS(1)
# A_SM08_rhs =  0.5*inner(Z_US0('+'), J(U_M)('+')*dot(dot(P_F('+')*I(U_F)('+'),F_invT(U_M)('+')), dot(grad(v_M('+')).T, dot(F_invT(U_M)('+'), N))))*dS(1)

# # Collect A_SM form
# A_SM_lhs = A_SM01_lhs + A_SM02_lhs + A_SM03_lhs + A_SM04_lhs + A_SM05_lhs + A_SM06_lhs + A_SM07_lhs + A_SM08_lhs
# A_SM_rhs = A_SM01_rhs + A_SM02_rhs + A_SM03_rhs + A_SM04_rhs + A_SM05_rhs + A_SM06_rhs + A_SM07_rhs + A_SM08_rhs

# Structure eq. linearized around mesh variable
# LHS
# FIXME: Can't use lhs/rhs to work with forms containing restrictions
A_SM01_lhs = -inner(Z_US('+'), DJ(U_M,v_M)('+')*mu_F*dot(dot(grad(U_F('+')), F_inv(U_F)('+')), dot(F_invT(U_M)('+'), N)))*dS(1) # FIXME: Replace with Sigma_F
A_SM02_lhs = -inner(Z_US('+'), DJ(U_M,v_M)('+')*mu_F*dot(dot(F_invT(U_F)('+'), grad(U_F('+')).T), dot(F_invT(U_M)('+'), N)))*dS(1)# FIXME: Replace with Sigma_F
A_SM03_lhs =  inner(Z_US('+'), DJ(U_M,v_M)('+')*dot(P_F('+')*I(U_F)('+'), dot(F_invT(U_M)('+'),N)))*dS(1)# FIXME: Replace with Sigma_F
A_SM04_lhs =  inner(Z_US('+'), J(U_M)('+')*mu_F*dot(dot(grad(U_F('+')), dot(F_inv(U_M)('+'),grad(v_M('+')))), dot(F_inv(U_M)('+'), dot(F_invT(U_M)('+'), N))))*dS(1) 
A_SM05_lhs =  inner(Z_US('+'), J(U_M)('+')*mu_F*dot(dot(grad(U_F('+')).T, dot(F_invT(U_M)('+'), grad(v_M('+')).T)), dot(F_invT(U_M)('+'), dot(F_invT(U_M)('+'),N))))*dS(1)
A_SM06_lhs =  inner(Z_US('+'), J(U_M)('+')*mu_F*dot(dot(grad(U_F('+')),F_inv(U_M)('+')),dot(F_invT(U_M)('+'), dot(grad(v_M('+')).T, dot(F_invT(U_M)('+'),N)))))*dS(1)
A_SM07_lhs =  inner(Z_US('+'), J(U_M)('+')*mu_F*dot(dot(F_invT(U_M)('+'),grad(U_M('+')).T),dot(F_invT(U_M)('+'), dot(grad(v_M('+')).T, dot(F_invT(U_M)('+'),N)))))*dS(1)
A_SM08_lhs = -inner(Z_US('+'), J(U_M)('+')*dot(dot(P_F('+')*I(U_F)('+'),F_invT(U_M)('+')), dot(grad(v_M('+')).T, dot(F_invT(U_M)('+'), N))))*dS(1)
# RHS
A_SM01_rhs =  0.0*inner(Z_US0('+'), DJ(U_M,v_M)('+')*mu_F*dot(dot(grad(U_F('+')), F_inv(U_F)('+')), dot(F_invT(U_M)('+'), N)))*dS(1)# FIXME: Replace with Sigma_F
A_SM02_rhs =  0.0*inner(Z_US0('+'), DJ(U_M,v_M)('+')*mu_F*dot(dot(F_invT(U_F)('+'), grad(U_F('+')).T), dot(F_invT(U_M)('+'), N)))*dS(1)# FIXME: Replace with Sigma_F
A_SM03_rhs = -0.0*inner(Z_US0('+'), DJ(U_M,v_M)('+')*dot(P_F('+')*I(U_F)('+'), dot(F_invT(U_M)('+'),N)))*dS(1)# FIXME: Replace with Sigma_F
A_SM04_rhs = -0.0*inner(Z_US0('+'), J(U_M)('+')*mu_F*dot(dot(grad(U_F('+')), dot(F_inv(U_M)('+'),grad(v_M('+')))), dot(F_inv(U_M)('+'), dot(F_invT(U_M)('+'), N))))*dS(1) 
A_SM05_rhs = -0.0*inner(Z_US0('+'), J(U_M)('+')*mu_F*dot(dot(grad(U_F('+')).T, dot(F_invT(U_M)('+'), grad(v_M('+')).T)), dot(F_invT(U_M)('+'), dot(F_invT(U_M)('+'),N))))*dS(1)
A_SM06_rhs = -0.0*inner(Z_US0('+'), J(U_M)('+')*mu_F*dot(dot(grad(U_F('+')),F_inv(U_M)('+')),dot(F_invT(U_M)('+'), dot(grad(v_M('+')).T, dot(F_invT(U_M)('+'),N)))))*dS(1)
A_SM07_rhs = -0.0*inner(Z_US0('+'), J(U_M)('+')*mu_F*dot(dot(F_invT(U_M)('+'),grad(U_M('+')).T),dot(F_invT(U_M)('+'), dot(grad(v_M('+')).T, dot(F_invT(U_M)('+'),N)))))*dS(1)
A_SM08_rhs =  0.0*inner(Z_US0('+'), J(U_M)('+')*dot(dot(P_F('+')*I(U_F)('+'),F_invT(U_M)('+')), dot(grad(v_M('+')).T, dot(F_invT(U_M)('+'), N))))*dS(1)

# Collect A_SM form
A_SM_lhs = A_SM01_lhs + A_SM02_lhs + A_SM03_lhs + A_SM04_lhs + A_SM05_lhs + A_SM06_lhs + A_SM07_lhs + A_SM08_lhs
A_SM_rhs = A_SM01_rhs + A_SM02_rhs + A_SM03_rhs + A_SM04_rhs + A_SM05_rhs + A_SM06_rhs + A_SM07_rhs + A_SM08_rhs


# Mesh eq. linearized around mesh variable
# LHS
A_MM01_lhs = 0.5*inner(sym_gradient(Z_UM), sigma_M(v_M))*dx(0)
A_MM02_lhs = 0.5*inner(Z_UM('+'),q_M('+'))*dS(1) 
A_MM03_lhs = 0.5*inner(Z_PM('+'),v_M('+'))*dS(1)
# RHS
A_MM01_rhs = -0.5*inner(sym_gradient(Z_UM0), sigma_M(v_M))*dx(0)
A_MM02_rhs = -0.5*inner(Z_UM0('+'),q_M('+'))*dS(1) 
A_MM03_rhs = -0.5*inner(Z_PM0('+'),v_M('+'))*dS(1)

# Collect A_MM form
A_MM_lhs = A_MM01_lhs + A_MM02_lhs + A_MM02_lhs
A_MM_rhs = A_MM01_rhs + A_MM02_rhs + A_MM02_rhs

# Mesh eq. linearized around strucure variable
A_MS_lhs = - 0.5*inner(Z_PM('+'), v_S('+'))*dS(1)
A_MS_rhs =   0.5*inner(Z_PM0('+'), v_S('+'))*dS(1)

# Define goal funtionals
goal_F = inner(v_F('+'), dot(grad(U_F('+')), N))*dS(1) + inner(v_F('+'), dot(grad(U_F('+')).T, N))*dS(1) - P_F('+')*inner(v_F('+'), dot(I(v_F)('+'), N))*dS(1)
jada = Constant((1.0, 0.0))
goal_F = inner(grad(v_F), grad(U_F))*dx(0)
goal_S = inner(v_S, U_S)*dx(1)
GOAL = goal_F #+ goal_S

# Define the dual rhs and lhs
A_dual = A_FF_lhs + A_FM_lhs + A_SS_lhs + A_SF_lhs + A_SM_lhs + A_MM_lhs + A_MS_lhs 
L_dual = A_FF_rhs + A_FM_rhs + A_SS_rhs + A_SF_rhs + A_SM_rhs + A_MM_rhs + A_MS_rhs + GOAL

# Define BCs
bc_U_F   = DirichletBC(W.sub(0), Constant((0,0)), noslip)
bc_P_F0  = DirichletBC(W.sub(1), Constant(0.0), inflow)
bc_P_F1  = DirichletBC(W.sub(1), Constant(0.0), outflow)
bc_U_S   = DirichletBC(W.sub(2), Constant((0,0)), dirichlet_boundaries)
bc_P_S   = DirichletBC(W.sub(3), Constant((0,0)), dirichlet_boundaries) # FIXME: Correct BC?
bc_U_M1  = DirichletBC(W.sub(4), Constant((0,0)), DomainBoundary())
bc_U_M2  = DirichletBC(W.sub(4), Constant((0,0)), interior_facet_domains, 1)
bc_U_PM1 = DirichletBC(W.sub(5), Constant((0,0)), DomainBoundary())
bc_U_PM2 = DirichletBC(W.sub(5), Constant((0,0)), interior_facet_domains, 1)

# Collect BCs
bcs = [bc_U_F, bc_P_F0, bc_P_F1, bc_U_S, bc_P_S, bc_U_M1, bc_U_M2, bc_U_PM1, bc_U_PM2]

# Create files 
file_Z_UF = File("Z_UF.pvd")
file_Z_PF = File("Z_PF.pvd")
file_Z_US = File("Z_US.pvd")
file_Z_UM = File("Z_UM.pvd")
file_Z_PM = File("Z_PM.pvd")

# Time stepping
while t < T :
    
   print "*******************************************"
   print "-------------------------------------------"
   print "Solving the DUAL problem at t = ", str(t)
   print "-------------------------------------------"
   print "*******************************************"

   # Get primal data
   get_primal_data(t)

   # Assemble 
   dual_matrix = assemble(A_dual, cell_domains = cell_domains, interior_facet_domains = interior_facet_domains)
   dual_vector = assemble(L_dual, cell_domains = cell_domains, interior_facet_domains = interior_facet_domains)

   # Apply bcs
   for bc in bcs:
      bc.apply(dual_matrix, dual_vector)

   # Remove inactive dofs
   dual_matrix.ident_zeros()
      
   # Compute dual solution
   Z = Function(W)
   solve(dual_matrix, Z.vector(), dual_vector)
   (Z_UF, Z_PF, Z_US, Z_PS, Z_UM, Z_PM) = Z.split()

   # Copy solution from previous interval
   Z_UF0.assign(Z_UF)
   Z_PF0.assign(Z_PF)
   Z_US0.assign(Z_US)
   Z_PS0.assign(Z_PS)
   Z_UM0.assign(Z_UM)
   Z_PM0.assign(Z_PM)
   U_M0.assign(U_M)  # FIXME: Shoul be done when we get the primal data
   
   # Save solutions
   file_Z_UF << Z_UF
   file_Z_PF << Z_PF
   file_Z_US << Z_US
   file_Z_UM << Z_UM
   file_Z_PM << Z_PM

   # Plot solutions
   plot(Z_UF, title="Dual velocity")
   #plot(Z_PF, title="Dual pressure")
   plot(Z_US,  title="Dual displacement")
   plot(Z_UM,  title="Dual mesh displacement")
   #plot(Z_PM,  title="Dual mesh Lagrange Multiplier")
#   interactive()

   # Move to next time interval
   t += kn
      






















# Define goal functionals
#goal_F = inner(v_F('+'), dot(grad(U_F('+')), N))*dS(1) + inner(v_F('+'), dot(grad(U_F('+')).T, N))*dS(1) - P_F('+')*inner(v_F('+'), dot(I(v_F)('+'), N))*dS(1)
#MS_hat = Function((1.0, 0.0))
#goal_S = #MS_hat*v_S*dx(1)
#goal_F = inner(v_F('+'), dot(grad(U_F('+')), N))*dS(1) + inner(v_F('+'), dot(grad(U_F('+')).T, N))*dS(1) - P_F('+')*inner(v_F('+'), dot(I(v_F)('+'), N))*dS(1)
# goal_M = inner(grad(U_M), grad(v_M))*dx(1)
# L_dual = goal_M 


# Define goal functionals
# # Define epsilon
# def epsilon(v):
#     return 0.5*(grad(v) + (grad(v).T))
    
# # Define sigma
# def sigma(v,q):
#     return 2.0*mu_F*epsilon(v) - dot(q, I(v))

# n_prick_t= [[0, 0], [1, 0]]
# cut_off = Cutoff(V_F)

# goal_MF = cut_off*dot(sigma(v_F,q_F), n_prick_t)*dx

# F = Cutoff(V_F)
