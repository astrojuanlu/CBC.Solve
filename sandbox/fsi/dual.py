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

# Dual varibles
Z_UF0 = Function(V_F2)
Z_PF0 = Function(Q_F)
Z_US0 = Function(V_S)
Z_PS0 = Function(Q_S)
Z_UM0 = Function(V_M)
Z_PM0 = Function(Q_M)

# Primal varibles
U_M0  = Function(V_M) # FIXME: should be taken care of when retrieve the primal data
U_F0  = Function(V_F1)                      # or no?t
U_S0  = Function(V_S)

# Define time evaluation (cG1 or dG0)
cG1 = False

if cG1 == True: # FIXME: doesn't work
    # Define cG(1) evaluation of non-time derivatives (mid-point)
    Z_UF_ip = 0.5*(Z_UF + Z_UF0)
    Z_PF_ip = 0.5*(Z_PF + Z_PF0)
    Z_US_ip = 0.5*(Z_US + Z_US0)
    Z_PS_ip = 0.5*(Z_PS + Z_PS0)
    Z_UM_ip = 0.5*(Z_UM + Z_UM0) # FIXME: should be taken care of when retrieve the primal data
    Z_PM_ip = 0.5*(Z_PM + Z_PM0) # or not?
#     # Stupid fix for the cG(1) "FSI-boundary" forms FIXME: Can this be done in another way?
#     Z_UM_cg1 = 0.5*(Z_UM('+') + Z_UM0('+'))
#     Z_US_cg1 = 0.5*(Z_US('+') + Z_US0('+'))
#     Z_PM_cgi = 0.5*(Z_PM('+') + Z_PM0('+'))
    
else:
    # Define dG(0) evaluation of non-time derivatives
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
    # if we use cG(1)
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
A_FF02 =  inner(Z_UF_ip, J(U_M)*rho_F*dot(dot(grad(v_F),F_inv(U_M)), (U_F - (U_M - U_M0)/kn)))*dx(0) # FIXME: Check time-derivative on U_M.
A_FF03 =  inner(Z_UF_ip, J(U_M)*dot(grad(U_F) , dot(F_inv(U_M), v_F)))*dx(0)
A_FF04 =  inner(grad(Z_UF_ip), J(U_M)*mu_F*dot(grad(v_F) , dot(F_inv(U_M), F_invT(U_M))))*dx(0)
A_FF05 =  inner(grad(Z_UF_ip), J(U_M)*mu_F*dot(F_invT(U_M) , dot(grad(v_F).T, F_invT(U_M))))*dx(0)
A_FF06 = -inner(grad(Z_UF_ip), J(U_M)*q_F*F_invT(U_M))*dx(0)
A_FF07 =  inner(Z_PF_ip, div(J(U_M)*dot(F_inv(U_M),v_F)))*dx(0)

# Collect A_FF form
A_FF = A_FF01 + A_FF02 + A_FF03 + A_FF04 + A_FF05 + A_FF06 + A_FF07

# Fluid eq. linearized around mesh variable
A_FM01 =  (1/kn)*inner(Z_UF_ip, rho_F*DJ(U_M, v_M)*(U_F - U_F0))*dx(0)
A_FM02 =  (1/kn)*inner(Z_UF_ip, rho_F*DJ(U_M, v_M)*dot(grad(U_F), dot(F_inv(U_M), (U_M - U_M0))))*dx(0) 
A_FM03 = -inner(Z_UF,  rho_F*J(U_M)*dot((dot(grad(U_F), dot(F_inv(U_M), dot(grad(v_M),F_inv(U_M))))),(U_F - (U_M - U_M0)/kn)))*dx(0) 
A_FM04 =  (1/kn)*inner((Z_UF - Z_UF0), rho_F*J(U_M)*dot(grad(U_F), dot(F_inv(U_M) ,v_M )))*dx(0)
A_FM05 =  inner(grad(Z_UF_ip), DJ(U_M, v_M)*dot(Sigma_F(U_F, P_F, U_M),F_invT(U_M)))*dx(0)
A_FM06 = -inner(grad(Z_UF_ip), J(U_M)*dot(mu_F*(dot(grad(U_F), dot(F_inv(U_M), dot(grad(v_M).T, F_inv(U_M))))), F_invT(U_M)))*dx(0)
A_FM07 = -inner(grad(Z_UF_ip), J(U_M)*dot(mu_F*(dot(F_invT(U_M), dot(grad(v_M).T, dot(F_invT(U_M), grad(U_F).T )))), F_invT(U_M)))*dx(0)
A_FM08 = -inner(grad(Z_UF_ip), J(U_M)*dot(mu_F*(dot(grad(U_F), dot(F_inv(U_M), dot(F_invT(U_M), grad(v_M).T )))), F_invT(U_M)))*dx(0)
A_FM09 = -inner(grad(Z_UF_ip), J(U_M)*dot(mu_F*(dot(F_invT(U_M), dot(grad(U_F).T, dot(F_invT(U_M), grad(v_M).T )))), F_invT(U_M)))*dx(0)
A_FM10 =  inner(grad(Z_UF_ip), J(U_M)*dot(dot( P_F*I(U_F),F_invT(U_M)) ,  dot(grad(v_M).T ,F_invT(U_M) )))*dx(0)
A_FM11 =  inner(Z_PF_ip, div(DJ(U_M,v_M)*dot(F_inv(U_M), U_F)))*dx(0)
A_FM12 = -inner(Z_PF_ip, div(J(U_M)*dot(dot(F_inv(U_M),grad(v_M)), dot(F_inv(U_M) ,U_F))))*dx(0)

# Collect A_FM form
A_FM =  A_FM01 + A_FM02 + A_FM03 + A_FM04 + A_FM05 + A_FM06 + A_FM07 + A_FM08 + A_FM09 + A_FM10 + A_FM11 + A_FM12

# Structure eq. linearized around the fluid variables
A_SF01 = -inner(Z_US_ip('+'), mu_F*J(U_M)('+')*dot(dot(grad(v_M('+')), F_inv(U_M)('+')), dot(F_invT(U_M)('+'), N)))*dS(1)
A_SF02 = -inner(Z_US_ip('+'), mu_F*J(U_M)('+')*dot(dot(F_invT(U_M)('+'), grad(v_M('+')).T), dot(F_invT(U_M)('+'), N)))*dS(1)
A_SF03 =  inner(Z_US_ip('+'), mu_F*J(U_M)('+')*q_F('+')*dot(I(U_M)('+'), dot(F_invT(U_M)('+'), N)))*dS(1)

# Collect A_SF form
A_SF = A_SF01 + A_SF02 + A_SF03

# Structure eq. linearized around the structure variable
# Note that we solve the srtucture as a first order system in time
# FIXME: for this to be consistent, we need to change the PRIMAL solver as well
# FIXME: Add A_SS08 term (or A_SS09...)
A_SS01 = -(1/kn)*inner((Z_PS - Z_PS0), rho_S*v_S)*dx(1) 
A_SS02 =  inner(grad(Z_US_ip), mu_S*dot(grad(v_S), dot(F_T(U_S), F(U_S)) - I(U_S)))*dx(1)
A_SS03 =  inner(grad(Z_US_ip), mu_S*dot(F(U_S), dot(grad(v_S).T, F(U_S)) - I(U_S)))*dx(1)
A_SS04 =  inner(grad(Z_US_ip), mu_S*dot(F(U_S), dot(F_T(U_S), grad(v_S)) - I(U_S)))*dx(1)
A_SS05 =  inner(grad(Z_US_ip), 0.5*lamb_S*dot(grad(v_S), tr(dot(F(U_S),F_T(U_S)))*I(U_S)))*dx(1)
A_SS06 =  inner(grad(Z_US_ip), 0.5*lamb_S*dot(F(U_S), tr(dot(grad(v_S),F_T(U_S)))*I(U_S)))*dx(1)
A_SS07 =  inner(grad(Z_US_ip), 0.5*lamb_S*dot(F(U_S), tr(dot(F(U_S), grad(v_S).T))*I(U_S)))*dx(1)
A_SS08 =  inner(Z_PS_ip, q_S)*dx(1)  

# Collect A_SS form
A_SS =  A_SS02 + A_SS02 + A_SS04 + A_SS05 + A_SS06 + A_SS07 + A_SS08 

# Structure eq. linearized around mesh variable
A_SM01 = -inner(Z_US_ip('+'), DJ(U_M,v_M)('+')*mu_F*dot(dot(grad(U_F('+')), F_inv(U_F)('+')), dot(F_invT(U_M)('+'), N)))*dS(1) # FIXME: Replace with Sigma_F
A_SM02 = -inner(Z_US_ip('+'), DJ(U_M,v_M)('+')*mu_F*dot(dot(F_invT(U_F)('+'), grad(U_F('+')).T), dot(F_invT(U_M)('+'), N)))*dS(1)# FIXME: Replace with Sigma_F
A_SM03 =  inner(Z_US_ip('+'), DJ(U_M,v_M)('+')*dot(P_F('+')*I(U_F)('+'), dot(F_invT(U_M)('+'),N)))*dS(1)# FIXME: Replace with Sigma_F
A_SM04 =  inner(Z_US_ip('+'), J(U_M)('+')*mu_F*dot(dot(grad(U_F('+')), dot(F_inv(U_M)('+'),grad(v_M('+')))), dot(F_inv(U_M)('+'), dot(F_invT(U_M)('+'), N))))*dS(1) 
A_SM05 =  inner(Z_US_ip('+'), J(U_M)('+')*mu_F*dot(dot(grad(U_F('+')).T, dot(F_invT(U_M)('+'), grad(v_M('+')).T)), dot(F_invT(U_M)('+'), dot(F_invT(U_M)('+'),N))))*dS(1)
A_SM06 =  inner(Z_US_ip('+'), J(U_M)('+')*mu_F*dot(dot(grad(U_F('+')),F_inv(U_M)('+')),dot(F_invT(U_M)('+'), dot(grad(v_M('+')).T, dot(F_invT(U_M)('+'),N)))))*dS(1)
A_SM07 =  inner(Z_US_ip('+'), J(U_M)('+')*mu_F*dot(dot(F_invT(U_M)('+'),grad(U_M('+')).T),dot(F_invT(U_M)('+'), dot(grad(v_M('+')).T, dot(F_invT(U_M)('+'),N)))))*dS(1)
A_SM08 = -inner(Z_US_ip('+'), J(U_M)('+')*dot(dot(P_F('+')*I(U_F)('+'),F_invT(U_M)('+')), dot(grad(v_M('+')).T, dot(F_invT(U_M)('+'), N))))*dS(1)

# Collect A_SM form
A_SM = A_SM01 + A_SM02 + A_SM03 + A_SM04 + A_SM05 + A_SM06 + A_SM07 + A_SM08

# Mesh eq. linearized around mesh variable
A_MM01 = inner(sym_gradient(Z_UM_ip), sigma_M(v_M))*dx(0)
A_MM02 = inner(Z_UM_ip('+'),q_M('+'))*dS(1) 
A_MM03 = inner(Z_PM_ip('+'),v_M('+'))*dS(1)

# Collect A_MM form
A_MM = A_MM01 + A_MM02 + A_MM03

# Mesh eq. linearized around strucure variable
A_MS = - inner(Z_PM_ip('+'), v_S('+'))*dS(1)

# Define goal funtionals
psi_S_t = Constant((1.0, 0.0))
goal_S = (1/T)*inner(v_S, psi_S_t)*dx(1)
n_F = FacetNormal(Omega_F)
goal_F = inner(v_F, n_F)*ds(2)
goal_functionals =  goal_S #+ goal_F

# Define the dual rhs and lhs
A_dual = lhs(A_FF + A_FM + A_SS + A_SF + A_SM + A_MM + A_MS)
L = rhs(A_FF + A_FM + A_SS + A_SF + A_SM + A_MM + A_MS)
L_dual = L + goal_functionals

# Define BCs (i.e. define the dual trial space = homo. Dirichlet BCs)
bc_U_F   = DirichletBC(W.sub(0), Constant((0,0)), noslip) 
bc_P_F0  = DirichletBC(W.sub(1), Constant(0.0), inflow) # FIME: Make sure it goes to zero in both P/D
bc_P_F1  = DirichletBC(W.sub(1), Constant(0.0), outflow)# FIME: Make sure it goes to zero in both P/D
bc_U_S   = DirichletBC(W.sub(2), Constant((0,0)), dirichlet_boundaries)
bc_P_S   = DirichletBC(W.sub(3), Constant((0,0)), DomainBoundary())            # FIXME: Correct BC?
bc_U_M1  = DirichletBC(W.sub(4), Constant((0,0)), DomainBoundary())
bc_U_M2  = DirichletBC(W.sub(4), Constant((0,0)), interior_facet_domains, 1)
bc_U_PM1 = DirichletBC(W.sub(5), Constant((0,0)), DomainBoundary())            # FIXME: Correct BC?
bc_U_PM2 = DirichletBC(W.sub(5), Constant((0,0)), interior_facet_domains, 1)   # FIXME: Correct BC? 

# Define dual initial conditions (i.e. Z_T = <v, psi_T> etc.
bc_ZF_T = DirichletBC(W.sub(0), Constant((DOLFIN_EPS, 0.0)), outflow)

# Collect bcs
bcs = [bc_U_F, bc_P_F0, bc_P_F1, bc_U_S, bc_P_S, bc_U_M1, bc_U_M2, bc_U_PM1, bc_U_PM2, bc_ZF_T]

# Create files 
file_Z_UF = File("Z_UF.pvd")
file_Z_PF = File("Z_PF.pvd")
file_Z_US = File("Z_US.pvd")
file_Z_UM = File("Z_UM.pvd")
file_Z_PM = File("Z_PM.pvd")

# Create solution functions
Z = Function(W)
(Z_UF, Z_PF, Z_US, Z_PS, Z_UM, Z_PM) = Z.split()

# Time stepping
while t < T:
    
   print "*******************************************"
   print "-------------------------------------------"
   print "Solving the DUAL problem at t = ", str(t)
   print "-------------------------------------------"
   print "*******************************************"

   # Get primal data
   get_primal_data(t)

   # Assemble 
   dual_matrix = assemble(A_dual, cell_domains = cell_domains, interior_facet_domains = interior_facet_domains, exterior_facet_domains = exterior_boundary)
   dual_vector = assemble(L_dual, cell_domains = cell_domains, interior_facet_domains = interior_facet_domains, exterior_facet_domains = exterior_boundary)

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
   Z_PM0.assign(Z_PM)
   
   # Primal varibles
   U_M0.assign(U_M)  # FIXME: Should be done when we get the primal data 
   U_F0.assign(U_F)  # FIXME: Should be done when we get the primal data
   
   # Save solutions
   file_Z_UF << Z_UF
   file_Z_PF << Z_PF
   file_Z_US << Z_US
   file_Z_UM << Z_UM
   file_Z_PM << Z_PM

   # Plot solutions
  # plot(Z_PS, title="Dual structure velocity")
   plot(Z_PF, title="Dual pressure")
   plot(Z_UM, title="Dual mesh displacement")
   plot(Z_UF, title="Dual velocity")
   plot(Z_US, title="Dual displacement")
  # plot(Z_PM, title="Dual mesh Lagrange Multiplier")
  # interactive()
 
   # Move to next time interval
   t += kn
   # FIXME: Change to t =+ float(kn)   



















#goal_F = 0.01*(inner(v_F('+'), dot(grad(U_F('+')), N))*dS(1) + inner(v_F('+'), dot(grad(U_F('+')).T, N))*dS(1) - P_F('+')*inner(v_F('+'), dot(I(v_F)('+'), N))*dS(1))
