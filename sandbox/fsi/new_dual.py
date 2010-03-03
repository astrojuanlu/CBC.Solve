from common import *
from cbc.common import CBCSolver
from cbc.twist.kinematics import SecondOrderIdentity
from numpy import array, append, zeros
from dualoperators import *

# Define function spaces defined on the whole domain
V_F1 = VectorFunctionSpace(Omega, "CG", 1)
V_F2 = VectorFunctionSpace(Omega, "CG", 2)
Q_F  = FunctionSpace(Omega, "CG", 1)
V_S  = VectorFunctionSpace(Omega, "CG", 1)
V_M  = VectorFunctionSpace(Omega, "CG", 1) 
Q_M = VectorFunctionSpace(Omega, "CG", 1) 

# Create mixed function space
mixed_space = (V_F2, Q_F, V_S, V_M, Q_M)
W = MixedFunctionSpace(mixed_space)

# Define test functions
(v_F, q_F, v_S, v_M, q_M) = TestFunctions(W)

# Define trial functions
(Z_UF, Z_PF, Z_US, Z_UM, Z_PM) = TrialFunctions(W)

# Create primal functions on Omega
U_F = Function(V_F1)
P_F = Function(Q_F)
U_S = Function(V_S)
U_M = Function(V_M)
#P_M = Function(Q_M) FIXME: Check for sure that Im not needed!

# Create initial conditions FIXME: Add "real" initial conditions
Z_UF0 = Function(V_F2)
Z_US0 = Function(V_S)
U_M0 = Function(V_M)

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

    # Get primal data 
    primal_u_F.retrieve(u_F_subdofs, t)
    primal_p_F.retrieve(p_F_subdofs, t)
    primal_U_M.retrieve(U_M_subdofs, t)
    primal_U_S.retrieve(U_S_subdofs, t)

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

    return U_F, P_F, U_S, U_M

# Initialize first primal data
get_primal_data(0)

# Fluid eq. linearized around fluid variables
A_FF01 = -inner((Z_UF - Z_UF0), rho_F*v_F)*dx(0)                                   #time-dependent                             
A_FF02 =  inner(Z_UF, dot(dot(grad(v_F),F_inv(U_M)), (U_F - (U_M - U_M0))))*dx(0)  #time-dependent
A_FF03 =  kn*inner(Z_UF, dot(grad(U_F) , dot(F(U_M), v_F)))*dx(0)
A_FF04 =  kn*inner(grad(Z_UF), mu_F*dot(grad(v_F) , dot(F_inv(U_M), F_invT(U_M))))*dx(0)
A_FF05 =  kn*inner(grad(Z_UF), mu_F*dot(F_invT(U_M) , dot(grad(v_F).T, F_invT(U_M))))*dx(0)
#A_FF06 = -kn**inner(grad(Z_UF), (q_F*dot(I(U_M), F_invT(U_M))))*dx # FIXME: Which one should we use?
A_FF06 = -kn*inner(grad(Z_UF), q_F*F_invT(U_M))*dx(0)
A_FF07 =  kn*inner(Z_PF, inner(F_invT(U_M), grad(v_F)))*dx(0) 

# Collect A_FF form
A_FF_sum = A_FF01 + A_FF03 + A_FF04 + A_FF05 + A_FF06 + A_FF07

# Fluid eq. linearized around mesh variable
A_FM01 =  inner(Z_UF, dot((dot(grad(U_F), dot(F_inv(U_M), dot(grad(v_M),F_inv(U_M))))),(U_F - (U_M - U_M0))))*dx(0) #time-dependent
A_FM02 =  inner((U_M - U_M0), dot(grad(U_F), dot(F_inv(U_M),v_M )))*dx(0)                                           #time-dependent
A_FM03 = -kn*inner(grad(Z_UF), dot(mu_F*(dot(grad(U_F), dot(F_inv(U_M), dot(grad(v_M),F_inv(U_M))))),F_invT(U_M)))*dx(0)
A_FM04 = -kn*inner(grad(Z_UF), dot(mu_F*(dot(F_invT(U_M), dot(grad(v_M).T, dot(F_invT(U_M), grad(U_F).T )))),F_invT(U_M)))*dx(0)
A_FM05 = -kn*inner(grad(Z_UF), dot(mu_F*(dot(grad(U_F), dot(F_inv(U_M), dot(F_invT(U_M), grad(v_M).T )))),F_invT(U_M)))*dx(0)
A_FM06 = -kn*inner(grad(Z_UF), dot(mu_F*(dot(F_invT(U_M) , dot(grad(U_M).T, dot(F_invT(U_M), grad(v_M).T )))),F_invT(U_M)))*dx(0)
A_FM07 =  kn*inner(grad(Z_UF), dot(dot( P_F*I(U_F),F_invT(U_M)) ,  dot(grad(v_M).T ,F_invT(U_M) )))*dx(0)
A_FM08 = -kn*inner(Z_PF, inner( dot(F_invT(U_M), grad(v_M).T)  , dot( I(U_M), grad(U_F))))*dx(0) # FIXME: Is this the right way to write it????

# Collect A_FM form
A_FM_sum =  A_FM01 + A_FM02 + A_FM03 + A_FM04 + A_FM05 + A_FM06 + A_FM07 + A_FM08

# Structure eq. linearized around the fluid variables
A_SF01 = -inner(Z_US('+'), mu_F*J(U_M)('+')*dot(dot(grad(v_M('+')), F_inv(U_M)('+')), dot(F_invT(U_M)('+'), N)))*dS(1)
A_SF02 = -inner(Z_US('+'), mu_F*J(U_M)('+')*dot(dot(F_invT(U_M)('+'), grad(v_M('+')).T), dot(F_invT(U_M)('+'), N)))*dS(1)
A_SF03 =  inner(Z_US('+'), mu_F*J(U_M)('+')*q_F('+')*dot(I(U_M)('+'), dot(F_invT(U_M)('+'), N)))*dS(1)

# Collect A_SF form
A_SF_sum = A_SF01 + A_SF02 + A_SF03 

# Structure eq. linearized around the structure variable
A_SS01 = 0 #FIXME: Add and talk to Harish! 
A_SS02 = inner(grad(Z_US), mu_S*dot(grad(v_S), F_T(U_S)))*dx(1)
A_SS03 = inner(grad(Z_US), mu_S*dot(F(U_S), grad(v_S).T))*dx(1)
A_SS04 = inner(grad(Z_US), 0.5*lamb_S*tr(dot( grad(v_S), F_T(U_S)))*I(U_S))*dx(1)
A_SS05 = inner(grad(Z_US), 0.5*lamb_S*tr(dot(F(U_S), grad(v_S)))*I(U_S))*dx(1)

# Collect A_SS form
A_SS_sum = A_SS02 + A_SS02 + A_SS04 + A_SS05

# Structure eq. linearized around mesh variable
A_SM01 = -inner(Z_US('+'), DJ(U_M,v_M)('+')*mu_F*dot(dot(grad(U_F('+')), F_inv(U_F)('+')), dot(F_invT(U_M)('+'), N)))*dS(1)
A_SM02 = -inner(Z_US('+'), DJ(U_M,v_M)('+')*mu_F*dot(dot(F_invT(U_F)('+'), grad(U_F('+')).T), dot(F_invT(U_M)('+'), N)))*dS(1)
A_SM03 =  inner(Z_US('+'), DJ(U_M,v_M)('+')*dot(P_F('+')*I(U_F)('+'), dot(F_invT(U_M)('+'),N)))*dS(1)
A_SM04 =  inner(Z_US('+'), J(U_M)('+')*mu_F*dot(dot(grad(U_F('+')), dot(F_inv(U_M)('+'),grad(v_M('+')))), dot(F_inv(U_M)('+'), dot(F_invT(U_M)('+'), N))))*dS(1) 
A_SM05 =  inner(Z_US('+'), J(U_M)('+')*mu_F*dot(dot(grad(U_F('+')).T, dot(F_invT(U_M)('+'), grad(v_M('+')).T)), dot(F_invT(U_M)('+'), dot(F_invT(U_M)('+'),N))))*dS(1)
A_SM06 =  inner(Z_US('+'), J(U_M)('+')*mu_F*dot(dot(grad(U_F('+')),F_inv(U_M)('+')),dot(F_invT(U_M)('+'), dot(grad(v_M('+')).T, dot(F_invT(U_M)('+'),N)))))*dS(1)
A_SM07 =  inner(Z_US('+'), J(U_M)('+')*mu_F*dot(dot(F_invT(U_M)('+'),grad(U_M('+')).T),dot(F_invT(U_M)('+'), dot(grad(v_M('+')).T, dot(F_invT(U_M)('+'),N)))))*dS(1)
A_SM08 = -inner(Z_US('+'), J(U_M)('+')*dot(dot(P_F('+')*I(U_F)('+'),F_invT(U_M)('+')), dot(grad(v_M('+')).T, dot(F_invT(U_M)('+'), N))))*dS(1)

# Collect A_SM form
A_SM_sum = A_SM01 + A_SM02 + A_SM03 + A_SM04 + A_SM05 + A_SM06 + A_SM07 + A_SM08 

# Mesh eq. linearized around mesh variable
A_MM_sum = inner(sym_gradient(Z_UM), sigma_M(v_M))*dx(0) + inner(Z_UM('+'),q_M('+'))*dS(1) + inner(Z_PM('+'),v_M('+'))*dS(1)

# Mesh eq. linearized around strucure variable
A_MS_sum = - inner(Z_PM('+'), v_S('+'))*dS(1)

# Collect the dual matrix
A_dual = A_FF_sum + A_SF_sum + A_SS_sum + A_FM_sum + A_SM_sum + A_MM_sum + A_MS_sum

# Define goal functionals
goal_F = #v_F[0]*ds(0)  #inner(v_F('+'), dot(grad(U_F('+')), N))*dS(1) + inner(v_F('+'), dot(grad(U_F('+')).T, N))*dS(1) - P_F('+')*inner(v_F('+'), dot(I(v_F)('+'), N))*dS(1)
#goal_M = - v_S[0]*dx(1)
goal_F = inner(v_F('+'), dot(grad(U_F('+')), N))*dS(1) + inner(v_F('+'), dot(grad(U_F('+')).T, N))*dS(1) - P_F('+')*inner(v_F('+'), dot(I(v_F)('+'), N))*dS(1)
L_dual = goal_F 
   
# Define BCs
bcuF = DirichletBC(W.sub(0), Constant((0,0)), noslip)
bcp0 = DirichletBC(W.sub(1), Constant(0.0), inflow)
bcp1 = DirichletBC(W.sub(1), Constant(0.0), outflow)
bcuS = DirichletBC(W.sub(2), Constant((0,0)), dirichlet_boundaries)
bcuM1 = DirichletBC(W.sub(3), Constant((0,0)), DomainBoundary())
bcuM2 = DirichletBC(W.sub(3), Constant((0,0)), interior_facet_domains, 1)
bcuPM1 = DirichletBC(W.sub(4), Constant((0,0)), DomainBoundary())
bcuPM2 = DirichletBC(W.sub(4), Constant((0,0)), interior_facet_domains, 1)

# Collect BCs
bcs = [bcuF, bcp0, bcp1, bcuS, bcuM1, bcuM2, bcuPM1, bcuPM2]

# Create files 
file_Z_UF = File("Z_UF.pvd")
file_Z_PF = File("Z_PF.pvd")
file_Z_S = File("Z_S.pvd")
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

   # Fix inactive dofs
   dual_matrix.ident_zeros()
   
   # Compute dual solution
   Z = Function(W)
   solve(dual_matrix, Z.vector(), dual_vector)
   (Z_UF, Z_PF, Z_S, Z_UM, Z_PM) = Z.split()

   # Copy solution from previous interval
   Z_UF0.assign(Z_UF)
   U_M0.assign(U_M)
   #Z_US0.assing(Z_US)
   
   # Save solutions
   file_Z_UF << Z_UF
   file_Z_PF << Z_PF
   file_Z_S << Z_S
   file_Z_UM << Z_UM
   file_Z_PM << Z_PM

   # Move to next time interval
   t += kn
      
# # Plot solutions
# plot(Z_UF, title="Dual velocity")
# plot(Z_PF, title="Dual pressure")
# plot(Z_S,  title="Dual displacement")
# plot(Z_UM,  title="Dual mesh displacement")
# #plot(Z_PM,  title="Dual mesh Lagrange Multiplier")
# #interactive()































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
