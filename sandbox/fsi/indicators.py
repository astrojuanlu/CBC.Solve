" Computes the error indicators on the reference domain. Note that all error "
" indicators are evaluated on Omega since the dual solution lives on Omega "

from dolfin import *
from operators import *
from common import *
from numpy import append        


# Define function spaces for primal/dual 
V1 = VectorFunctionSpace(Omega, "CG", 1)
V2 = VectorFunctionSpace(Omega, "CG", 2)
V3 = VectorFunctionSpace(Omega, "CG", 3)
Q  = FunctionSpace(Omega, "CG", 1)
DG = FunctionSpace(Omega, "DG" , 0)
    
# Define a the mixed finite element space for dual varibles 
mixed_space = (V2, Q, V1, V1, V1, V1)
W = MixedFunctionSpace(mixed_space)

# Create dual mixed function
Z = Function(W)

# Create primal functions on Omega
U_F = Function(V1)
P_F = Function(Q)
U_S = Function(V1)
P_S = Function(V1)
U_M = Function(V1)

# Create primal functions for time-derivateves
U_F0 = Function(V1)
U_S0 = Function(V1)
P_S0 = Function(V1)
U_M0 = Function(V1)

# Create dual functions on Omega
Z_UF = Function(V2) 
Z_PF = Function(V1)
Z_US = Function(V1)
Z_PS = Function(V1)
Z_UM = Function(V1)
Z_PM = Function(V1)

# Create extrapolated dual functions
eZ_UF = Function(V3)
eZ_PF = Function(V2)
eZ_US = Function(V2)
eZ_PS = Function(V2)
eZ_UM = Function(V2)
eZ_PM = Function(V2)
        
# Create characteristic function on each triangle
chi = Function(DG)

# Create test functions for R_k
v = TestFunction(V1)
q = TestFunction(Q)

# Create facet normals
N   = FacetNormal(Omega)
N_F = FacetNormal(Omega_F)
N_S = FacetNormal(Omega_S)



# Retrieve dual data
def get_dual_data(t):
    " Retrieve dual solution and extrapolate"
    
    # Retrieve dual at time t
    # Note that the dual solution is stored in
    # "wrong" order in the dualsolver.py
    dual_Z.retrieve(Z.vector(), T - t)

    # Split the dual
    (Z_UF, Z_PF, Z_US, Z_PS, Z_UM, Z_PM) = Z.split()
        
    # Extrapolate 
    eZ_UF.extrapolate(Z_UF)      
    eZ_PF.extrapolate(Z_PF)
    eZ_US.extrapolate(Z_US)
    eZ_PS.extrapolate(Z_PS)
    eZ_UM.extrapolate(Z_UM)
    eZ_PM.extrapolate(Z_PM)

    return eZ_UF, eZ_PF, eZ_US, eZ_PS, eZ_UM, eZ_PM


# Retrieve primal data
def get_primal_data(t):
     " Retrieve primal solution and extend by zero"

     # Initialize edges on fluid sub mesh (needed for P2 elements)
     Omega_F.init(1)

     # Define number of vertices and edges
     Nv = Omega.num_vertices()
     Nv_F = Omega_F.num_vertices()
     Ne_F = Omega_F.num_edges()

     # Create vectors for primal dofs
     u_F_subdofs  = Vector()
     u_F0_subdofs = Vector()
     p_F_subdofs  = Vector()
     U_S_subdofs  = Vector()
     U_S0_subdofs = Vector()
     P_S_subdofs  = Vector()
     U_M_subdofs  = Vector()
     U_M0_subdofs = Vector()

     # Get primal data
     primal_u_F.retrieve(u_F_subdofs, t)
     primal_u_F.retrieve(u_F0_subdofs, t - dt)
     primal_p_F.retrieve(p_F_subdofs, t)
     primal_U_S.retrieve(U_S_subdofs, t)
     primal_U_S.retrieve(U_S0_subdofs, t - dt)
     primal_P_S.retrieve(P_S_subdofs, t)
     primal_U_M.retrieve(U_M_subdofs, t)
     primal_U_M.retrieve(U_M0_subdofs, t - dt)    

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
     U_F0_global_dofs = append(F_global_index, F_global_index + Nv)
     P_F_global_dofs = F_global_index
     U_S_global_dofs = append(S_global_index, S_global_index + Nv)
     P_S_global_dofs = append(S_global_index, S_global_index + Nv)
     U_M_global_dofs = append(M_global_index, M_global_index + Nv)

     # Get rid of P2 dofs for u_F and create a P1 function
     u_F_subdofs = append(u_F_subdofs[:Nv_F], u_F_subdofs[Nv_F + Ne_F: 2*Nv_F + Ne_F])
     u_F0_subdofs = append(u_F0_subdofs[:Nv_F], u_F0_subdofs[Nv_F + Ne_F: 2*Nv_F + Ne_F])

     # Transfer the stored primal solutions on the dual mesh Omega
     U_F.vector()[U_F_global_dofs] = u_F_subdofs
     U_F0.vector()[U_F0_global_dofs] = u_F0_subdofs
     P_F.vector()[P_F_global_dofs] = p_F_subdofs
     U_S.vector()[U_S_global_dofs] = U_S_subdofs
     U_S0.vector()[U_S_global_dofs] = U_S0_subdofs
     P_S.vector()[U_S_global_dofs] = P_S_subdofs
     U_M.vector()[U_M_global_dofs] = U_M_subdofs
     U_M0.vector()[U_M_global_dofs] = U_M0_subdofs

     return U_F, U_F0, P_F, U_S, U_S0, P_S, U_M, U_M0

# Test
T = 1.0
dt = 0.025
kn = Constant(dt)
t=0.3
get_dual_data(t)
get_primal_data(t)


# Define spatial error indicators \eta_K
RW_h_F_1 = chi * inner(eZ_UF, D_t(U_F, U_F0, kn, U_M, rho_F))*dx \
         - chi * inner(eZ_UF, div(J(U_M)*dot(Sigma_F(U_F, P_F, U_M) ,F_invT(U_M))))*dx
#RW_h_F_2 = chi * inner(eZ_PF, div(J(U_M)*dot(F_inv(U_M), U_F)))*dx
RW_h_F_3 = chi * inner(avg(eZ_UF), 2*mu_F*jump(dot(sym_gradient(U_F), N)))*dS  
RW_h_S_1 = chi * (1/kn)*inner(eZ_US, rho_S*(P_S - P_S0))*dx  - chi * inner(eZ_US, div(Sigma_S(U_S)))*dx
RW_h_S_2 = chi * inner(avg(eZ_US), 2*mu_F*jump(dot(Sigma_S(U_S), N_S)))*dS    
RW_h_S_3 = chi * inner(eZ_US('+'), dot((Sigma_S(U_S)('+') - (J(U_M)('+')*dot(Sigma_F(U_F,P_F,U_M)('+'), F_invT(U_M)('+')))), N_F('+')))*dS(1)
RW_h_S_4 = chi * inner(eZ_PS, (U_S - U_S0) - P_S)*dx
RW_h_M_1 = chi * inner(eZ_UM, alpha*(U_M - U_M0))*dx - inner(v, div(Sigma_M(U_M)))*dx
RW_h_M_2 = chi * inner(avg(eZ_UM), jump(dot(Sigma_M(U_M), N_F)))*dS           
RW_h_M_4 = chi * inner(eZ_PM('+'), U_M('+') - U_S('+'))*dS(1)

# Define time error inidicator R_k
r_k_F_mom = (1/kn)*inner(v, D_t(U_F, U_F0, kn, U_M, rho_F))*dx \
          + inner(grad(v), J(U_M)*dot(Sigma_F(U_F, P_F, U_M), F_invT(U_M)))*dx

r_k_F_con = inner(q, div(U_F))*dx
r_k_S     = (1/kn)*inner(v, rho_S*(P_S - P_S0))*dx \
          + inner(grad(v), Sigma_S(U_S))*dx \
          - inner(v('+'), J(U_M)('+')*dot(Sigma_F(U_F,P_F,U_M)('+'), dot(F_invT(U_M)('+'), N_S('+'))))*dS(1) \
          + inner(v, (U_S -U_S0) - P_S)*dx
    
r_k_M    = (1/kn)*inner(v, alpha*(U_S - U_S0))*dx \
         + inner(sym(grad(v)), Sigma_S(U_S))*dx \
         + inner(v('+'), U_M('+') - U_S('+'))*dS(1)

# Collect vector/scalar contributions
r_k_vector = r_k_F_mom + r_k_S + r_k_M
r_k_scalar = r_k_F_con
 
# Assemble time residuals
R_k_vector = assemble(r_k_vector)
R_k_scalar = assemble(r_k_scalar)

# Compute R_k
R_k = norm(R_k_vector) + norm(R_k_scalar)

print "RK  = ", R_k
plot(eZ_UF, title="extrapolated dual velocity", interactive=True)

