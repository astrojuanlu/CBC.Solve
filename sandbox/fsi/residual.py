# Computes the error indicators on the reference domain. Note that all error indicators are
# evaluated on Omega since the dual solution lives on Omega

from dolfin import *
from common import *
from operators import *

# Define function spaces defined on the whole domain
V = VectorFunctionSpace(Omega, "CG", 1)
Q = FunctionSpace(Omega, "CG", 1)

# Define test/trial functions
v = TestFunction(V)
q = TestFunction(Q)
qtrial = TrialFunction(Q)
vtrial = TrialFunction(V)

## Define projection spaces
V_DG = VectorFunctionSpace(Omega, "DG", 1)
Q_DG = FunctionSpace(Omega, "DG", 1)

# Define projection functions
vDG = Function(V_DG)
sDG = Function(V_DG)

# Create primal functions on Omega
U_F = Function(V)
P_F = Function(Q)
U_S = Function(V)
P_S = Function(V)
U_M = Function(V)

# Create primal functions for time-derivateves
U_F0 = Function(V)
U_S0 = Function(V)
P_S0 = Function(V)
U_M0 = Function(V)

# Define facet normals
N = FacetNormal(Omega)
N_F = FacetNormal(Omega_F)
N_S = FacetNormal(Omega_S)

# Create solution vectors for residuals
R_h_F_1 = Function(V)
R_h_F_2 = Function(Q)
R_h_F_3 = Function(V)
R_h_S_1 = Function(V)
R_h_S_2 = Function(V)
R_h_S_3 = Function(V)
R_h_S_4 = Function(V)
R_h_M_1 = Function(V)
R_h_M_2 = Function(V)
R_h_M_3 = Function(V)
R_h_M_4 = Function(V)

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
    U_S_subdofs = Vector()
    P_S_subdofs = Vector()
    U_M_subdofs = Vector()

    # Get primal data 
    primal_u_F.retrieve(u_F_subdofs, t)
    primal_p_F.retrieve(p_F_subdofs, t)
    primal_U_S.retrieve(U_S_subdofs, t)
    primal_P_S.retrieve(P_S_subdofs, t)
    primal_U_M.retrieve(U_M_subdofs, t)

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

k = 1
get_primal_data(0.5)

# Create .bin files for store residual information
# R_h = ...
# R_k = ...

# Define forms for residuals R_h (see paper for notation)
r_h_F_1 = (1/k)*inner(v, D_t(U_F, U_F0, U_M, rho_F))*dx \
          - inner(v, div(J(U_M)*dot(Sigma_F(U_F, P_F, U_M) ,F_invT(U_M))))*dx
r_h_F_2 = inner(q, div(J(U_M)*dot(F_inv(U_M), U_F)))*dx
#r_h_F_3 = inner(v('+'), 2*mu_F*dot(jump(sym(grad(U_F))), N_F('+')))*dS  # FIXME: Jump terms do not work
r_h_S_1 = inner(v, rho_S*(P_S - P_S0) - div(Sigma_S(U_S)))*dx 
#R_h_S_2 = inner(v('+'), 2*mu_F*dot(jump(Sigma_S(U_S)), N_S('+')))*dS    # FIXME: Jump terms do not work
r_h_S_3 = inner(v('+'), dot((Sigma_S(U_S)('+') + (J(U_M)('+')*dot(Sigma_F(U_F,P_F,U_M)('+'), F_invT(U_M)('+')))), N_F('+')))*dS(1)
r_h_S_4 = inner(v, (U_S - U_S0) - P_S)*dx
r_h_M_1 = inner(v, alpha*(U_M - U_M0))*dx - inner(v, div(Sigma_M(U_M)))*dx
#R_h_M_2 = inner(v('+'), dot(jump(Sigma_M(U_M)), N_F('+')))*dS           # FIXME: Jump terms do not work
#R_h_M_3 = inner(v, P_M)*dx                                              # FIXME: This term is zero in the primal, how to fix?
r_h_M_4 = inner(v('+'), U_M('+') - U_S('+'))*dS(1)

# Define forms for the residual R_k (see paper for details)


# Assemble forms for residuals R_h
L_h_F_1 = assemble(r_h_F_1, interior_facet_domains=interior_facet_domains)
L_h_F_2 = assemble(r_h_F_2, interior_facet_domains=interior_facet_domains)
#L_h_F_3 = assemble(r_h_F_3, interior_facet_domains=interior_facet_domains)
L_h_S_1 = assemble(r_h_S_1, interior_facet_domains=interior_facet_domains)
#L_h_S_2 = assemble(r_h_S_2, interior_facet_domains=interior_facet_domains)
L_h_S_3 = assemble(r_h_S_3, interior_facet_domains=interior_facet_domains)
L_h_S_4 = assemble(r_h_S_4, interior_facet_domains=interior_facet_domains)
L_h_M_1 = assemble(r_h_M_1, interior_facet_domains=interior_facet_domains)
#L_h_M_2 = assemble(r_h_M_2, interior_facet_domains=interior_facet_domains)
#L_h_M_3 = assemble(r_h_M_3, interior_facet_domains=interior_facet_domains)
L_h_M_4 = assemble(r_h_M_4, interior_facet_domains=interior_facet_domains)

# Define and assemble rhs
a = inner(vtrial, v)*dx
A = assemble(a, interior_facet_domains=interior_facet_domains)
a_s = qtrial*q*dx
As = assemble(a_s, interior_facet_domains=interior_facet_domains)

# Compute residuals R_h
solve(A, R_h_F_1.vector(), L_h_F_1)
solve(As, R_h_F_2.vector(), L_h_F_2)
# solve(A, R_h_F_3.vector(), L_h_F_3)
solve(A, R_h_S_1.vector(), L_h_S_1)
# solve(A, R_h_S_2.vector(), L_h_S_2)
solve(A, R_h_S_3.vector(), L_h_S_3)
solve(A, R_h_S_4.vector(), L_h_S_4)
solve(A, R_h_M_1.vector(), L_h_M_1)
# solve(A, R_h_M_2.vector(), L_h_M_2)
# solve(A, R_h_M_3.vector(), L_h_M_3)
solve(A, R_h_M_4.vector(), L_h_M_4)





# file_jada = File("jada.pvd")
# file_jada << R_h_S_3
# plot(R_h_S_3, mesh=Omega)
# interactive()





# # Compute residuals
# def compute_space_residuals(t):
    
    
    




