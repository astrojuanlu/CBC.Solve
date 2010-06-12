"Computes the error indicators on the reference domain"
from dolfin import *
from common import Omega

# Define constants FIXME: should be retrived from problem!!!
rho_F = 1.0
mu_F = 0.002
rho_S = 1.0
mu_S =  0.15
lmbda_S =  0.25
mu_M =  3.8461
lmbda_M =  5.76

# Define function spaces defined on the whole domain
V_F1 = VectorFunctionSpace(Omega, "CG", 1)
V_F2 = VectorFunctionSpace(Omega, "CG", 2)
Q_F  = FunctionSpace(Omega, "CG", 1)
V_S  = VectorFunctionSpace(Omega, "CG", 1)
Q_S  = VectorFunctionSpace(Omega, "CG", 1)
V_M  = VectorFunctionSpace(Omega, "CG", 1)
Q_M  = VectorFunctionSpace(Omega, "CG", 1)

# Define projection spaces
vectorDG = VectorFunctionSpace(Omega, "DG", 1)
scalarDG = FunctionSpace(Omega, "DG", 1)

# Create primal functions on Omega
U_F = Function(V_F1)
P_F = Function(Q_F)
U_S = Function(V_S)
P_S = Function(Q_S)
U_M = Function(V_M)
P_M = Function(Q_M) #FIXME: Check for sure that Im not needed in the primal!

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
