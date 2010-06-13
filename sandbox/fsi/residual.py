"Computes the error indicators on the reference domain"
from dolfin import *
from common import *

# Define constants FIXME: should be retrived from problem!!!
rho_F = 1.0
mu_F = 0.002
rho_S = 1.0
mu_S =  0.15
lmbda_S =  0.25
mu_M =  3.8461
lmbda_M =  5.76

# Define function spaces defined on the whole domain
vector = VectorFunctionSpace(Omega, "CG", 1)
scalar  = FunctionSpace(Omega, "CG", 1)

# Define test functions
v = TestFunction(vector)
q = TestFunction(scalar)
 
# Define projection spaces
vectorDG = VectorFunctionSpace(Omega, "DG", 1)
scalarDG = FunctionSpace(Omega, "DG", 1)

# Define projection functions
vDG = Function(vectorDG)
sDG = Function(scalarDG)

# Create primal functions on Omega
U_F = Function(vector)
P_F = Function(scalar)
U_S = Function(vector)
P_S = Function(vector)
U_M = Function(vector)

# Create primal functions for time-derivateves
U_F0 = Function(vector)
P_S0 = Function(vector)
U_M0 = Function(vector)

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

# k = 1

# get_primal_data(0.2)





# # Define stresses 
# def Sigma_F(u,p,v):
#     return  mu_F*(grad(u)*F_inv(v) + F_invT(v)*grad(u).T) - p*I

# # def Sigma_M(u):
# #     return 2.0*mu_M*sym_gradient(u) + lmbda_M*tr(sym_gradient(u))*I

# # def Sigma_M(u):
# #     return None 

# # Define forms for error the error E_h (see paper for notation)
# R_1_h_F = (1/k)*inner(v, (U_F -U_F0))*dx + inner(v, grad(U_F)*U_F)*dx \
           


# R = (1/k)*inner(grad(v), grad(U_M))*dx
# jada = assemble(R)
# print jada

# # Compute space residals (aka Rannacher's eta k's)
# def compute_space_residuals(t):
    
    
    
# Create .bin files for storinf residual information



