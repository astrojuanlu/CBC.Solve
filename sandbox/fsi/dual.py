# A simple dual solver for the FSI problem
# As a first try we solve a stationary problem
#
# The linearized problem on block form is A'*(v,Z) = M(v):
#
#   | A_FF    A_FS    A_FM |*   |(Z_UF, Z_PF)|
#   | A_SF    A_SS    A_SM |    |   Z_US     | =  M
#   | A_MF    A_MS    A_MM |    |   Z_UM     |
#
# Note that A_FS, A_MF and A_MS  = 0 by the construction
# of the problem. If we use Mats idea of an extion operator 
# for the mesh problem we can obtain a Neumann BC in the mesh problem 
# such that A_MS \noteq 0.
#
# In the forms above, the adjoint * has been applied on the matrix 
# as well on the test/trial functions, see appendix for details.
# The dual_matrix is then 
# 
#                  | A_FF    A_SF    0    |    
#  dual_matrix  =  | 0       A_SS    0    | 
#                  | A_FM    A_SM    A_MM |
# 
#
# Sub_domain markers are defined in common.py. The structure is fluid is marked as 0
# and the structure as 1 

from common import *
from cbc.common import CBCSolver
from cbc.twist.kinematics import SecondOrderIdentity
from numpy import array, append, zeros

# parameters["form_compiler"]["cpp_optimize"] = True
# parameters["form_compiler"]["optimize"] = True

# Define function spaces defined on the whole domain
V_F1 = VectorFunctionSpace(Omega, "CG", 1)
V_F2 = VectorFunctionSpace(Omega, "CG", 2)
Q_F  = FunctionSpace(Omega, "CG", 1)
V_S  = VectorFunctionSpace(Omega, "CG", 1)
V_M  = VectorFunctionSpace(Omega, "CG", 1) 
V_PM = VectorFunctionSpace(Omega, "CG", 1) 

# Create mixed function space
mixed_space = (V_F2, Q_F, V_S, V_M, V_PM)
W = MixedFunctionSpace(mixed_space)

# Define test functions
(v_F, q_F, v_S, v_M, q_M) = TestFunctions(W)

# Define trial functions
(Z_UF, Z_PF, Z_US, Z_UM, Z_PM) = TrialFunctions(W)

# Create dual functions
# Note that V_F etc are defined on the whole domain
U_F = Function(V_F1)
P_F = Function(Q_F)
U_S = Function(V_S)
U_M = Function(V_M)

# Create vectors for primal dofs
u_F_subdofs = Vector()
p_F_subdofs = Vector()
U_M_subdofs = Vector()
U_S_subdofs = Vector()

# Get primal data
primal_u_F.retrieve(u_F_subdofs, T - t)
primal_p_F.retrieve(p_F_subdofs, T - t)
primal_U_M.retrieve(U_M_subdofs, T - t)
primal_U_S.retrieve(U_S_subdofs, T - t)

# Initialize mesh convectivity
Omega.init(1,2)
Omega_F.init(1,2)

# Create mapping from Omega_(F,S,M) to Omega (i.e extend by zero)
global_edge_indices_F = Omega_F.data().create_mesh_function("global edge indices", 1)
global_vertex_indices_F = Omega_F.data().mesh_function("global vertex indices")
global_vertex_indices_S = Omega_S.data().mesh_function("global vertex indices")
global_vertex_indices_M = Omega_F.data().mesh_function("global vertex indices") 

# Create lists
F_global_index = zeros([global_vertex_indices_F.size()], "uint") 
F_global_edge_index = zeros([global_edge_indices_F.size()], "uint")
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
                
# Initialize edges on fluid sub mesh (needed for P2 elements)
Omega_F.init(1)

# Define number of vertices and edges
Nv = Omega.num_vertices()
Nv_F = Omega_F.num_vertices()
Ne_F = Omega_F.num_edges()

# Get global dofs 
U_F_global_dofs = append(F_global_index, F_global_index + Nv)
P_F_global_dofs = F_global_index
U_S_global_dofs = append(S_global_index, S_global_index + Nv)
U_M_global_dofs = append(M_global_index, M_global_index + Nv)

# Get rid of P2 dofs for u_F and create a P1 function
u_F_subdofs = append(u_F_subdofs[:Nv_F], u_F_subdofs[Nv_F + Ne_F: 2*Nv_F + Ne_F])

# Get inactive dofs for global system
active_dofs = set()
active_dofs.update(tuple(U_F_global_dofs))
active_dofs.update(tuple(P_F_global_dofs + 2*Nv))
active_dofs.update(tuple(U_S_global_dofs + 2*Nv + Nv))
active_dofs.update(tuple(U_M_global_dofs + 2*Nv + Nv + 2*Nv))
inactive_dofs = set(range(W.dim()))-active_dofs
inactive_dofs = array(tuple(inactive_dofs), dtype="I")

# Transfer the stored primal solutions on the dual mesh Omega
U_F.vector()[U_F_global_dofs] = u_F_subdofs
P_F.vector()[P_F_global_dofs] = p_F_subdofs
U_S.vector()[U_S_global_dofs] = U_S_subdofs
U_M.vector()[U_M_global_dofs] = U_M_subdofs

#plot(P_F, interactive=True)

file = File("U_F_dual.pvd")
file << U_F
file = File("U_S_dual.pvd")
file << U_S

# Define the Jacobian matrices and determinants
def F(u):
    I = SecondOrderIdentity(u)
    F = (I + grad(u)) 
    return F

def F_inv(u):
    I = SecondOrderIdentity(u)
    F_inv  = inv((I + grad(u)))
    return F_inv

def F_T(u):
    I = SecondOrderIdentity(u)
    F_T  = ((I + grad(u))).T
    return F_T

def F_invT(u):
    I = SecondOrderIdentity(u)
    F_invT  = (inv((I + grad(u)))).T
    return F_invT

def J(u):
    I = SecondOrderIdentity(u)
    J = det(I + grad(u))
    return J

# DJ(u,w) is J(u) linearized around w (w = test function)
def DJ(u,w):
    DJ = w[0].dx(0)*(1 - u[1].dx(1)) - w[0].dx(1)*u[1].dx(0) \
        -w[1].dx(0)*u[0].dx(1) + w[1].dx(1)*(1 + u[0].dx(0))
    return DJ

def I(u):
    I = SecondOrderIdentity(u)
    return I

def sym_gradient(u):
    sym_gradient = 0.5*(grad(u)+ grad(u).T)
    return sym_gradient
    
# Define constants
mu_F = 1
mu_S = 1
lamb_S = 1
mu_M = 1 
lamb_M = 1

def sigma_M(u):
    return 2.0*mu_M*sym_gradient(u) + lamb_M*tr(sym_gradient(u))*I(u)

# Fluid eq. linearized around fluid variables
A_FF01 = 0 # time--dependent
A_FF02 = 0 # time--dependent
A_FF03 =  inner(Z_UF, dot(grad(U_F) , dot(F(U_M), v_F)))*dx(0)
A_FF04 =  inner(grad(Z_UF), mu_F*dot(grad(v_F) , dot(F_inv(U_M), F_invT(U_M))))*dx(0)
A_FF05 =  inner(grad(Z_UF), mu_F*dot(F_invT(U_M) , dot(grad(v_F).T, F_invT(U_M))))*dx(0)
A_FF06 =  -inner(grad(Z_UF), (q_F*dot(I(U_M), F_invT(U_M))))*dx # FIXME: Which one should we use?
A_FF06 =  -inner(grad(Z_UF), q_F*F_invT(U_M))*dx(0)
A_FF07 =  inner(Z_PF, inner(F_invT(U_M), grad(v_F)))*dx(0) 

# Collect A_FF form
A_FF_sum = A_FF03 + A_FF04 + A_FF05 + A_FF06 + A_FF07

# Fluid eq. linearized around mesh variable
A_FM01 = 0 # time--dependent
A_FM02 = 0 # time--dependent
A_FM03 = -inner(grad(Z_UF), dot(mu_F*(dot(grad(U_F), dot(F_inv(U_M), dot(grad(v_M),F_inv(U_M))))),F_invT(U_M)))*dx(0)
A_FM04 = -inner(grad(Z_UF), dot(mu_F*(dot(F_invT(U_M), dot(grad(v_M).T, dot(F_invT(U_M), grad(U_F).T )))),F_invT(U_M)))*dx(0)
A_FM05 = -inner(grad(Z_UF), dot(mu_F*(dot(grad(U_F), dot(F_inv(U_M), dot(F_invT(U_M), grad(v_M).T )))),F_invT(U_M)))*dx(0)
A_FM06 = -inner(grad(Z_UF), dot(mu_F*(dot(F_invT(U_M) , dot(grad(U_M).T, dot(F_invT(U_M), grad(v_M).T )))),F_invT(U_M)))*dx(0)
A_FM07 =  inner(grad(Z_UF), dot(dot( P_F*I(U_F),F_invT(U_M)) ,  dot(grad(v_M).T ,F_invT(U_M) )))*dx(0)
A_FM08 = -inner(Z_PF, inner( dot(F_invT(U_M), grad(v_M).T)  , dot( I(U_M), grad(U_F))))*dx(0) # FIXME: Is this the right way to write it????

# Collect A_FM form
A_FM_sum = A_FM03 + A_FM04 + A_FM05 + A_FM06 + A_FM07 + A_FM08

# Define FSI normal FIXME: Change!!!!
N_S =  FacetNormal(Omega_S)
N =  N_S('+')

# Structure eq. linearized around the fluid variables
A_SF01 = -inner(Z_US('+'), mu_F*J(U_M)('+')*dot(dot(grad(v_M('+')), F_inv(U_M)('+')), dot(F_invT(U_M)('+'), N)))*dS(1)
A_SF02 = -inner(Z_US('+'), mu_F*J(U_M)('+')*dot(dot(F_invT(U_M)('+'), grad(v_M('+')).T), dot(F_invT(U_M)('+'), N)))*dS(1)
A_SF03 =  inner(Z_US('+'), mu_F*J(U_M)('+')*q_F('+')*dot(I(U_M)('+'), dot(F_invT(U_M)('+'), N)))*dS(1)

# Collect A_SF form
A_SF_sum = A_SF01 + A_SF02 + A_SF03 

# Structure eq. linearized around the structure variable
A_SS01 = 0 # time--dependent
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
A_MS_sum = - inner(Z_PM('+'), v_S('+'))*dS(1)\


# Collect the dual matrix
A_dual = A_FF_sum + A_SF_sum + A_SS_sum + A_FM_sum + A_SM_sum + A_MM_sum + A_MS_sum

# Define goal functionals
#goal_MF = inner(v_F('+'), dot(grad(U_F('+')), N))*dS(1) + inner(v_F('+'), dot(grad(U_F('+')).T, N))*dS(1) - P_F('+')*inner(v_F('+'), dot(I(v_F)('+'), N))*dS(1)
#goal_MS  = inner(sigma(v_F('+'), q_F('+')), N)[0]*dS(1)
#MS_hat  = Constant((1.0, 0.0)) # FIXME: Make smoother
#goal_MS = inner(v_S, U_S)*dx(1)
jada = Constant((1.0, 0.0))
goal_F = -inner(jada, v_F)*ds(0)
GOAL = goal_F


#  M_F = inner(sigma(v_F('+'), v_P('+')), N)[0]*dS(1)
L_dual = GOAL

# Assemble dual matrix
dual_matrix = assemble(A_dual, cell_domains = cell_domains, interior_facet_domains = interior_facet_domains)
dual_vector = assemble(L_dual, cell_domains = cell_domains, interior_facet_domains = interior_facet_domains)

mfile = File("matrix.m")
mfile << dual_matrix
# import sys
# sys.exit(1)

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

# Apply bcs
for bc in bcs:
    bc.apply(dual_matrix, dual_vector)

# Fix inactive dofsx
#dual_matrix.ident(inactive_dofs)
dual_matrix.ident_zeros()

# Compute dual solution
Z = Function(W)
solve(dual_matrix, Z.vector(), dual_vector)

(Z_UF, Z_PF, Z_S, Z_UM, Z_PM) = Z.split()


file_Z_UF = File("Z_UF.pvd")
file_Z_UF << Z_UF
file_Z_PF = File("Z_PF.pvd")
file_Z_PF << Z_PF
file_Z_S = File("Z_S.pvd")
file_Z_S << Z_S
file_Z_UM = File("Z_UM.pvd")
file_Z_UM << Z_UM
file_Z_PM = File("Z_PM.pvd")
file_Z_PM << Z_PM


plot(Z_UF, title="Dual velocity")
plot(Z_PF, title="Dual pressure")
plot(Z_S,  title="Dual displacement")
plot(Z_UM,  title="Dual mesh displacement")
plot(Z_PM,  title="Dual mesh Lagrange Multiplier")
interactive()
























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
