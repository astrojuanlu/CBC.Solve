from common import *
from cbc.common import CBCSolver
from cbc.common.utils import *
from cbc.twist.kinematics import SecondOrderIdentity
from numpy import array, append, zeros, linspace

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

# Define operators that is needed in the forms
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
# (u is always U_M)
def DJ(u,w):
    DJ = w[0].dx(0)*(1 - u[1].dx(1)) - w[0].dx(1)*u[1].dx(0) \
        -w[1].dx(0)*u[0].dx(1) + w[1].dx(1)*(1 + u[0].dx(0))
    return DJ

def J(u):
    return det(F(u))

I = variable(Identity(2))

def sym_gradient(u):
    sym_gradient = 0.5*(grad(u)+ grad(u).T)
    return sym_gradient

def Sigma_F(u,p,v):
    return  mu_F*(grad(u)*F_inv(v) + F_invT(v)*grad(u).T) - p*I

def sigma_M(u):
    return 2.0*mu_M*sym_gradient(u) + lmbda_M*tr(sym_gradient(u))*I

# Define constants FIXME: should be retrived from problem!!!
rho_F = 1.0
mu_F = 0.002
rho_S = 1.0
mu_S =  0.15
lmbda_S =  0.25
mu_M =  3.8461
lmbda_M =  5.76

# Fix time step if needed. Note that this has to be done
# in oder to save the primal data at the correct time
dt, t_range = timestep_range(T, dt)
kn = dt

# Define FSI normal
N_S =  FacetNormal(Omega_S)
N =  N_S('+')

# Fluid eq. linearized around fluid variables
A_FF01 = -(1/kn)*inner((Z_UF0 - Z_UF), rho_F*J(U_M)*v_F)*dx(0)
A_FF02 =  inner(Z_UF, rho_F*J(U_M)*dot(dot(grad(v_F),F_inv(U_M)), (U_F - (U_M0 - U_M)*(1/kn))))*dx(0)
A_FF03 =  inner(Z_UF, rho_F*J(U_M)*dot(grad(U_F) , dot(F_inv(U_M), v_F)))*dx(0)
A_FF04 =  inner(grad(Z_UF), J(U_M)*mu_F*dot(grad(v_F) , dot(F_inv(U_M), F_invT(U_M))))*dx(0)
A_FF05 =  inner(grad(Z_UF), J(U_M)*mu_F*dot(F_invT(U_M) , dot(grad(v_F).T, F_invT(U_M))))*dx(0)
A_FF06 = -inner(grad(Z_UF), J(U_M)*q_F*F_invT(U_M))*dx(0)
A_FF07 =  inner(Z_PF, div(J(U_M)*dot(F_inv(U_M),v_F)))*dx(0)

# Collect A_FF form
A_FF = A_FF01 + A_FF02 + A_FF03 + A_FF04 + A_FF05 + A_FF06 + A_FF07

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

# Collect A_FM form
A_FM =  A_FM01 + A_FM02 + A_FM03 + A_FM04 + A_FM05 + A_FM06 + A_FM07 + A_FM08 + A_FM09 + A_FM10 + A_FM11 + A_FM12

# Structure eq. linearized around the fluid variables
A_SF01 = -inner(Z_US('+'), mu_F*J(U_M)('+')*dot(dot(grad(v_F('+')), F_inv(U_M)('+')), dot(F_invT(U_M)('+'), N)))*dS(1)
A_SF02 = -inner(Z_US('+'), mu_F*J(U_M)('+')*dot(dot(F_invT(U_M)('+'), grad(v_F('+')).T), dot(F_invT(U_M)('+'), N)))*dS(1)
A_SF03 =  inner(Z_US('+'), mu_F*J(U_M)('+')*q_F('+')*dot(I('+'), dot(F_invT(U_M)('+'), N)))*dS(1)

# Collect A_SF form
A_SF = A_SF01 + A_SF02 + A_SF03

Fu = F(U_S)
#Fu = I
Eu = Fu*Fu.T - I
Ev = grad(v_S)*Fu.T + Fu*grad(v_S).T
Sv = grad(v_S)*(2*mu_S*Eu + lmbda_S*tr(Eu)*I) + Fu*(2*mu_S*Ev + lmbda_S*tr(Ev)*I)

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
A_MM01 = -(1/kn)*inner(v_M, Z_UM0 - Z_UM)*dx(0) + inner(sym_gradient(Z_UM), sigma_M(v_M))*dx(0)
A_MM02 = inner(Z_UM('+'),v_M('+'))*dS(1)
A_MM03 = inner(Z_PM('+'),q_M('+'))*dS(1)

# Collect A_MM form
A_MM = A_MM01 + A_MM02 + A_MM03

# Mesh eq. linearized around structure variable
#A_MS = - inner(Z_PM('+'), q_S('+'))*dS(1)

# FIXME: Temporary fix
A_MS = -inner(Z_PM('+'), q_S('+'))*dS(1)

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

# Create files
file_Z_UF = File("Z_UF.pvd")
file_Z_PF = File("Z_PF.pvd")
file_Z_US = File("Z_US.pvd")
file_Z_PS = File("Z_PS.pvd")
file_Z_UM = File("Z_UM.pvd")
file_Z_PM = File("Z_PM.pvd")

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
   get_primal_data(t)

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
   U_M0.assign(U_M)  # FIXME: Should be done when we get the primal data
   U_F0.assign(U_F)  # FIXME: Should be done when we get the primal data

   # Save solutions
   file_Z_UF << Z_UF
   file_Z_PF << Z_PF
   file_Z_US << Z_US
   file_Z_PS << Z_PS
   file_Z_UM << Z_UM
   file_Z_PM << Z_PM

#    # Plot solutions
#    plot(Z_UF, title="Dual fluid velocity")
#    plot(Z_PF, title="Dual fluid pressure")
#    plot(Z_US, title="Dual displacement")
#    plot(Z_PS, title="Dual structure velocity")
#    plot(Z_UM, title="Dual mesh displacement")
#    plot(Z_PM, title="Dual mesh Lagrange Multiplier")
#    interactive()

   # Move to next time interval
   t += kn
   # FIXME: Change to t =+ float(kn)

