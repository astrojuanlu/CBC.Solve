" Computes the error indicators on the reference domain. Note that all error indicators are "
" evaluated on Omega since the dual solution lives on Omega "

from dolfin import *
from operators import *
from common import *

class Residual():

    def __init__(self):

        # Define projection spaces
        V = VectorFunctionSpace(Omega, "DG", 0)
        Q = FunctionSpace(Omega, "DG", 0)

        # Define test/trial functions
        v = TestFunction(V)
        q = TestFunction(Q)

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
        R_k_v   = Function(V)
        R_k_s   = Function(Q)
        # FIXME: Add this term!!!
        R_C     = Function(V)

        # Get time step
        kn = Constant(dt)

        # Define forms for residuals R_h (see paper for notation)
        r_h_F_1 = (1/kn)*inner(v, D_t(U_F, U_F0, U_M, rho_F))*dx \
                - inner(v, div(J(U_M)*dot(Sigma_F(U_F, P_F, U_M) ,F_invT(U_M))))*dx
        r_h_F_2 = inner(q, div(J(U_M)*dot(F_inv(U_M), U_F)))*dx
        r_h_F_3 = inner(avg(v), 2*mu_F*jump(dot(sym_gradient(U_F), N)))*dS
        r_h_S_1 = (1/kn)*inner(v, rho_S*(P_S - P_S0))*dx  - inner(v, div(Sigma_S(U_S)))*dx
        r_h_S_2 = inner(avg(v), 2*mu_F*jump(dot(Sigma_S(U_S), N_S)))*dS
        r_h_S_3 = inner(v('+'), dot((Sigma_S(U_S)('+') - (J(U_M)('+')*dot(Sigma_F(U_F,P_F,U_M)('+'), F_invT(U_M)('+')))), N_F('+')))*dS(1) # FIXME: Check if this is correct
        r_h_S_4 = inner(v, (U_S - U_S0) - P_S)*dx
        r_h_M_1 = inner(v, alpha*(U_M - U_M0))*dx - inner(v, div(Sigma_M(U_M)))*dx
        r_h_M_2 = inner(avg(v), jump(dot(Sigma_M(U_M), N_F)))*dS
        r_h_M_4 = inner(v('+'), U_M('+') - U_S('+'))*dS(1)

        # Define forms for the residual R_k (see paper for details)
        r_k_F_mom = (1/kn)*inner(v, D_t(U_F, U_F0, U_M, rho_F))*dx \
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

        # Store variabels for time stepping
        self.U_F  = U_F
        self.P_F  = P_F
        self.U_S  = U_S
        self.P_S  = P_S
        self.U_M  = U_M
        self.U_F0 = U_F0
        self.U_S0 = U_S0
        self.P_S0 = P_S0
        self.U_M0 = U_M0
        self.r_h_F_1 = r_h_F_1
        self.r_h_F_2 = r_h_F_2
        self.r_h_F_3 = r_h_F_3
        self.r_h_S_1 = r_h_S_1
        self.r_h_S_2 = r_h_S_2
        self.r_h_S_3 = r_h_S_3
        self.r_h_S_4 = r_h_S_4
        self.r_h_M_1 = r_h_M_1
        self.r_h_M_2 = r_h_M_2
        self.r_h_M_4 = r_h_M_4
        self.r_k_vector = r_k_vector
        self.r_k_scalar = r_k_scalar

    # Retrieve primal data
    def get_primal_data(self, t, dt):

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
        self.U_F.vector()[U_F_global_dofs] = u_F_subdofs
        self.U_F0.vector()[U_F0_global_dofs] = u_F0_subdofs
        self.P_F.vector()[P_F_global_dofs] = p_F_subdofs
        self.U_S.vector()[U_S_global_dofs] = U_S_subdofs
        self.U_S0.vector()[U_S_global_dofs] = U_S0_subdofs
        self.P_S.vector()[U_S_global_dofs] = P_S_subdofs
        self.U_M.vector()[U_M_global_dofs] = U_M_subdofs
        self.U_M0.vector()[U_M_global_dofs] = U_M0_subdofs

        return self.U_F, self.U_F0, self.P_F, self.U_S, self.U_S0, self.P_S, self.U_M, self.U_M0


    def compute_residuals(self, t, dt):

        # Get primal data
        self.get_primal_data(t, dt)

        # Assemble/compute forms for residuals R_h
        R_h_F_1 = assemble(self.r_h_F_1, interior_facet_domains=interior_facet_domains)
        R_h_F_2 = assemble(self.r_h_F_2, interior_facet_domains=interior_facet_domains)
        R_h_F_3 = assemble(self.r_h_F_3, interior_facet_domains=interior_facet_domains)
        R_h_S_1 = assemble(self.r_h_S_1, interior_facet_domains=interior_facet_domains)
        R_h_S_2 = assemble(self.r_h_S_2, interior_facet_domains=interior_facet_domains)
        R_h_S_3 = assemble(self.r_h_S_3, interior_facet_domains=interior_facet_domains)
        R_h_S_4 = assemble(self.r_h_S_4, interior_facet_domains=interior_facet_domains)
        R_h_M_1 = assemble(self.r_h_M_1, interior_facet_domains=interior_facet_domains)
        R_h_M_2 = assemble(self.r_h_M_2, interior_facet_domains=interior_facet_domains)
        R_h_M_4 = assemble(self.r_h_M_4, interior_facet_domains=interior_facet_domains)

        # Assemble R_k
        R_k_vector = assemble(self.r_k_vector)
        R_k_scalar = assemble(self.r_k_scalar)

        # Compute R_k (just a real number)
        R_k = norm(R_k_vector) + norm(R_k_scalar)

        # Just for checking
        #return  [R_h_F_1 , R_h_F_2, R_h_F_3, R_h_S_1, R_h_S_2, R_h_S_3, R_h_S_4, R_h_M_1, R_h_M_2, R_h_M_4, R_k]
        return R_k




