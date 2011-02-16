from dolfin import *
from cbc.beat.heart import Heart

class DynamicBidomainSolver:

    def __init__(self):
        pass

    def __call__(self, heart):
        self.simulate(heart)

class FullyImplicit(DynamicBidomainSolver):

    def __init__(self, dt=0.01, k=1):
        self.time_step = dt
        self.degree = k

    def _backward_euler(self, v, v0):
        return (1.0/Constant(self.time_step))*(v - v0)

    def simulate(self, heart):

        Dt = self._backward_euler

        mesh = heart.mesh()

        # Definition of numerical scheme
        V = FunctionSpace(mesh, "CG", self.degree)
        Q = FunctionSpace(mesh, "CG", self.degree)
        S = FunctionSpace(mesh, "DG", self.degree-1) # FIXME
        W = MixedFunctionSpace([V, Q, S])

        # Unknowns and test functions
        y = Function(W)
        (v, u, s) = split(y)
        (w, q, r) = TestFunctions(W)

        if not heart.essential_boundaries():
            info("No essential boundaries. Assuming vanishing average u.")
            R = FunctionSpace(mesh, "R", 0)
            W = MixedFunctionSpace([V, Q, S, R])
            y = Function(W)
            (w, q, r, d_u) = TestFunctions(W)
            (v, u, s, c_u) = split(y)

        # Conductivities
        (M_i, M_ie) = heart.conductivities()

        # Initial conditions
        ics = heart.initial_conditions()
        if not ics:
            info("No initial conditions prescribed. Assuming starting at rest.")
            (v0, s0) = (Function(V), Function(V))

        # Get applied boundary current
        g = heart.boundary_current()
        if not g:
            g = Constant(0.0)

        # Get applied body current
        I_e = heart.applied_current()
        if not I_e:
            I_e = Constant(0.0)

        # Extract cell model dependent stuff
        I = heart.cell_model().I
        F = heart.cell_model().F

        # Parabolic equation
        F0 = (Dt(v, v0)*w
              + inner(M_i(grad(v)), grad(w)) + inner(M_i(grad(u)), grad(w))
              + I(v, s)*w)*dx \
              + g*w*ds

        # Elliptic equation
        F1 = (inner(M_i(grad(v)), grad(q)) + inner(M_ie(grad(u)), grad(q))
              + I_e*q)*dx \
              + g*q*ds

        # State variables
        F2 = (Dt(s, s0)*r - F(v, s)*r)*dx

        # Full system
        E = F0 + F1 + F2

        # Non-singularize (ground system) if no essential boundary
        # conditions are applied for u
        bcs = [] # FIXME
        if not heart.essential_boundaries():
            info("Applying vanishing average Lagrange multiplier")
            F3 = (u*d_u + q*c_u)*dx
            E = E + F3

        # Find Jacobian
        dy = TrialFunction(W)
        dE = derivative(E, y, dy)

        solution = Function(W)

        t = 0.0
        T = heart.end_time()
        file = File("beat_generated_data/membrane.pvd")
        sfile = File("beat_generated_data/state.pvd")
        while (t <= T):

            # Update sources
            g.t = t
            #I_e.t = t

            # Solve (non-linear) variational problem
            pde = VariationalProblem(E, dE)

            # Solve
            pde.solve(y)

            # Extract interesting variables
            (v, u, s) = y.split()[0:3]

            # Plot/Store solutions
            #plot(v, title="Membrane potential")
            file << v
            sfile << s

            # Update solutions at previous time-step
            v0.assign(v)
            s0.assign(s)

            # Update time
            t += self.time_step

