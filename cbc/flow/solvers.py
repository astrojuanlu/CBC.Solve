__author__ = "Kristian Valen-Sendstad and Anders Logg"
__copyright__ = "Copyright (C) 2009 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

# Modified by Harish Narayanan, 2012
# Last changed: 2012-05-01

__all__ = ["NavierStokesSolver"]

from dolfin import *
from cbc.common.utils import *
from cbc.common import *

class NavierStokesSolver(CBCSolver):
    "Navier-Stokes solver"

    def __init__(self, problem):
        "Initialize Navier-Stokes solver"

        # Initialize base class
        CBCSolver.__init__(self)

        # Set up parameters
        self.parameters = Parameters("solver_parameters")
        self.parameters.add("plot_solution", True)
        self.parameters.add("zero_average_pressure", False)
        self.parameters.add("save_solution", True)
        self.parameters.add("store_solution_data", False)

        # Get mesh and time step range
        mesh = problem.mesh()
        dt, t_range = timestep_range_cfl(problem, mesh)
        info("Using time step dt = %g" % dt)

        # Function spaces
        V1 = VectorFunctionSpace(mesh, "CG", 1)
        V = VectorFunctionSpace(mesh, "CG", 2)
        Q = FunctionSpace(mesh, "CG", 1)
        R = FunctionSpace(mesh, "R", 0)

        # Coefficients
        mu = Constant(problem.viscosity())  # Dynamic viscosity [Ps x s]
        rho = Constant(problem.density())   # Density [kg/m^3]
        n = FacetNormal(mesh)
        k = Constant(dt)
        f = problem.body_force(V1)
        g = problem.boundary_traction(V1)
        w = problem.mesh_velocity(V1)

        # If no body forces are specified, assume it is 0
        if f == [] or f is None:
            f = Constant((0,)*V1.mesh().geometry().dim())

        # If no boundary forces are specified, assume it is 0
        if g is None or g == []:
            g = Constant((0,)*V1.mesh().geometry().dim())

        # Create boundary conditions
        bcu = create_dirichlet_conditions(problem.velocity_dirichlet_values(),
                                          problem.velocity_dirichlet_boundaries(),
                                          V)
        bcp = create_dirichlet_conditions(problem.pressure_dirichlet_values(),
                                          problem.pressure_dirichlet_boundaries(),
                                          Q)

        # Create initial conditions
        u0 = create_initial_condition(problem.velocity_initial_condition(), V)
        p0 = create_initial_condition(problem.pressure_initial_condition(), Q)

        # Test and trial functions
        v = TestFunction(V)
        q = TestFunction(Q)
        u = TrialFunction(V)
        p = TrialFunction(Q)

        # Functions
        u1 = interpolate(u0, V)
        p1 = interpolate(p0, Q)

        # Define Cauchy stress tensor
        def sigma(v,w):
            return 2.0*mu*0.5*(grad(v) + grad(v).T)  - w*Identity(v.cell().d)

        # Define symmetric gradient
        def epsilon(v):
            return  0.5*(grad(v) + grad(v).T)

        # Tentative velocity step (sigma formulation)
        U = 0.5*(u0 + u)
        F1 = rho*(1/k)*inner(v, u - u0)*dx \
            + rho*inner(v, grad(u0)*(u0 - w))*dx \
            + inner(epsilon(v), sigma(U, p0))*dx \
            - inner(v, g)*ds \
            - inner(v, f)*dx \
            + inner(v, p0*n)*ds \
            - mu*inner(grad(U).T*n, v)*ds
        # MER: I don't like these. Yes, I know about the swirl.
        # GB: They seem to be necessary for the channel with the flap so the two terms
##                    + inner(v, p0*n)*ds \
##            - mu*inner(grad(U).T*n, v)*ds
##            have been added i again.

        a1 = lhs(F1)
        L1 = rhs(F1)

        # Pressure correction
        a2 = inner(grad(q), k*grad(p))*dx
        L2 = inner(grad(q), k*grad(p0))*dx - q*rho*div(u1)*dx

        # Add alternative using proper constraint
        QR = Q*R
        q_r, s_r = TestFunctions(QR)
        p_r, r_r = TrialFunctions(QR)
        self.QR = QR

        a2_r = inner(grad(q_r), k*grad(p_r))*dx + r_r*q_r*dx + p_r*s_r*dx
        L2_r = inner(grad(q_r), k*grad(p0))*dx - q_r*rho*div(u1)*dx

        # Velocity correction
        a3 = inner(v, rho*u)*dx
        L3 = inner(v, rho*u1)*dx + inner(v, k*grad(p0 - p1))*dx

        # Create solvers
        #solver1 = LUSolver()
        #solver2 = LUSolver()
        #solver3 = LUSolver()
        solver1 = KrylovSolver("gmres", "ilu")
        solver2 = KrylovSolver("gmres", "amg")
        solver3 = KrylovSolver("gmres", "ilu")
        solver1.parameters["relative_tolerance"] = 1e-14
        solver2.parameters["relative_tolerance"] = 1e-14
        solver3.parameters["relative_tolerance"] = 1e-14

        # Store variables needed for time-stepping
        self.dt = dt
        self.k = k
        self.t_range = t_range
        self.bcu = bcu
        self.bcp = bcp
        self.f = f
        self.u0 = u0
        self.u1 = u1
        self.p0 = p0
        self.p1 = p1
        self.L1 = L1
        self.L2 = L2
        self.L2_r = L2_r
        self.L3 = L3
        self.a1 = a1
        self.a2 = a2
        self.a2_r = a2_r
        self.a3 = a3
        self.solver1 = solver1
        self.solver2 = solver2
        self.solver3 = solver3

        # Empty file handlers / time series
        self.velocity_file = None
        self.pressure_file = None
        self.velocity_series = None
        self.pressure_series = None

        # Assemble matrices
        self.reassemble()

    def solve(self):
        "Solve problem and return computed solution (u, p)"

        # Time loop
        for t in self.t_range:

            # Solve for current time step
            self.step(self.dt)

            # Update
            self.update(t)
            self._end_time_step(t, self.t_range[-1])

        return self.u1, self.p1

    def step(self, dt):
        "Compute solution for new time step"

        # Check if we need to reassemble
        if not dt == self.dt:
            info("Using actual timestep: %g" % dt)
            self.dt = dt
            self.k.assign(dt)
            self.reassemble()

        # Compute tentative velocity step
        begin("Computing tentative velocity")
        b = assemble(self.L1)
        [bc.apply(self.A1, b) for bc in self.bcu]
        self.solver1.solve(self.A1, self.u1.vector(), b)
        end()

        # Pressure correction
        begin("Computing pressure correction")
        if self.parameters["zero_average_pressure"]:
            info_red("Using L2 average constraint")
            b = assemble(self.L2_r)
            qr = Function(self.QR)
            solve(self.A2_r, qr.vector(), b)
            self.p1.assign(qr.split()[0])
        else:
            b = assemble(self.L2)

            if len(self.bcp) == 0 or is_periodic(self.bcp):
                normalize(b)

            [bc.apply(self.A2, b) for bc in self.bcp]
            if is_periodic(self.bcp):
                solve(self.A2, self.p1.vector(), b)
            else:
                self.solver2.solve(self.A2, self.p1.vector(), b)
            if len(self.bcp) == 0 or is_periodic(self.bcp):
                normalize(self.p1.vector())
        end()

        # Velocity correction
        begin("Computing velocity correction")
        b = assemble(self.L3)
        [bc.apply(self.A3, b) for bc in self.bcu]
        self.solver3.solve(self.A3, self.u1.vector(), b)
        end()

        return self.u1, self.p1

    def update(self, t):

        # Update the time on the body force
        self.f.t = t

        # Propagate values
        self.u0.assign(self.u1)
        self.p0.assign(self.p1)

        # Plot solution
        if self.parameters["plot_solution"]:
            plot(self.p1, title="Pressure", rescale=True)
            plot(self.u1, title="Velocity", rescale=True)

        # Store solution (for plotting)
        if self.parameters["save_solution"]:
            if self.velocity_file is None: self.velocity_file = File("velocity.pvd")
            if self.pressure_file is None: self.pressure_file = File("pressure.pvd")
            self.velocity_file << self.u1
            self.pressure_file << self.p1

        # Store solution data
        if self.parameters["store_solution_data"]:
            if self.velocity_series is None:
                self.velocity_series = TimeSeries("velocity")
            if self.pressure_series is None:
                self.pressure_series = TimeSeries("pressure")
            self.velocity_series.store(self.u1.vector(), t)
            self.pressure_series.store(self.p1.vector(), t)

        return self.u1, self.p1

    def reassemble(self):
        "Reassemble matrices, needed when mesh or time step has changed"
        info("(Re)assembling matrices")
        self.A1 = assemble(self.a1)
        self.A2 = assemble(self.a2)
        self.A2_r = assemble(self.a2_r)
        self.A3 = assemble(self.a3)
        self.b1 = assemble(self.L1)
        self.b2 = assemble(self.L2)
        self.b2_r = assemble(self.L2_r)
        self.b3 = assemble(self.L3)

    def solution(self):
        "Return current solution values"
        return self.u1, self.p1

    def solution_values(self):
        "Return solution values at t_{n-1} and t_n"
        return self.u0, self.u1, self.p0, self.p1
