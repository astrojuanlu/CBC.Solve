__author__ = "Kristian Valen-Sendstad and Anders Logg"
__copyright__ = "Copyright (C) 2009 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

# Last changed: 2010-06-30

__all__ = ["NavierStokesSolver", "NavierStokesDualSolver"]

from dolfin import *
from cbc.common.utils import *
from cbc.common import CBCSolver

class NavierStokesSolver(CBCSolver):
    "Navier-Stokes solver"

    def __init__(self, problem):
        "Initialize Navier-Stokes solver"

        # Initialize base class
        CBCSolver.__init__(self)

        # Set up parameters
        self.parameters = Parameters("solver_parameters")
        self.parameters.add("plot_solution", True)        # Plot when running
        self.parameters.add("save_solution", True)        # Store solution for later plotting
        self.parameters.add("store_solution_data", False) # Store solution data in binary format

        # Get mesh and time step range
        mesh = problem.mesh()
        dt, t_range = timestep_range_cfl(problem, mesh)
        info("Using time step dt = %g" % dt)

        # Function spaces
        V1 = VectorFunctionSpace(mesh, "CG", 1)
        V = VectorFunctionSpace(mesh, "CG", 2)
        Q = FunctionSpace(mesh, "CG", 1)

        # Initial and boundary conditions
        u0, p0 = problem.initial_conditions(V, Q)
        bcu, bcp = problem.boundary_conditions(V, Q)

        # Test and trial functions
        v = TestFunction(V)
        q = TestFunction(Q)
        u = TrialFunction(V)
        p = TrialFunction(Q)

        # Functions
        u0 = interpolate(u0, V)
        u1 = interpolate(u0, V)
        p0 = interpolate(p0, Q)
        p1 = interpolate(p0, Q)

        # Coefficients
        #nu = Constant(problem.viscosity()) # Kinematic viscosity [m^2/s]
        mu = Constant(problem.viscosity())  # Dynamic viscosity [Ps x s]
        rho = Constant(problem.density())   # Density [kg/m^3]
        k = Constant(dt)
        f = problem.body_force(V1)
        w = problem.mesh_velocity(V1)

        # Tentative velocity step
        U = 0.5*(u0 + u)
        F1 = rho*(1/k)*inner(v, u - u0)*dx + rho*inner(v, grad(u0)*(u0 - w))*dx \
            + mu*inner(grad(v), grad(U))*dx + inner(v, grad(p0))*dx \
            - inner(v, f)*dx
        a1 = lhs(F1)
        L1 = rhs(F1)

        # Pressure correction
        a2 = inner(grad(q), k*grad(p))*dx
        L2 = inner(grad(q), k*grad(p0))*dx - q*div(u1)*dx

        # Velocity correction
        a3 = inner(v, u)*dx
        L3 = inner(v, u1)*dx + inner(v, k*grad(p0 - p1))*dx

        # Store variables needed for time-stepping
        self.dt = dt
        self.t_range = t_range
        self.bcu = bcu
        self.bcp = bcp
        self.u0 = u0
        self.u1 = u1
        self.p0 = p0
        self.p1 = p1
        self.L1 = L1
        self.L2 = L2
        self.L3 = L3
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3

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

        # FIXME: Need to update time step here and reassemble

        # Compute tentative velocity step
        begin("Computing tentative velocity")
        b = assemble(self.L1)
        [bc.apply(self.A1, b) for bc in self.bcu]
        solve(self.A1, self.u1.vector(), b, "gmres", "ilu")
        end()

        # Pressure correction
        begin("Computing pressure correction")
        b = assemble(self.L2)
        if len(self.bcp) == 0 or is_periodic(self.bcp): normalize(b)
        [bc.apply(self.A2, b) for bc in self.bcp]
        if is_periodic(self.bcp):
            solve(self.A2, self.p1.vector(), b)
        else:
            solve(self.A2, self.p1.vector(), b, 'gmres', 'amg_hypre')
        if len(self.bcp) == 0 or is_periodic(self.bcp): normalize(self.p1.vector())
        end()

        # Velocity correction
        begin("Computing velocity correction")
        b = assemble(self.L3)
        [bc.apply(self.A3, b) for bc in self.bcu]
        solve(self.A3, self.u1.vector(), b, "gmres", "ilu")
        end()

        return self.u1, self.p1

    def update(self, t):

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
            if self.velocity_series is None: self.velocity_series = TimeSeries("velocity")
            if self.pressure_series is None: self.pressure_series = TimeSeries("pressure")
            self.velocity_series.store(self.u1.vector(), t)
            self.pressure_series.store(self.p1.vector(), t)

        return self.u1, self.p1

    def reassemble(self):
        "Reassemble matrices, needed when mesh or time step has changed"
        info("(Re)assembling matrices")
        self.A1 = assemble(self.a1)
        self.A2 = assemble(self.a2)
        self.A3 = assemble(self.a3)
        self.b1 = assemble(self.L1)
        self.b2 = assemble(self.L2)
        self.b3 = assemble(self.L3)

class NavierStokesDualSolver(CBCSolver):
    "Navier-Stokes dual solver"

    def __init__(self, problem):
        CBCSolver.__init__(self)

        # Set up parameters
        self.parameters = Parameters("solver_parameters")
        self.parameters.add("plot_solution", True)        # Plot when running
        self.parameters.add("save_solution", True)        # Store solution for later plotting
        self.parameters.add("store_solution_data", False) # Store solution data in binary format

        # Load primal solutions
        self.velocity_series = TimeSeries("velocity")
        self.pressure_series = TimeSeries("pressure")

        # Get mesh and time step range
        mesh = problem.mesh()
        n = FacetNormal(mesh)
        dt, t_range = timestep_range(problem, mesh)
        info("Using time step dt = %g" % dt)

        # Function spaces
        V1 = VectorFunctionSpace(mesh, "CG", 1)
        V = VectorFunctionSpace(mesh, "CG", 2)
        Q = FunctionSpace(mesh, "CG", 1)

        # Test and trial functions
        system = MixedFunctionSpace([V, Q])
        (v, q) = TestFunctions(system)
        (w, r) = TrialFunctions(system)

        # Functions for plotting without opening multiple windows
        w_plot = Function(V)
        r_plot = Function(Q)

        # Initial and boundary conditions (which are homogenised
        # versions of the primal conditions)

        # FIXME: Initial conditions should not be 0 and should depend
        # instead on the goal
        w1 = project(Constant((0.0, 0.0)), V)
        r1 = project(Constant(0.0), Q)

        bcw, bcr = problem.boundary_conditions(V, Q)

        # Functions
        w0 = Function(V)
        r0 = Function(Q)
        u_h = Function(V)
        p_h = Function(Q)

        # Coefficients
        nu = Constant(problem.viscosity())
        f = problem.body_force(V1)

        # Dual forms
        sigma = nu*(grad(v) + grad(v).T) - q*Identity(v.cell().d)
        a_tilde = inner(grad(u_h)*v + grad(v)*u_h, w)*dx \
            + inner(sigma, 0.5*(grad(w) + grad(w).T))*dx \
            - nu*inner((grad(v) + grad(v).T)*n, w)*ds \
            + div(v)*r*dx

        a = inner(v, w)*dx + dt*a_tilde

        goal = problem.functional(v, q, V, Q, n)
        L = inner(v, w1)*dx + dt*goal

        # Store variables needed for time-stepping
        self.dt = dt
        self.t_range = t_range
        self.bcw = bcw
        self.bcr = bcr
        self.w0 = w0
        self.w1 = w1
        self.r0 = r0
        self.r1 = r1
        self.L = L
        self.a = a
        self.u_h = u_h
        self.p_h = p_h
        self.w_plot = w_plot
        self.r_plot = r_plot
        self.exterior_facet_domains = problem.boundary_markers()

        # Empty file handlers / time series
        self.dual_velocity_file = None
        self.dual_pressure_file = None
        self.dual_velocity_series = None
        self.dual_pressure_series = None

    def solve(self):
        "Solve problem and return computed solution (u, p)"

        # Time loop
        for t in reversed(self.t_range):

            # Load primal solution at the current time step
            self.velocity_series.retrieve(self.u_h.vector(), t)
            self.pressure_series.retrieve(self.p_h.vector(), t)

            # Solve for current time step
            self.step(self.dt)

            # Update
            self.update()
            self._end_time_step(t, self.t_range[-1])

        return self.w0, self.r0

    def step(self, dt):

        # Compute dual solution
        begin("Computing dual solution")
        pde_dual = VariationalProblem(self.a, self.L, self.bcw + self.bcr, exterior_facet_domains = self.exterior_facet_domains)
        (self.w0, self.r0) = pde_dual.solve().split(True)
        end()

        return self.w0, self.r0

    def update(self):

        # Propagate values
        self.w1.assign(self.w0)
        self.r1.assign(self.r0)

        # Plot solution
        if self.parameters["plot_solution"]:
            # Copy to a fixed function to trick Viper into not opening
            # up multiple windows
            self.w_plot.assign(self.w0)
            self.r_plot.assign(self.r0)
            plot(self.r_plot, title="Pressure", rescale=True)
            plot(self.w_plot, title="Velocity", rescale=True)

        # Store solution (for plotting)
        if self.parameters["save_solution"]:
            if self.dual_velocity_file is None: self.dual_velocity_file = File("dual_velocity.pvd")
            if self.dual_pressure_file is None: self.dual_pressure_file = File("dual_pressure.pvd")
            self.dual_velocity_file << self.w0
            self.dual_pressure_file << self.r0

        # Store solution data
        if self.parameters["store_solution_data"]:
            if self.dual_velocity_series is None: self.velocity_series = TimeSeries("dual_velocity")
            if self.dual_pressure_series is None: self.pressure_series = TimeSeries("dual_pressure")
            self.dual_velocity_series.store(self.w0.vector(), t)
            self.dual_pressure_series.store(self.r0.vector(), t)

        return self.w0, self.r0
