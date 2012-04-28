__author__ = "Marie E. Rognes"
__copyright__ = "Copyright (C) 2012 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

# Last changed: 2012-04-28

__all__ = ["TaylorHoodSolver"]

from dolfin import *
from cbc.common.utils import *
from cbc.common import *

class TaylorHoodSolver(CBCSolver):
    """Navier-Stokes solver using a plain saddle point
    formulation. This should be ridiculously robust. No boundary
    forces allowed."""

    def __init__(self, problem):
        "Initialize Navier-Stokes solver"

        # Initialize base class
        CBCSolver.__init__(self)

        # Set up parameters
        self.parameters = Parameters("solver_parameters")
        self.parameters.add("plot_solution", False)
        self.parameters.add("save_solution", False)
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
        W = MixedFunctionSpace([V, Q, R])

        # Coefficients
        mu = Constant(problem.viscosity())  # Dynamic viscosity [Ps x s]
        rho = Constant(problem.density())   # Density [kg/m^3]
        n = FacetNormal(mesh)
        k = Constant(dt)
        f = problem.body_force(V1)
        w = problem.mesh_velocity(V1)

        # If no body forces are specified, assume it is 0
        if f == []:
            f = Constant((0,)*V1.mesh().geometry().dim())

        # Create boundary conditions
        bcu = create_dirichlet_conditions(problem.velocity_dirichlet_values(),
                                          problem.velocity_dirichlet_boundaries(),
                                          V)

        # Create initial conditions
        u0 = create_initial_condition(problem.velocity_initial_condition(), V)
        u0 = interpolate(u0, V)

        # Create initial function
        upr0 = Function(W)
        upr0.vector()[:V.dim()] = u0.vector()

        W0 = W.sub(0).collapse()
        W1 = W.sub(1).collapse()

        # Create function for solution at previous time
        upr_ = Function(W)
        upr_.assign(upr0)
        (u_, p_, r_) = split(upr_)
        u0 = Function(W0)
        p0 = Function(W1)

        # Test and trial functions
        upr = Function(W)
        (u, p, r) = split(upr)
        (v, q, s) = TestFunctions(W)
        u1 = Function(W0)
        p1 = Function(W1)

        # Define Cauchy stress tensor
        def sigma(v, p):
            return 2.0*mu*sym(grad(v))  - p*Identity(v.cell().d)

        # Mixed formulation
        U = 0.5*(u_ + u)
        F = (rho*(1/k)*inner(u - u_, v)*dx
             + rho*inner(grad(U)*(U - w), v)*dx
             + inner(sigma(U, p), sym(grad(v)))*dx
             + div(U)*q*dx
             - inner(f, v)*dx
             + p*s*dx + q*r*dx)

        # Store variables needed for time-stepping
        self.mesh_velocity = w
        self.W = W
        self.dt = dt
        self.k = k
        self.t_range = t_range
        self.bcu = bcu
        self.f = f
        self.upr_ = upr_
        self.upr = upr
        self.u0 = u0
        self.u1 = u1
        self.p0 = p0
        self.p1 = p1
        self.F = F

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

        #plot(self.mesh_velocity, mesh=self.W.mesh(),
        #     title="mesh_velocity in step")

        # Always do this
        self.dt = dt
        self.k.assign(dt)
        self.reassemble()

        # Compute solution
        begin("Computing velocity and pressure and multiplier")
        solve(self.F == 0, self.upr, self.bcu)
        self.u1.assign(self.upr.split()[0])
        self.p1.assign(self.upr.split()[1])
        end()

        return (self.u1, self.p1)

    def update(self, t):

        # This is hardly robust
        # Update the time on the body force
        self.f.t = t

        # Propagate values
        self.upr_.assign(self.upr)
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
            if self.series is None:
                self.series = TimeSeries("velocity-pressure-multiplier")
            self.series.store(self.upr.vector(), t)

        return self.u1, self.p1

    def reassemble(self):
        "Reassemble matrices, needed when mesh or time step has changed"
        info("(Re)assembling matrices")
        info("No action taken here in this solver")

    def solution(self):
        "Return current solution values"
        return self.u1, self.p1

    def solution_values(self):
        "Return solution values at t_{n-1} and t_n"
        return (self.u0, self.u1, self.p0, self.p1)
