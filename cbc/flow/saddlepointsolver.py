__author__ = "Marie E. Rognes"
__copyright__ = "Copyright (C) 2012 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

# Last changed: 2012-05-01

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
        zero_average_pressure = False

        # Get mesh and time step range
        mesh = problem.mesh()
        dt, t_range = timestep_range_cfl(problem, mesh)
        info("Using time step dt = %g" % dt)

        # Function spaces
        V1 = VectorFunctionSpace(mesh, "CG", 1)
        V = VectorFunctionSpace(mesh, "CG", 2)
        Q = FunctionSpace(mesh, "CG", 1)

        if zero_average_pressure:
            R = FunctionSpace(mesh, "R", 0)
            W = MixedFunctionSpace([V, Q, R])
        else:
            W = V*Q

        # Coefficients
        mu = Constant(problem.viscosity())  # Dynamic viscosity [Ps x s]
        rho = Constant(problem.density())   # Density [kg/m^3]
        n = FacetNormal(mesh)
        k = Constant(dt)
        f = problem.body_force(V1)
        g = problem.boundary_traction(V1)
        w = problem.mesh_velocity(V1)

        # If no body forces are specified, assume it is 0
        if f == []:
            f = Constant((0,)*V1.mesh().geometry().dim())
        if g == []:
            g = Constant((0,)*V1.mesh().geometry().dim())

        # Create boundary conditions
        bcu = create_dirichlet_conditions(problem.velocity_dirichlet_values(),
                                          problem.velocity_dirichlet_boundaries(),
                                          W.sub(0))

        # Allow this just to be able to set all values directly
        bcp = create_dirichlet_conditions(problem.pressure_dirichlet_values(),
                                          problem.pressure_dirichlet_boundaries(),
                                          W.sub(1))

        # Create initial conditions
        u0 = create_initial_condition(problem.velocity_initial_condition(), V)
        u0 = interpolate(u0, V)

        p0 = create_initial_condition(problem.pressure_initial_condition(), Q)
        p0 = interpolate(p0, Q)

        # Create initial function
        upr0 = Function(W)
        upr0.vector()[:V.dim()] = u0.vector()
        upr0.vector()[V.dim():V.dim()+Q.dim()] = p0.vector()

        # Create function for solution at previous time
        upr_ = Function(W)
        upr_.assign(upr0)
        if zero_average_pressure:
            (u_, p_, r_) = split(upr_)
        else:
            (u_, p_) = split(upr_)
        #u0 = Function(V)
        #p0 = Function(Q)

        # Test and trial functions
        upr = Function(W)
        if zero_average_pressure:
            (u, p, r) = split(upr)
            (v, q, s) = TestFunctions(W)
        else:
            (u, p) = split(upr)
            (v, q) = TestFunctions(W)
        u1 = Function(V)
        p1 = Function(Q)

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
             - inner(g, v)*ds)

        if zero_average_pressure:
            F += p*s*dx + q*r*dx

        # Store variables needed for time-stepping
        self.mesh_velocity = w
        self.W = W
        self.dt = dt
        self.k = k
        self.t_range = t_range
        self.bcu = bcu
        self.bcp = bcp
        self.f = f
        self.g = g
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

        # Always do this
        self.dt = dt
        self.k.assign(dt)
        self.reassemble()

        # Allow pressure boundary conditions for debugging
        bcs = self.bcu
        if self.bcp != []:
            info_green("Including pressure DirichletBC at your risk")
            bcs += self.bcp

        # Compute solution
        begin("Computing velocity and pressure and multiplier")
        solve(self.F == 0, self.upr, bcs)
        self.u1.assign(self.upr.split()[0])
        self.p1.assign(self.upr.split()[1])
        end()

        return (self.u1, self.p1)

    def update(self, t):

        # This is hardly robust
        # Update the time on the body force
        self.f.t = t
        self.g.t = t

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
