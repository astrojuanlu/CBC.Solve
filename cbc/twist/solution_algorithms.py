__author__ = "Harish Narayanan"
__copyright__ = "Copyright (C) 2009 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

from dolfin import *
from cbc.common import CBCSolver
from cbc.common.utils import *
from cbc.twist.kinematics import Grad, DeformationGradient
from sys import exit
from numpy import array, loadtxt

class StaticMomentumBalanceSolver(CBCSolver):
    "Solves the static balance of linear momentum"

    def __init__(self, problem):
        """Initialise the static momentum balance solver"""

        # Set up parameters
        self.parameters = Parameters("solver_parameters")
        self.parameters.add("plot_solution", True)
        self.parameters.add("save_solution", False)
        self.parameters.add("store_solution_data", False)

        # Get problem parameters
        mesh = problem.mesh()

        # Define function spaces
        vector = VectorFunctionSpace(mesh, "CG", 1)

        # Get Dirichlet boundary conditions on the displacement field
        bcu = []

        dirichlet_conditions = problem.dirichlet_conditions()
        dirichlet_boundaries = problem.dirichlet_boundaries()

        if len(dirichlet_conditions) != len(problem.dirichlet_boundaries()):
            print "Please make sure the number of your Dirichlet conditions match the number of your Dirichlet boundaries"
            exit(2)

        for (i, dirichlet_condition) in enumerate(dirichlet_conditions):
            bcu.append(DirichletBC(vector, dirichlet_condition, \
            compile_subdomains(dirichlet_boundaries[i])))

        # Define fields
        # Test and trial functions
        v = TestFunction(vector)
        u = Function(vector)
        du = TrialFunction(vector)

        # Driving forces
        B = problem.body_force()

        # If no body forces are specified, assume it is 0
        if B == []:
            B = Constant((0,)*vector.mesh().geometry().dim())

        # First Piola-Kirchhoff stress tensor based on the material
        # model
        P  = problem.first_pk_stress(u)

        # The variational form corresponding to hyperelasticity
        L = inner(P, Grad(v))*dx - inner(B, v)*dx

        # Add contributions to the form from the Neumann boundary
        # conditions

        # Get Neumann boundary conditions on the stress
        neumann_conditions = problem.neumann_conditions()

        # If no Neumann conditions are specified, assume it is 0
        if neumann_conditions == []:
            neumann_conditions = Constant((0,)*vector.mesh().geometry().dim())

        neumann_boundaries = problem.neumann_boundaries()

        boundary = MeshFunction("uint", mesh, mesh.topology().dim() - 1)
        boundary.set_all(len(neumann_boundaries) + 1)

        for (i, neumann_boundary) in enumerate(neumann_boundaries):
            compiled_boundary = compile_subdomains(neumann_boundary)
            compiled_boundary.mark(boundary, i)
            L = L - inner(neumann_conditions[i], v)*ds(i)

        a = derivative(L, u, du)

        # Setup problem
        equation = VariationalProblem(a, L, bcu, exterior_facet_domains = boundary, nonlinear = True)
        equation.parameters["newton_solver"]["absolute_tolerance"] = 1e-12
        equation.parameters["newton_solver"]["relative_tolerance"] = 1e-16
        equation.parameters["newton_solver"]["maximum_iterations"] = 100

        # Store variables needed for time-stepping
        # FIXME: Figure out why I am needed
        self.mesh = mesh
        self.equation = equation
        self.u = u

    def solve(self):
        """Solve the mechanics problem and return the computed
        displacement field"""

        # Solve problem
        self.equation.solve(self.u)

        # Plot solution
        if self.parameters["plot_solution"]:
            plot(self.u, title="Displacement", mode="displacement", rescale=True)
            interactive()

        # Store solution (for plotting)
        if self.parameters["save_solution"]:
            displacement_file = File("displacement.pvd")
            displacement_file << self.u

        # Store solution data
        if self.parameters["store_solution_data"]:
            displacement_series = TimeSeries("displacement")
            displacement_series.store(self.u, 0.0)

        return self.u

class MomentumBalanceSolver(CBCSolver):
    "Solves the quasistatic/dynamic balance of linear momentum"

    def __init__(self, problem):

        """Initialise the momentum balance solver"""

        # Set up parameters
        self.parameters = Parameters("solver_parameters")
        self.parameters.add("plot_solution", True)
        self.parameters.add("save_solution", False)
        self.parameters.add("store_solution_data", False)

        # Get problem parameters
        mesh        = problem.mesh()
        dt, t_range = timestep_range(problem, mesh)
        end_time    = problem.end_time()

        # Define function spaces
        scalar = FunctionSpace(mesh, "CG", 1)
        vector = VectorFunctionSpace(mesh, "CG", 1)

        # Get initial conditions
        u0, v0 = problem.initial_conditions()

        # If no initial conditions are specified, assume they are 0
        if u0 == []:
            u0 = Constant((0,)*vector.mesh().geometry().dim())
        if v0 == []:
            v0 = Constant((0,)*vector.mesh().geometry().dim())

        # If either are text strings, assume those are file names and
        # load conditions from those files
        if isinstance(u0, str):
            print "Loading initial displacement from file"
            file_name = u0
            u0 = Function(vector)
            u0.vector()[:] = loadtxt(file_name)[:]
        if isinstance(v0, str):
            print "Loading initial velocity from file"
            file_name = v0
            v0 = Function(vector)
            v0.vector()[:] = loadtxt(file_name)[:]

        # Get Dirichlet boundary conditions on the displacement field
        bcu = []

        dirichlet_conditions = problem.dirichlet_conditions()
        dirichlet_boundaries = problem.dirichlet_boundaries()

        if len(dirichlet_conditions) != len(problem.dirichlet_boundaries()):
            print "Please make sure the number of your Dirichlet conditions match the number of your Dirichlet boundaries"
            exit(2)

        for (i, dirichlet_condition) in enumerate(dirichlet_conditions):
            print "Applying Dirichlet boundary condition at", dirichlet_boundaries[i]
            bcu.append(DirichletBC(vector, dirichlet_condition, \
            compile_subdomains(dirichlet_boundaries[i])))

        # Driving forces
        B  = problem.body_force()

        # If no body forces are specified, assume it is 0
        if B == []:
            B = Constant((0,)*vector.mesh().geometry().dim())

        # Define fields
        # Test and trial functions
        v  = TestFunction(vector)
        u1 = Function(vector)
        v1 = Function(vector)
        a1 = Function(vector)
        du = TrialFunction(vector)

        # Initial displacement and velocity
        u0 = interpolate(u0, vector)
        v0 = interpolate(v0, vector)
        v1 = interpolate(v0, vector)

        # Parameters pertinent to (HHT) time integration
        # alpha = 1.0
        beta = 0.25
        gamma = 0.5

        # Determine initial acceleration
        a0 = TrialFunction(vector)
        P0 = problem.first_pk_stress(u0)
        a_accn = inner(a0, v)*dx
        L_accn = - inner(P0, Grad(v))*dx + inner(B, v)*dx

        # Add contributions to the form from the Neumann boundary
        # conditions

        # Get Neumann boundary conditions on the stress
        neumann_conditions = problem.neumann_conditions()

        # If no Neumann conditions are specified, assume it is 0
        if neumann_conditions == []:
            neumann_conditions = Constant((0,)*vector.mesh().geometry().dim())

        neumann_boundaries = problem.neumann_boundaries()

        boundary = MeshFunction("uint", mesh, mesh.topology().dim() - 1)
        boundary.set_all(len(neumann_boundaries) + 1)

        for (i, neumann_boundary) in enumerate(neumann_boundaries):
            compiled_boundary = compile_subdomains(neumann_boundary)
            compiled_boundary.mark(boundary, i)
            L_accn = L_accn + inner(neumann_conditions[i], v)*ds(i)

        problem_accn = VariationalProblem(a_accn, L_accn, exterior_facet_domains = boundary)
        a0 = problem_accn.solve()

        k = Constant(dt)
        a1 = a0*(1.0 - 1.0/(2*beta)) - (u0 - u1 + k*v0)/(beta*k**2)

        # Get reference density
        rho0 = problem.reference_density()

        # If no reference density is specified, assume it is 1.0
        if rho0 == []:
            rho0 = Constant(1.0)

        density_type = str(rho0.__class__)
        if not ("dolfin" in density_type):
            print "Converting given density to a DOLFIN Constant"
            rho0 = Constant(rho0)

        # Piola-Kirchhoff stress tensor based on the material model
        P = problem.first_pk_stress(u1)

#         # FIXME: A general version of the trick below is what should
#         # be used instead. The commentend-out lines only work well for
#         # quadratically nonlinear models, e.g. St. Venant Kirchhoff.

#         # S0 = problem.second_pk_stress(u0)
#         # S1 = problem.second_pk_stress(u1)
#         # Sm = 0.5*(S0 + S1)
#         # Fm = DeformationGradient(0.5*(u0 + u1))
#         # P  = Fm*Sm

        # The variational form corresponding to hyperelasticity
        L = int(problem.is_dynamic())*rho0*inner(a1, v)*dx \
        + inner(P, Grad(v))*dx - inner(B, v)*dx

        # Add contributions to the form from the Neumann boundary
        # conditions

        # Get Neumann boundary conditions on the stress
        neumann_conditions = problem.neumann_conditions()
        neumann_boundaries = problem.neumann_boundaries()

        boundary = MeshFunction("uint", mesh, mesh.topology().dim() - 1)
        boundary.set_all(len(neumann_boundaries) + 1)

        for (i, neumann_boundary) in enumerate(neumann_boundaries):
            print "Applying Neumann boundary condition at", neumann_boundary
            compiled_boundary = compile_subdomains(neumann_boundary)
            compiled_boundary.mark(boundary, i)
            L = L - inner(neumann_conditions[i], v)*ds(i)

        a = derivative(L, u1, du)

        # Store variables needed for time-stepping
        self.dt = dt
        self.t_range = t_range
        self.end_time = end_time
        self.a = a
        self.L = L
        self.bcu = bcu
        self.u0 = u0
        self.v0 = v0
        self.a0 = a0
        self.u1 = u1
        self.v1 = v1
        self.a1 = a1
        self.k  = k
        self.beta = beta
        self.gamma = gamma
        self.vector = vector
        self.B = B
        self.dirichlet_conditions = dirichlet_conditions
        self.neumann_conditions = neumann_conditions
        self.boundary = boundary

        # FIXME: Figure out why I am needed
        self.mesh = mesh
        self.t = 0

        # Empty file handlers / time series
        self.displacement_file = None
        self.velocity_file = None
        self.displacement_series = None
        self.velocity_series = None        

    def solve(self):
        """Solve the mechanics problem and return the computed
        displacement field"""

        # Time loop
        for t in self.t_range:
            print "Solving the problem at time t = " + str(self.t)
            self.step(self.dt)
            self.update()

        if self.parameters["plot_solution"]:
            interactive()

    def step(self, dt):
        """Setup and solve the problem at the current time step"""

        # FIXME: Setup all stuff in the constructor and call assemble instead of VariationalProblem
        equation = VariationalProblem(self.a, self.L, self.bcu, exterior_facet_domains = self.boundary, nonlinear = True)
        equation.parameters["newton_solver"]["absolute_tolerance"] = 1e-12
        equation.parameters["newton_solver"]["relative_tolerance"] = 1e-12
        equation.parameters["newton_solver"]["maximum_iterations"] = 100
        equation.solve(self.u1)
        return self.u1

    def update(self):
        """Update problem at time t"""

        # Compute new accelerations and velocities based on new
        # displacement
        a1 = self.a0*(1.0 - 1.0/(2*self.beta)) \
            - (self.u0 - self.u1 + self.k*self.v0)/(self.beta*self.k**2)
        self.a1 = project(a1, self.vector)
        v1 = self.v0 + self.k*((1 - self.gamma)*self.a1 + self.gamma*self.a0)
        self.v1 = project(v1, self.vector)

        # Propogate the displacements, velocities and accelerations
        self.u0.assign(self.u1)
        self.v0.assign(self.v1)
        self.a0.assign(self.a1)

        # Plot solution
        if self.parameters["plot_solution"]:
            plot(self.u0, title="Displacement", mode="displacement", rescale=True)

        # Store solution (for plotting)
        if self.parameters["save_solution"]:
            if self.displacement_file is None: self.displacement_file = File("displacement.pvd")
            if self.velocity_file is None: self.velocity_file = File("velocity.pvd")
            self.displacement_file << self.u0
            self.velocity_file << self.v0

        # Store solution data
        if self.parameters["store_solution_data"]:
            if self.displacement_series is None: self.displacement_series = TimeSeries("displacement")
            if self.velocity_series is None: self.velocity_series = TimeSeries("velocity")
            self.displacement_series.store(self.u0, self.t)
            self.velocity_series.store(self.v0, self.t)

        # Move to next time step
        self.t = self.t + self.dt

        # Inform time-dependent functions of new time
        for bc in self.dirichlet_conditions:
            bc.t = self.t
        for bc in self.neumann_conditions:
            bc.t = self.t
        self.B.t = self.t

class CG1MomentumBalanceSolver(CBCSolver):
    """Solves the dynamic balance of linear momentum using a CG1
    time-stepping scheme"""

    def __init__(self, problem):

        """Initialise the momentum balance solver"""

        # Set up parameters
        self.parameters = Parameters("solver_parameters")
        self.parameters.add("plot_solution", False)
        self.parameters.add("save_solution", False)
        self.parameters.add("save_plot", False)

        # Create binary files to store solutions
        if self.parameters["save_solution"]:
            self.displacement_velocity_series = TimeSeries("displacement_velocity")

        # Create pvd files to store paraview plots
        if self.parameters["save_plot"]:
            self.displacement_plot_file = File("displacement.pvd")
            self.velocity_plot_file = File("velocity.pvd")

        # Get problem parameters
        mesh        = problem.mesh()
        dt, t_range = timestep_range(problem, mesh)
        end_time    = problem.end_time()
        info("Using time step dt = %g" % dt)

        # Define function spaces
        scalar = FunctionSpace(mesh, "CG", 1)
        vector = VectorFunctionSpace(mesh, "CG", 1)

        mixed_element = MixedFunctionSpace([vector, vector])
        V = TestFunction(mixed_element)
        dU = TrialFunction(mixed_element)
        U = Function(mixed_element)
        U0 = Function(mixed_element)

        # Get initial conditions
        u0, v0 = problem.initial_conditions()

        # If no initial conditions are specified, assume they are 0
        if u0 == []:
            u0 = Constant((0,)*vector.mesh().geometry().dim())
        if v0 == []:
            v0 = Constant((0,)*vector.mesh().geometry().dim())

        # If either are text strings, assume those are file names and
        # load conditions from those files
        if isinstance(u0, str):
            print "Loading initial displacement from file"
            file_name = u0
            _u0 = loadtxt(file_name)[:]
            U0.vector()[0:len(_u0)] = _u0[:]
        if isinstance(v0, str):
            print "Loading initial velocity from file"
            file_name = v0
            _v0 = loadtxt(file_name)[:]
            U0.vector()[len(_v0) + 1:2*len(_v0) - 1] = _v0[:]

        # Get Dirichlet boundary conditions on the displacement field
        bcu = []

        dirichlet_conditions = problem.dirichlet_conditions()
        dirichlet_boundaries = problem.dirichlet_boundaries()

        if len(dirichlet_conditions) != len(problem.dirichlet_boundaries()):
            print "Please make sure the number of your Dirichlet conditions match the number of your Dirichlet boundaries"
            exit(2)

        for (i, dirichlet_condition) in enumerate(dirichlet_conditions):
            print "Applying Dirichlet boundary condition at", dirichlet_boundaries[i]
            bcu.append(DirichletBC(vector, dirichlet_condition, \
            compile_subdomains(dirichlet_boundaries[i])))

        # Driving forces
        B  = problem.body_force()

        # If no body forces are specified, assume it is 0
        if B == []:
            B = Constant((0,)*vector.mesh().geometry().dim())

        # Functions
        xi, eta = split(V)
        u, v = split(U)
        u0, v0 = split(U0)

        # Evaluate displacements and velocities at mid points
        u_mid = 0.5*(u0 + u)
        v_mid = 0.5*(v0 + v)

        # Get reference density
        rho0 = problem.reference_density()

        # If no reference density is specified, assume it is 1.0
        if rho0 == []:
            rho0 = Constant(1.0)

        density_type = str(rho0.__class__)
        if not ("dolfin" in density_type):
            print "Converting given density to a DOLFIN Constant"
            rho0 = Constant(rho0)

        # Piola-Kirchhoff stress tensor based on the material model
        P = problem.first_pk_stress(u_mid)

        # Convert time step to a DOLFIN constant
        k = Constant(dt)

        # The variational form corresponding to hyperelasticity
        L = rho0*inner(v - v0, xi)*dx + k*inner(P, grad(xi))*dx \
            - k*inner(B, xi)*dx + inner(u - u0, eta)*dx \
            - k*inner(v_mid, eta)*dx 

        # Add contributions to the form from the Neumann boundary
        # conditions

        # Get Neumann boundary conditions on the stress
        neumann_conditions = problem.neumann_conditions()
        neumann_boundaries = problem.neumann_boundaries()

        boundary = MeshFunction("uint", mesh, mesh.topology().dim() - 1)
        boundary.set_all(len(neumann_boundaries) + 1)

        for (i, neumann_boundary) in enumerate(neumann_boundaries):
            print "Applying Neumann boundary condition at", neumann_boundary
            compiled_boundary = compile_subdomains(neumann_boundary)
            compiled_boundary.mark(boundary, i)
            L = L - dt*inner(neumann_conditions[i], xi)*ds(i)

        a = derivative(L, U, dU)

        # Store variables needed for time-stepping
        self.dt = dt
        self.t_range = t_range
        self.end_time = end_time
        self.a = a
        self.L = L
        self.bcu = bcu
        self.U0 = U0
        self.U = U
        self.B = B
        self.dirichlet_conditions = dirichlet_conditions
        self.neumann_conditions = neumann_conditions
        self.boundary = boundary

        # FIXME: Figure out why I am needed
        self.mesh = mesh
        self.t = 0

    def solve(self):
        """Solve the mechanics problem and return the computed
        displacement field"""

        # Time loop
        for t in self.t_range:
            print "Solving the problem at time t = " + str(self.t)
            self.step(self.dt)
            if self.parameters["save_solution"]:
                self.displacement_velocity_series.store(self.U.vector(), t)
            if self.parameters["save_plot"]:
                u, v = self.U.split(True)
                self.displacement_plot_file << u
                self.velocity_plot_file << v
            self.update()

    def step(self, dt):
        """Setup and solve the problem at the current time step"""

        equation = VariationalProblem(self.a, self.L, self.bcu, exterior_facet_domains = self.boundary, nonlinear = True)
        equation.parameters["newton_solver"]["absolute_tolerance"] = 1e-12
        equation.parameters["newton_solver"]["relative_tolerance"] = 1e-12
        equation.parameters["newton_solver"]["maximum_iterations"] = 100
        equation.solve(self.U)
        return self.U

    def update(self):
        """Update problem at time t"""

        u, v = self.U.split()

        # Propogate the displacements and velocities
        self.U0.assign(self.U)

        # Plot solution
        if self.parameters["plot_solution"]:
            plot(u, title="Displacement", mode="displacement", rescale=True)

        # Move to next time step
        self.t = self.t + self.dt

        # Inform time-dependent functions of new time
        for bc in self.dirichlet_conditions:
            bc.t = self.t
        for bc in self.neumann_conditions:
            bc.t = self.t
        self.B.t = self.t

