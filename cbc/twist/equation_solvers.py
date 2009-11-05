__author__ = "Harish Narayanan"
__copyright__ = "Copyright (C) 2009"
__license__  = "GNU GPL Version 3 or any later version"

from dolfin import *
from cbc.common import CBCSolver
from cbc.twist.kinematics import Grad, DeformationGradient

class StaticMomentumBalanceSolver(CBCSolver):
    "Solves the static balance of linear momentum"

    def solve(self, problem):
        """Solve the mechanics problem and return the computed
        displacement field"""

        # Get problem parameters
        mesh = problem.mesh()

        # Define function spaces
        vector = VectorFunctionSpace(mesh, "CG", 1)

        # Get Dirichlet boundary conditions on the displacement field
        bcu = problem.boundary_conditions(vector)

        # Define fields
        # Test and trial functions
        v = TestFunction(vector)
        u = Function(vector)
        du = TrialFunction(vector)
        
        # Driving forces
        B = problem.body_force(vector)
        T = problem.surface_force(vector)

        # First Piola-Kirchhoff stress tensor based on the material
        # model
        P  = problem.first_pk_stress(u)

        # The variational form corresponding to hyperelasticity
        L = inner(P, Grad(v))*dx - inner(B, v)*dx - inner(T, v)*ds
        a = derivative(L, u, du)

        # Setup and solve the problem
        equation = VariationalProblem(a, L, bcu, nonlinear = True)
        equation.solve(u)

        plot(u, interactive = True)

        return u

class MomentumBalanceSolver(CBCSolver):
    "Solves the quasistatic/dynamic balance of linear momentum"

    def solve(self, problem):
        """Solve the mechanics problem and return the computed
        displacement field"""

        # Get problem parameters
        mesh      = problem.mesh()
        dt        = problem.time_step()
        end_time  = problem.end_time()

        # Set initial time
        t = 0

        # Define function spaces
        scalar = FunctionSpace(mesh, "CG", 1)
        vector = VectorFunctionSpace(mesh, "CG", 1)

        # Get initial conditions
        u0, v0 = problem.initial_conditions(vector)

        # Get reference density
        rho0 = problem.reference_density(scalar)
        density_type = str(rho0.__class__)
        if not ("dolfin" in density_type):
            print "Converting given density to a DOLFIN Constant"
            rho0 = Constant(mesh, rho0)

        # Get time-dependent boundary conditions and driving
        # forces
        bcu = problem.boundary_conditions(t, vector)
        B  = problem.body_force(t, vector)
        T  = problem.surface_force(t, vector)

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

        # Determine initial acceleration
        a0 = TrialFunction(vector)
        P = problem.first_pk_stress(u0)
        a_accn = inner(a0, v)*dx
        L_accn = - inner(P, Grad(v))*dx + inner(B, v)*dx \
                 + inner(T, v)*ds
        problem_accn = VariationalProblem(a_accn, L_accn)
        a0 = problem_accn.solve()

        # Time step size
        k  = Constant(mesh, dt)
        
        # Piola-Kirchhoff stress tensor based on the material model
        P = problem.first_pk_stress(u1)

        # FIXME: A general version of the trick below is what should
        # be used instead. The commentend-out lines only work well for
        # quadratically nonlinear models, e.g. St. Venant Kirchhoff.
        
        # S0 = problem.second_pk_stress(u0)
        # S1 = problem.second_pk_stress(u1)
        # Sm = 0.5*(S0 + S1)
        # Fm = DeformationGradient(0.5*(u0 + u1))
        # P  = Fm*Sm

        # Parameters pertinent to (HHT) time integration
        # alpha = 1.0
        beta = 0.25
        gamma = 0.5

        a1 = a0*(1.0 - 1.0/(2*beta)) - (u0 - u1 + k*v0)/(beta*k**2)

        # The variational form corresponding to hyperelasticity
        L = int(problem.is_dynamic())*rho0*inner(a1, v)*dx \
        + inner(P, Grad(v))*dx - inner(T, v)*ds \
        - inner(B, v)*dx 
        a = derivative(L, u1, du)

        # Set initial solve time
        t = dt

        # Time loop
        while t <= end_time:

            print "Solving the problem at time t = " + str(t)

            # Get time-dependent boundary conditions and driving
            # forces
            bcu = problem.boundary_conditions(t, vector)
            B  = problem.body_force(t, vector)
            T  = problem.surface_force(t, vector)

            # Setup and solve the problem at the current time step
            equation = VariationalProblem(a, L, bcu, nonlinear = True)
            equation.solve(u1)

            # Update
            self.update(problem, t, u1)
            a1 = a0*(1.0 - 1.0/(2*beta)) - (u0 - u1 + k*v0)/(beta*k**2)
            a1 = project(a1, vector)
            v1 = v0 + k*((1 - gamma)*a1 + gamma*a0)
            v1 = project(v1, vector)

            u0.assign(u1)
            v0.assign(v1)
            a0.assign(a1)

            t = t + dt

    def update(self, problem, t, u):
        """Update problem at time t"""

        plot(u, title="Displacement", rescale=True)
