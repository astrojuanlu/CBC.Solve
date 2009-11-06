__author__ = "Kristian Valen-Sendstad and Anders Logg"
__copyright__ = "Copyright (C) 2009 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

# Last changed: 2009-11-06

__all__ = ["StaticNavierStokesSolver", "NavierStokesSolver"]

from dolfin import *
from numpy import linspace
from cbc.common.utils import *
from cbc.common import CBCSolver

class StaticNavierStokesSolver(CBCSolver):
    "Navier-Stokes solver (static)"

    def solve(self):
        "Solve problem and return computed solution (u, p)"
        error("Static Navier-Stokes solver not implemented.")

class NavierStokesSolver(CBCSolver):
    "Navier-Stokes solver (dynamic)"

    def solve(self, problem):
        "Solve problem and return computed solution (u, p)"

        # Get mesh and time step range
        mesh = problem.mesh()
        dt, t_range = timestep_range(problem, mesh)

        # Function spaces
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
        u1 = Function(V)
        p0 = interpolate(p0, Q)
        p1 = Function(Q)

        # Coefficients
        nu = Constant(mesh, problem.viscosity())
        k = Constant(mesh, dt)
        f = problem.body_force(V)

        # Tentative velocity step
        U = 0.5*(u0 + u)
        F1 = (1/k)*inner(v, u - u0)*dx + inner(v, grad(u0)*u0)*dx \
            + nu*inner(grad(v), grad(U))*dx + inner(v, grad(p0))*dx \
            - inner(v, f)*dx
        a1 = lhs(F1)
        L1 = rhs(F1)

        # Pressure correction
        a2 = inner(grad(q), k*grad(p))*dx
        L2 = inner(grad(q), k*grad(p0))*dx - q*div(u1)*dx

        # Velocity correction
        a3 = inner(v, u)*dx
        L3 = inner(v, u1)*dx + inner(v, k*grad(p0 - p1))*dx

        # Assemble matrices
        A1 = assemble(a1)
        A2 = assemble(a2)
        A3 = assemble(a3)

        # Time loop
        for t in t_range:

            # Compute tentative velocity step
            b = assemble(L1)
            [bc.apply(A1, b) for bc in bcu]
            solve(A1, u1.vector(), b, "gmres", "ilu")

            # Pressure correction
            b = assemble(L2)
            if len(bcp) == 0 or is_periodic(bcp): normalize(b)
            [bc.apply(A2, b) for bc in bcp]
            if is_periodic(bcp):
                solve(A2, p1.vector(), b)
            else:
                solve(A2, p1.vector(), b, 'gmres', 'amg_hypre')
            if len(bcp) == 0 or is_periodic(bcp): normalize(p1.vector())

            # Velocity correction
            b = assemble(L3)
            [bc.apply(A3, b) for bc in bcu]
            solve(A3, u1.vector(), b, "gmres", "ilu")

            # Update
            self.update(problem, t, u1, p1)
            u0.assign(u1)
            p0.assign(p1)

        return u1, p1

    def update(self, problem, t, u, p):
        "Update problem at time t"

        # Plot solution
        plot(u, title="Velocity", rescale=True)
        plot(p, title="Pressure", rescale=True)

def timestep_range(problem, mesh):
    "Return time step and time step range for given problem"

    # Get problem parameters
    T = problem.end_time()
    dt = problem.time_step()
    nu = problem.viscosity()
    U = problem.max_velocity()

    # Use time step specified in problem if available
    if not dt is None:
        n = int(T / dt + 1.0)
    # Otherwise, base time step on mesh size
    else:
        h = mesh.hmin()
        dt = 0.25*h**2 / (U*(nu + h*U))
        n = int(T / dt + 1.0)

    # Compute range
    t_range = linspace(0, T, n + 1)[1:]
    dt = t_range[0]

    return dt, t_range
