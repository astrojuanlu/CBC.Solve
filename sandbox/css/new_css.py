__author__ = "Mikael Mortensen <Mikael.Mortensen@ffi.no>"
__date__ = "2010-05-18"
__copyright__ = "Copyright (C) 2010 " + __author__
__license__  = "GNU GPL version 3 or any later version"

from solverbase import *

class Solver(SolverBase):
    "New consistent splitting scheme."

    def __init__(self, options):
        SolverBase.__init__(self, options)

    def solve(self, problem):

        # Get problem parameters
        mesh = problem.mesh
        dt, t, t_range = problem.timestep(problem)

        # Define function spaces
        V = VectorFunctionSpace(mesh, "CG", 2)
        Q = FunctionSpace(mesh, "CG", 1)
        DG = FunctionSpace(mesh, "DG", 0)

        # Get initial and boundary conditions
        u0, p0 = problem.initial_conditions(V, Q)
        bcu, bcp = problem.boundary_conditions(V, Q, t)

        # Remove boundary stress term is problem is periodic
        if is_periodic(bcp):
            beta = Constant(0)
        else:
            beta = Constant(1)

        # Test and trial functions
        v = TestFunction(V)
        q = TestFunction(Q)
        u = TrialFunction(V)
        p = TrialFunction(Q)

        # Functions
        u0 = interpolate(u0, V)
        u1 = interpolate(u0, V)
        u2 = interpolate(u0, V)
        p0 = interpolate(p0, Q)
        p1 = interpolate(p0, Q)
        p2 = interpolate(p0, Q)
        nu = Constant(problem.nu)
        k  = Constant(dt)
        f  = problem.f
        n  = FacetNormal(mesh)

        ps = 1.5*p1-0.5*p0 # Since p is computed at integer timesteps but is required on t+1/2 in F1
        #ps = p1

        # Tentative velocity step
        U = 0.5*(u1 + u)
        F1 = (1/k)*inner(v, u - u1)*dx + inner(v, grad(u1)*u1)*dx \
            + nu*inner(grad(v), grad(U))*dx + inner(v, grad(ps))*dx \
            - inner(v, f)*dx
        a1 = lhs(F1)
        L1 = rhs(F1)

        # Velocity correction
        a2 = inner(grad(v), grad(u))*dx
        L2 = inner(grad(v), grad(u2))*dx - inner(div(v),div(u2))*dx

        # Pressure term
        a3 = inner(grad(q), grad(p))*dx
        L3 = -inner(grad(q), grad(u2)*u2)*dx

        # Assemble matrices
        A1 = assemble(a1)
        A2 = assemble(a2)
        A3 = assemble(a3)

        [bc.apply(A1) for bc in bcu]
        [bc.apply(A2) for bc in bcu]
        [bc.apply(A3) for bc in bcp]

        # Compute initial pressure from u0
        b = assemble(L3)
        if len(bcp) == 0 or is_periodic(bcp): normalize(b)
        [bc.apply(b) for bc in bcp]
        if is_periodic(bcp):
            solve(A3, p0.vector(), b)
        else:
            solve(A3, p0.vector(), b, "bicgstab", 'ilu')
        if len(bcp) == 0 or is_periodic(bcp): normalize(p1.vector())
        p1.assign(p0)

        # Time loop
        self.start_timing()
        for t in t_range:

            # Compute tentative velocity step
            b = assemble(L1)
            #A1 = assemble(a1)
            [bc.apply(b) for bc in bcu]
            solve(A1, u2.vector(), b, "bicgstab", "ilu")

            # Velocity correction
            b = assemble(L2)
            [bc.apply(b) for bc in bcu]
            solve(A2, u2.vector(), b, 'bicgstab', 'ilu')

            # Pressure
            b = assemble(L3)
            if len(bcp) == 0 or is_periodic(bcp): normalize(b)
            [bc.apply(b) for bc in bcp]
            if is_periodic(bcp):
                solve(A3, p2.vector(), b)
            else:
                solve(A3, p2.vector(), b, "bicgstab", 'ilu')
            if len(bcp) == 0 or is_periodic(bcp): normalize(p1.vector())

            # Update
            self.update(problem, t, u2, p2)
            u0.assign(u1)
            u1.assign(u2)
            p0.assign(p1)
            p1.assign(p2)

        return u1, p1

    def __str__(self):
        return "New CSS"
