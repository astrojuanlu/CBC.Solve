"Simple test program for adaptive Poisson"

__author__ = "Kristoffer Selim and Anders Logg"
__copyright__ = "Copyright (C) 2010 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

from dolfin import *
from numpy import max
import pylab

f = Expression("1")

def solve(mesh):
    "Compute solution on given mesh"

    V = FunctionSpace(mesh, "CG", 1)
    u = TrialFunction(V)
    v = TestFunction(V)
    a = inner(grad(u), grad(v))*dx
    L = f*v*dx
    bc = DirichletBC(V, 0.0, DomainBoundary())

    problem = VariationalProblem(a, L, bc)
    u = problem.solve()

    return u

def evaluate_jumps(u):
    "Evaluate jumps"

    mesh = u.function_space().mesh()
    DG = FunctionSpace(mesh, "DG", 0)
    w = TestFunction(DG)
    j = Function(DG)
    n = FacetNormal(mesh)
    eta = avg(w)*inner(jump(dot(grad(u), n)), jump(dot(grad(u), n)))*dS
    assemble(eta, tensor=j.vector())

    print "Maximum jump:", max(j.vector().array())
    print "Maximum cell size:", mesh.hmax()

    return j

def run_experiment(refinement_type):
    "Run experiment for given refinement type ('uniform' or 'adaptive')"

    mesh = UnitSquare(2, 2)
    file = File("jumps_%s.pvd" % refinement_type)

    hmaxs = []
    jumps = []

    while mesh.num_cells() < 100000:

        u = solve(mesh)
        j = evaluate_jumps(u)

        jumps.append(max(abs(j.vector().array())))
        hmaxs.append(mesh.hmax())
        file << j

        if refinement_type == "uniform":
            markers = CellFunction("bool", mesh)
            markers.set_all(True)
            mesh = refine(mesh, markers)
        else:
            mesh = refine(mesh)

    return hmaxs, jumps

h1, j1 = run_experiment("uniform")
h2, j2 = run_experiment("adaptive")

import pylab
pylab.loglog(h1, j1, '-o')
pylab.loglog(h2, j2, 'r-o')
pylab.legend(["uniform", "adaptive"])
pylab.grid(True)
pylab.xlabel("h max")
pylab.ylabel("Maximum squared jump indicator")
pylab.title("Simple Poisson model problem")
pylab.savefig("jumps.png")
pylab.show()
