"Simple test program for adaptive Poisson"

__author__ = "Kristoffer Selim and Anders Logg"
__copyright__ = "Copyright (C) 2010 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

from dolfin import *
from numpy import max, sum, argsort
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

    dofs = []
    jumps = []

    while mesh.num_cells() < 100000:

        u = solve(mesh)
        j = evaluate_jumps(u)

        indicators = abs(j.vector().array())

        E = max(indicators)

        jumps.append(E)
        dofs.append(mesh.num_vertices())

        file << j

        marking_fraction = 0.2
        markers = CellFunction("bool", mesh)
        markers.set_all(False)
        indices = list(argsort(indicators))
        indices.reverse()
        for index in indices[:int(marking_fraction*len(indices))]:
            markers[int(index)] = True

        if refinement_type == "uniform":
            mesh = refine(mesh)
        elif refinement_type == "bisection":
            parameters["refinement_algorithm"] = "recursive_bisection"
            mesh = refine(mesh, markers)
        else:
            parameters["refinement_algorithm"] = "regular_cut"
            mesh = refine(mesh, markers)

    return dofs, jumps, mesh

n1, j1, m1 = run_experiment("uniform")
n2, j2, m2 = run_experiment("bisection")
n3, j3, m3 = run_experiment("regular")

import pylab
pylab.loglog(n1, j1, '-o')
pylab.loglog(n2, j2, 'r-o')
pylab.loglog(n3, j3, 'g-o')
pylab.legend(["uniform", "bisection", "regular"])
pylab.grid(True)
pylab.xlabel("#dofs")
pylab.ylabel("Error estimate")
pylab.title("Simple Poisson model problem")
pylab.savefig("jumps.png")

plot(m1, title="Uniform")
plot(m2, title="Bisection")
plot(m3, title="Regular cut")

pylab.show()
interactive()
