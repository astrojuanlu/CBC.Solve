"Simple test program for adaptive Poisson"

__author__ = "Kristoffer Selim and Anders Logg"
__copyright__ = "Copyright (C) 2010 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

from dolfin import *
from numpy import max
from pylab import *

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

    return j

def refine_mesh(mesh, j):
    "Refine mesh based on jumps"

    #return refine(mesh)

    fraction = 0.5
    indicators = j.vector().array()
    indices = list(argsort(indicators))
    indices.reverse()
    markers = CellFunction("bool", mesh)
    markers.set_all(False)
    for index in indices[:int(round(fraction*len(indices)))]:
        markers[int(index)] = True
    markers.set_all(True)
    return refine(mesh, markers)

mesh = UnitSquare(2, 2)
num_refinements = 7
file = File("jumps.pvd")
jumps = []

for level in range(num_refinements + 1):

    print
    print "Level", level
    print

    u = solve(mesh)
    j = evaluate_jumps(u)

    #plot(u, title="Solution")
    #plot(j, title="Jumps")

    jumps.append(max(j.vector().array()))
    file << j

    mesh = refine_mesh(mesh, j)

print
print "Maximum jumps:", str(jumps)

semilogy(range(num_refinements + 1), jumps, '-o')
grid(True)
xlabel("Refinement level")
ylabel("Maximum squared jump indicator")
title("Simple Poisson model problem")
show()
