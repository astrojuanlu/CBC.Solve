from dolfin import *

# Test mesh
n = 99
mesh = UnitSquare(n, n)

# Boundaries
left   = "x[0] <= 0.0 + DOLFIN_EPS && on_boundary"
right  = "x[0] >= 1.0 - DOLFIN_EPS && on_boundary"
bottom = "x[1] <= 0.0 + DOLFIN_EPS && on_boundary"
top    = "x[1] >= 1.0 - DOLFIN_EPS && on_boundary"

# Specifying Neumann boundary conditions in two parts
# The condition itself
def neumann_conditions(vector):
    pull_left = Expression(("force", "0.0"), V = vector)
    pull_right  = Expression(("force", "0.0"), V = vector)
    pull_left.force  = -1.0
    pull_right.force =  1.0
    return [pull_left, pull_right]

# The boundary where it acts
def neumann_boundaries():
    return [left, right, top, bottom]

# Function spaces
scalarDG = FunctionSpace(mesh, "DG", 0)
vectorDG = VectorFunctionSpace(mesh, "DG", 0)

#neumann_conditions = neumann_conditions(vector)
neumann_boundaries = neumann_boundaries()

u = TrialFunction(scalarDG)
v = TestFunction(scalarDG)
one = Constant(mesh, 1.0)

boundaries = MeshFunction("uint", mesh, mesh.topology().dim() - 1)

boundaries.set_all(len(neumann_boundaries) + 1)

for (i, neumann_boundary) in enumerate(neumann_boundaries):
    compiled_boundary = compile_subdomains(neumann_boundary)
    compiled_boundary.mark(boundaries, i)

    a = v*u*dx
    L = v*ds(i)

    problem = VariationalProblem(a, L, exterior_facet_domains = boundaries)
    u1 = problem.solve()

    plot(u1, interactive = True)
