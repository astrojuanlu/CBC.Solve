from dolfin import *

n = 7
mesh = UnitSquare(n, n)

left    = "x[0] == 0.0"
right   = "x[0] == 1.0"
top     = "x[1] == 0.0"
bottom  = "x[1] == 1.0"
left_half = "x[0] <= 0.5"
right_half = "x[0] >= 0.5"

class Left_Half(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] < 0.5 + DOLFIN_EPS

def neumann_conditions(vector):
    pull_left = Expression(("force", "0.0"), V = vector)
    pull_right  = Expression(("force", "0.0"), V = vector)
    pull_left.force  = -1.0
    pull_right.force =  1.0
    return [pull_left, pull_right]

def neumann_boundaries():
    return [left, right]

# def body_force_domains():
#     return [left_half, right_half]

scalar   = FunctionSpace(mesh, "CG", 2)
vector   = VectorFunctionSpace(mesh, "CG", 2)
scalarDG = FunctionSpace(mesh, "DG", 2)
vectorDG = VectorFunctionSpace(mesh, "DG", 2)

neumann_conditions = neumann_conditions(vector)
neumann_boundaries = neumann_boundaries()
body_force_domains = [left_half, right_half]

u = TrialFunction(vector)
v = TestFunction(vector)
one = Constant(mesh, 1.0)
vector_one = Constant(mesh, (1.0, 1.0))

domain = MeshFunction("uint", mesh, mesh.topology().dim())
boundaries = MeshFunction("uint", mesh, mesh.topology().dim() - 1)

domain.set_all(0)

#compiled_domain = compile_subdomains(left_half)
#compiled_domain.mark(domain, 1)

#left_half = Left_Half()
left_half = compile_subdomains(left_half)
left_half.mark(domain, 1)

a = dot(v, u)*dx
L = dot(v, vector_one)*dx(1)

problem = VariationalProblem(a, L)
u1 = problem.solve()

plot(u1, interactive = True)

# for (i, body_force_domain) in enumerate(body_force_domains):
#     compiled_domain = compile_subdomains(body_force_domain)
#     compiled_domain.mark(domain, i)

#     a = v*u*dx
#     L = v*one*dx(i)

#     problem = VariationalProblem(a, L)
#     u1 = problem.solve()

#     plot(u1, interactive = True)

# boundaries = MeshFunction("uint", mesh, mesh.topology().dim() - 1)
# boundaries.set_all(2)

# for (i, neumann_boundary) in enumerate(neumann_boundaries):
#     compiled_boundary = compile_subdomains(neumann_boundary)
#     compiled_boundary.mark(boundaries, i)

#     a = v*u*dx
#     L = v*one*ds(i)

#     problem = VariationalProblem(a, L)
#     u1 = problem.solve()

#     plot(u1, interactive = True)
