from dolfin import *
from numpy import array, loadtxt

parameters["form_compiler"]["cpp_optimize"] = True
# parameters["form_compiler"]["optimize"] = True

length = 4
height = 1
nlength = 80
nheight = 20
mesh = Rectangle(0.0, 0.0, length, height, nlength, nheight)
D = mesh.topology().dim()
vector = VectorFunctionSpace(mesh, "CG", 1)

olength = 0.4
oheight = 0.6

class Obstruction(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] >= (length - olength)/2.0 \
            and x[0] <= (length + olength)/2.0 \
            and x[1] <= oheight

sub_domains = MeshFunction("uint", mesh, D)
sub_domains.set_all(2)

obstruction = Obstruction()
obstruction.mark(sub_domains, 1)

# Mark exterior boundaries
boundary_1 = compile_subdomains("x[0] == 0.0")
boundary_2 = compile_subdomains("x[1] == 1.0")
boundary_3 = compile_subdomains("x[0] == 4.0")
boundary_4 = compile_subdomains("x[0] >= (4.0 + 0.4)/2.0")
boundary_6 = compile_subdomains("x[0] <= (4.0 - 0.4)/2.0")
boundary_7 = compile_subdomains("x[0] > (4.0 - 0.4)/2.0 && x[0] < (4.0 + 0.4)/2.0")

boundary = MeshFunction("uint", mesh, D - 1)
boundary.set_all(0)
boundary_1.mark(boundary, 1)
boundary_2.mark(boundary, 2)
boundary_3.mark(boundary, 3)
boundary_4.mark(boundary, 4)
boundary_6.mark(boundary, 6)
boundary_7.mark(boundary, 7)

interior_boundary = MeshFunction("uint", mesh, D - 1)
interior_boundary.set_all(0)

facet_orientation = mesh.data().create_mesh_function("Facet orientation", D - 1)
facet_orientation.set_all(0)

# Mark interior "boundaries"
for facet in facets(mesh):

    if facet.num_entities(D) == 1:
        continue

    # Get the two cell indices
    c0, c1 = facet.entities(D)

    # Create the two cells
    cell0 = Cell(mesh, c0)
    cell1 = Cell(mesh, c1)

    # Get the two midpoints
    p0 = cell0.midpoint()
    p1 = cell1.midpoint()

    # Check if the points are inside
    p0_inside = obstruction.inside(p0, False)
    p1_inside = obstruction.inside(p1, False)

    # Look for points where exactly one is inside the obstruction
    if p0_inside and not p1_inside:
        interior_boundary[facet.index()] = 5
        facet_orientation[facet.index()] = c1
    elif p1_inside and not p0_inside:
        interior_boundary[facet.index()] = 5
        facet_orientation[facet.index()] = c0

B  = Expression(("0.0", "0.0"))
T  = Expression(("-1.0", "0.0"))

mixed_element = MixedFunctionSpace([vector, vector])
V = TestFunction(mixed_element)
dU = TrialFunction(mixed_element)
U = Function(mixed_element)
U0 = Function(mixed_element)

xi, eta = split(V)
u, v = split(U)
u0, v0 = split(U0)

clamp = Constant((0.0, 0.0))
left = compile_subdomains("x[0] == 0.0")
bcl = DirichletBC(mixed_element.sub(0), clamp, left)

pull = Expression(("e1*t", "0.0"))
pull.e1 = 0.01
right = compile_subdomains("x[0] == right")
right.right = length
bcr = DirichletBC(mixed_element.sub(0), pull, right)

u_mid = 0.5*(u0 + u)
v_mid = 0.5*(v0 + v)

I = Identity(v.cell().d)
F = I + grad(u_mid)
C = F.T*F
E = (C - I)/2
E = variable(E)

mu    = Constant(3.85)
lmbda = Constant(5.77)

psi = lmbda/2*(tr(E)**2) + mu*tr(E*E)
S = diff(psi, E)
P = F*S

obstruction_mesh = SubMesh(mesh, sub_domains, 1)
N = FacetNormal(obstruction_mesh)

rho0 = Constant(1.0)
dt = Constant(0.1)
L = rho0*inner(v - v0, xi)*dx(2) + dt*inner(P, grad(xi))*dx(2) \
    - dt*inner(B, xi)*dx(2) - dt*inner(T, xi)*dS(5) \
    + inner(u - u0, eta)*dx(2) - dt*inner(v_mid, eta)*dx(2) \
    + rho0*inner(v - v0, xi)*dx(1) + dt*inner(P, grad(xi))*dx(1) \
    - dt*inner(B, xi)*dx(1) \
    + inner(u - u0, eta)*dx(1) - dt*inner(v_mid, eta)*dx(1)
a = derivative(L, U, dU)



t = 0.0
T = 1.0

problem = VariationalProblem(a, L, [bcl, bcr], 
                             cell_domains = sub_domains,
                             interior_facet_domains = interior_boundary,
                             exterior_facet_domains = boundary,
                             nonlinear = True)

file = File("displacement.pvd")

while t < T:

    t = t + float(dt)
    pull.t = t

    problem.solve(U)
    u, v = U.split()
    file << u

    plot(u)

    U0.assign(U)
