##Meshtesting##

from dolfin import *
mesh = Mesh("mesh.xml")
##plot(mesh)
##interactive()

domains = mesh.domains()
markers = domains.markers(1)
cell_domains = domains.cell_domains(mesh)
facet_domains = domains.facet_domains(mesh)
help(markers)
print facet_domains
#print markers
V = FunctionSpace(mesh,"CG",1)
BC = DirichletBC(V,0,cell_domains,0)
print len(BC.get_boundary_values().keys())

##u = TrialFunction(V)
##v = TestFunction(V)
###f = Function(V)
##
##a = 1*v*dx(1)
##A = assemble(a,cell_domains = cell_domains)
##
##a = u*v*dx(0) + u*v*dx(1)
##L = v*dx(0)
##
##u = Function(V)
##solve(a == L, u)
##plot(u, interactive=True)
##
##
