from dolfin import *

subdomain = "x[0] < DOFLIN_EPS"
compile_subdomains(subdomain)

#f = Expression("sin(x[0] + DOLFIN_EPS)")
