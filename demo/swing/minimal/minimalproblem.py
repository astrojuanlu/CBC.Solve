
"""
A very small FSI problem that is used to test the FSI Newton Solver during development
A thin structure wall has a flowing fluid above it
"""

__author__ = "Gabriel Balaban"
__copyright__ = "Copyright (C) 2010 Simula Research Laboratory and %s"% __author__
__license__  = "GNU GPL Version 3 or any later version"

from dolfin import *
from cbc.swing.fsiproblem import FSI

#Mesh Parameters
meshlength = 1.0 
meshheight = 1.2 
strucheight = 0.2
fluidheight = meshheight - strucheight
nx = 5
ny = 6

#####################################
#Define Boundaries 
#####################################

up = "near(x[1],%g)"%(meshheight)
down = "near(x[1],0.0)"

#Important! Exclude the FSI Nodes
yinfluid = "x[1] < %g - DOLFIN_EPS"%(fluidheight)

#Important! Include the FSI Nodes
yinstruc = "x[1] > %g - DOLFIN_EPS"%(fluidheight)

xright = "near(x[0],%g)"%(meshlength)
xleft = "near(x[0],0.0)"

fluidleft = xleft + "&&" + yinfluid
fluidright = xright + "&&" + yinfluid
strucleft = xleft + "&&" + yinstruc
strucright = xright + "&&" + yinstruc

strucbottom = "near(x[1],%g - %g)"%(meshheight,strucheight)

class FluidDN(SubDomain):
    def inside(self,x,on_boundary):
        return x[1] < fluidheight - DOLFIN_EPS and \
               near(x[0],meshlength) or \
               near(x[0],0.0) 
    
#Define Structure Subdomain
class Structure(SubDomain):
    def inside(self, x, on_boundary):
        return x[1] > fluidheight - DOLFIN_EPS

class FSIMini(FSI):
    """FSI Miniproblem"""
    def __init__(self):
        mesh = Rectangle(0.0,0.0,meshlength,meshheight,nx,ny)
        FSI.__init__(self,mesh)
                                     
    def end_time(self):
        return 0.2

    def initial_step(self):
        return 0.05

    def __str__(self):
        return "FSI Miniproblem"

    #--- Fluid problem ---
    def fluid_density(self):
        return 1.0

    def fluid_viscosity(self):
        return 1.0/8.0

    def fluid_velocity_dirichlet_boundaries(self):
        return [down]
    
    def fluid_pressure_dirichlet_values(self):
        return (1.0, 0.0)

    def fluid_pressure_dirichlet_boundaries(self):
        return [fluidleft, fluidright]
    
    def fluid_donothing_boundaries(self):
        return [FluidDN()]

    #Needs to match pressure BC for the Newton solver
    def fluid_pressure_initial_condition(self):
        return "1.0 - (x[0]/%g)*(x[0]/%g)" % (meshlength,meshlength)
    
    #--- Structure problem ---
    def structure(self):
        return Structure()

    def structure_density(self):
        return 10.0

    def structure_mu(self):
        return 3.8461

    def structure_lmbda(self):
        return 5.76
    
    def structure_dirichlet_boundaries(self):
        return [strucleft,strucright]
    #--- Mesh Problem ---

    def mesh_mu(self):
        return 3.8461
    
    def mesh_lmbda(self):
        return 5.76

    def mesh_dirichlet_boundaries(self):
        return [fluidleft,fluidright,down]

if __name__ == "__main__":
    import fsinewton.solver.solver_fsinewton as sfsi
    import fsinewton.utils.misc_func as mf
    
    problem = FSIMini()
    solver = sfsi.AutoDerivativeFSINewtonSolver(problem)
    solver.solve(plot = True)
    interactive()
