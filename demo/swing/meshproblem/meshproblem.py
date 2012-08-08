"""A Mesh problem with a given initial structure displacement"""
__author__ = "Gabriel Balaban"
__copyright__ = "Copyright (C) 2012 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

from dolfin import *
from cbc.swing.fsiproblem import FSI

class Structure(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] > 1.0 - DOLFIN_EPS

#Mesh Parameters
meshlength = 1.25 
meshheight = 1.0 
struclength = 0.25 
fluidlength = meshlength - struclength
nx = 5
ny = 4

#####################################
#Define Boundaries 
#####################################
h = 0.1 #Parabola height
xright = "near(x[0],%g)"%(meshlength)
xleft = "near(x[0],0.0)"
yup = "near(x[1],%g)"%(meshheight)
ydown = "near(x[1],0.0)"


#Here the FSI Nodes are excluded
xinfluid = "x[0] < %g - DOLFIN_EPS"%(fluidlength)

#Here the FSI Nodes are included
xinstruc = "x[0] > %g - DOLFIN_EPS"%(fluidlength)

#Fluid BC and Mesh
fluidleft = xleft
fluidup = yup + "&&" + xinfluid
fluiddown = ydown + "&&" + xinfluid

#Structure BC
strucright = xright
strucup =  yup + "&&" + xinstruc
strucdown =  ydown + "&&" + xinstruc

class MeshProblem(FSI):
    """A initial structure displacement is given which should
        result in mesh movement"""
    
    def __init__(self):
        mesh = Rectangle(0.0,0.0,meshlength,meshheight,nx,ny)
        FSI.__init__(self,mesh)

    def end_time(self):
        return 1.0

    def initial_step(self):
        return 1.e-1
    
    def __str__(self):
        return "Mesh only problem"
    
    def mesh_mu(self):
        return 8.8461
    
    def mesh_lmbda(self):
        return 1.76

    def mesh_dirichlet_boundaries(self):
        return [fluidleft,fluidup,fluiddown]

    def struc_displacement_initial_condition(self):
        return ("4.0*%g*(x[1] - 0.5)*(x[1] - 0.5) - %g"%(h,h),"0.0")
    def structure(self):
        return Structure()

class FSIMeshProblem(MeshProblem):
    """The MeshProblem is extended to a full FSI problem"""
    def __str__(self):
        return "Mesh FSI problem"
    
    def structure_dirichlet_boundaries(self):
        return [strucup,strucdown]

    def fluid_velocity_dirichlet_boundaries(self):
        return [fluidleft]

if __name__ == "__main__":
    import fsinewton.solver.solver_fsinewton as sfsi
    import fsinewton.utils.misc_func as mf
    problem = FSIMeshProblem()
    solver = sfsi.AutoDerivativeFSINewtonSolver(problem,plot = "mesh",\
                                                plotfinal = True)
    solver.solve(single_step = True)
    interactive()
