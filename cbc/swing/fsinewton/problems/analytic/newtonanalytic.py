__author__ = "Gabriel Balaban"
__copyright__ = "Copyright (C) 2012 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

# First added:  2012-03-04
# Last changed: 2012-05-03
import io
from math import sin
from cbc.swing import *
import right_hand_sides as rhs
import demo.swing.analytic.analytic as ana
import fsinewton.problems.base as base
import fsinewton.solver.solver_fsinewton as sfsi
import fsinewton.utils.misc_func as mf

#Exclude FSI Nodes
influid = "x[0] < 1.0 - DOLFIN_EPS"

#include FSI  Nodes
influidfsi = "x[0] < 1.0 + DOLFIN_EPS"

#Include FSI Nodes
instruc = "x[0] >= 1.0 - DOLFIN_EPS"

meshbc = "on_boundary &&" + influid
strucnoslip = instruc + "&&" + ana.noslip
strucright = instruc  + "&&" + ana.right
fluidnoslip = influid + "&&" + ana.noslip


from cbc.twist import PiolaTransform
from cbc.swing.operators import Sigma_F as _Sigma_F

# Define Fluid Neumann BC subdomain
class FluidNeumann(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0],0.0) 

class NewtonAnalytic(ana.Analytic,base.NewtonFSI):
    """
    FSI Analytic problem wrapped for Newtons method
        parameters
            bctype - set of boundary conditions to use on structure fsi interface
                normal, dirichlet, neumann
    
    """

    def __init__(self, num_refine = 0, bctype = "normal", endtime = 0.1):
        
        #Initialize the analytic problem first
        ana.Analytic.__init__(self)
        self.singlemesh = self._original_mesh
        self.bctype = bctype

        #Initialize the reference fluid forces
        self.F_F = Expression(rhs.cpp_F_F, degree=2)
        self.G_F = Expression(rhs.cpp_G_F, degree=1)

        #Exact reference domain fluid stress.
        self.G_F_FSI = Expression(rhs.cpp_G_F_FSI, degree = 5)

        # Exact reference domain fluid solutions
        self.U_F = Expression(rhs.cpp_U_F, degree=2)
        self.P_F = Expression(rhs.cpp_P_F, degree=3)
        self.inistep = ana.application_parameters["initial_timestep"]

        self.endtime = endtime
        
        # Initialize expressions
        forces = [self.F_F, self.G_F, self.G_F_FSI]
        solutions = [self.U_F, self.P_F]
        for f in forces + solutions:
            f.C = ana.C
            f.t = 0.0

        for i in range(num_refine):
            self.refine()
            
        #Then give the mesh to the FSINewtonTest class
        base.NewtonFSI.__init__(self,self.singlemesh)

    def refine(self):
        """Refine mesh and time step"""
        info_blue("Refining mesh and time step")
        #Space refinement
        self.singlemesh = refine(self.singlemesh)
        #Time refinement
        self.inistep *= 0.5

    def update(self, t0, t1, dt):
        ana.Analytic.update(self,t0,t1,dt)
        t = 0.5*(t0 + t1)
        
        self.F_F.t = t
        self.G_F.t = t

        self.U_F.t = t1
        self.P_F.t = t1
        self.U_S.t = t1
        self.G_F_FSI.t = t1
        
    def initial_step(self):
        return self.inistep

    def end_time(self):
        return self.endtime

    def struc_displacement_initial_condition(self):
        return self.U_S
    
    def struc_velocity_initial_condition(self):
        return self.P_S

    #get rid of V argument
    def fluid_velocity_initial_condition(self):
        return self.U_F
    
    def fluid_pressure_initial_condition(self):
        return self.P_F

    def fluid_velocity_neumann_boundaries(self):
        return FluidNeumann()
    
    def fluid_velocity_neumann_values(self):
        return self.G_F
    
    def fluid_body_force(self):
        return self.F_F

    ####################################################
    #Boundary Conditions
    ####################################################

    def fluid_velocity_dirichlet_values(self):
        return [self.U_F]
    
    def fluid_velocity_dirichlet_boundaries(self):
            return [ana.noslip]

    def structure_dirichlet_boundaries(self):
        if self.bctype == "dirichlet":
            info_blue("prescribing exact structure values on the fsi interface")
            return [strucnoslip,strucright,ana.interface]
        else:
            return [strucnoslip,strucright]

    def structure_dirichlet_values(self):
        if self.bctype == "dirichlet":
            return [self.U_S,self.U_S,self.U_S]
        else:
            return [self.U_S,self.U_S]

    def mesh_dirichlet_boundaries(self):
        return [meshbc]
    
    def structure_velocity_dirichlet_boundaries(self):
        return [ana.noslip]

    #Exact fluid stress
    def fluid_fsi_stress(self):
        if self.bctype == "neumann":
            return self.G_F_FSI
                                 
class SetStruc(object):    
    def structure_dirichlet_values(self):
        return [self.U_S,self.U_S]
    
    def structure_dirichlet_boundaries(self):
        return [ana.Structure(),ana.interface]

    def structure_velocity_dirichlet_boundaries(self):
        return [ana.Structure(),ana.interface]
    
    def structure_velocity_dirichlet_values(self):
        return [self.P_S,self.P_S]

class SetMesh(object):
    pass
    def mesh_lm_dirichlet_boundaries(self):
        return [ana.interface]

    def mesh_dirichlet_values(self):
        return [self.U_M]

    def mesh_dirichlet_boundaries(self):
        return [influid]

class SetFluid(object): 
    def fluid_lm_dirichlet_boundaries(self):
        return [ana.interface]

    def fluid_velocity_dirichlet_values(self):
        return [self.U_F,self.U_F]
    
    def fluid_velocity_dirichlet_boundaries(self):
        return [influid]

    def fluid_pressure_dirichlet_values(self):
        return [self.P_F]
    
    def fluid_pressure_dirichlet_boundaries(self):
        return [influidfsi]
    
class NewtonAnalyticFluid(SetStruc,SetMesh,NewtonAnalytic):
    "Fluid Only analytic problem"
    def __init__(self,num_refine,bctype = "normal",endtime = 0.1):
        NewtonAnalytic.__init__(self,num_refine,bctype = bctype,endtime = endtime)

class NewtonAnalyticStruc(SetFluid,SetMesh,NewtonAnalytic):
    "Struc Only analytic problem"
    def __init__(self,num_refine,bctype = "normal",endtime = 0.1):
        NewtonAnalytic.__init__(self,num_refine,bctype = bctype,endtime = endtime)
        
class NewtonAnalyticMesh(SetFluid,SetStruc,NewtonAnalytic):
    "Mesh Only analytic problem"  
    def __init__(self,num_refine,bctype = "normal",endtime = 0.1):
        NewtonAnalytic.__init__(self,num_refine,bctype = bctype,endtime = endtime)
        
class NewtonAnalyticFluidStruc(SetMesh,NewtonAnalytic):
    "Fluid Struc only analytic problem"  
    def __init__(self,num_refine,bctype = "normal",endtime = 0.1):
        NewtonAnalytic.__init__(self,num_refine,bctype = bctype,endtime = endtime)
        
class NewtonAnalyticStrucMesh(SetFluid,NewtonAnalytic):
    "Mesh Struc Only analytic problem"
    def __init__(self,num_refine,bctype = "normal",endtime = 0.1):
        NewtonAnalytic.__init__(self,num_refine,bctype = bctype,endtime = endtime)
        
class NewtonAnalyticMeshFluid(SetStruc,NewtonAnalytic):
    "Fluid Mesh Only analytic problem"  
    def __init__(self,num_refine,bctype = "normal",endtime = 0.1):
        NewtonAnalytic.__init__(self,num_refine,bctype = bctype,endtime = endtime)
    
if __name__ == "__main__":
    mode = "solve"

    from fsinewton.solver.default_params import solver_params
    solver_params["plot"]= False
##    solver_params["plotruntime"] = "results/"
##    solver_params["save"] = "results/"
    solver_params["solve"] = True
    solver_params["reuse_jacobian"] = True
    solver_params["jacobian"] = "auto"
    solver_params["plot"] = True 
    bctype = "dirichlet"
    
    #Define and solve problem
    problem = NewtonAnalyticStruc(num_refine = 1,bctype = bctype, endtime = 0.1)
    solver = sfsi.FSINewtonSolver(problem,solver_params)

    if mode == "solve":
        solver.solve()
        U_F = solver.U1.split()[0]
        P_F = solver.U1.split()[1]
        U_S = solver.U1.split()[3]
        U_M = solver.U1.split()[5]
        
    elif mode == "int":
        problem.update(0.0,0.02,0.02)
        U_F = interpolate(problem.U_F,solver.V_FC)       
        P_F = interpolate(problem.P_F,solver.Q_FC)
        U_S = interpolate(problem.U_S,solver.V_SC)
        U_M = interpolate(problem.U_M,solver.V_MC)
    
    print "\nError Norms\n" 
    print "U_F ",errornorm(U_F,problem.U_F, mesh = problem.fluidmesh)
    print "P_F ",errornorm(P_F,problem.P_F, mesh = problem.fluidmesh)
    print "U_S ",errornorm(U_S,problem.U_S, mesh = problem.strucmesh)
    print "U_M ",errornorm(U_M,problem.U_M, mesh = problem.fluidmesh)

    #Plot the difference in the structure equation
##    plot(U_M - problem.U_M,title ="diff U_M",mesh = problem.fluidmesh)#,mode = "displacement")
    plot(U_S - problem.U_S,title ="diff U_S", mesh = problem.strucmesh, mode = "displacement")
    plot(U_S ,title ="U_S FEM", mesh = problem.strucmesh, mode = "displacement")
    plot(problem.U_S,title ="U_S analytic", mesh = problem.strucmesh, mode = "displacement")

    test_point = (2.0,0.8)
    U_S_E = interpolate(problem.U_S,solver.spaces.V_SC)
    print U_S(test_point) - U_S_E((test_point))
    interactive()
