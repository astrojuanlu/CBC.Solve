"""Module containing implementation of monolithic FSI boundary conditions"""
__author__ = "Gabriel Balaban"
__copyright__ = "Copyright (C) 2010 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

from dolfin import *

class FSIBC(object):
    """
    Boundary Conditions class for Monolithic FSI

    Arguments
        problem
            object of type pfsi.FsiNewtonTest
         spaces
            object of type FSISpaces
    """
    
    def __init__(self,problem,spaces):
        self.problem = problem
        self.spaces = spaces
        
        #Time dependant BC are set on the intial guess and at each time step.
        self.bcallU1_ini = self.create_all_dirichlet_conditions("Initial guess")

        #Newton Increment BC are homogeneous
        self.bcallI = self.create_all_dirichlet_conditions("Newton Step")
        [bc.homogenize() for bc in self.bcallI]
    
    def create_all_dirichlet_conditions(self, bcsetname = ""):
        info_blue("\nCreating Dirichlet Boundary Conditions " + bcsetname)
        return self.create_fluid_bc() + self.create_structure_bc() + \
               self.create_mesh_bc()

    def create_bc(self,space,boundaries,values,bcname):
        
        #If Boundaries specified without values assume homogeneous 
        if boundaries is not None and (values == [] or values is None):
            dim = space.num_sub_spaces()
            #A Function Space returns dim 0 but really has dim 1.
            if dim == 0:
                dim = 1
            zeros = tuple(["0.0" for i in range(dim)])
            values = [zeros for i in range(len(boundaries))]
        #Try to generate the BC
        bcs = []
##         try:
        for boundary,value in zip(boundaries,values):
            if boundary == 'GammaFSI':
                fsibounds = self.problem.interiorboundarynums["FSI_bound"]
                interiormeshfunc = self.problem.meshfunctions["interiorfacet"]
                for fsibound in fsibounds:
                    print fsibound
                    bcs += [DirichletBC(space, value,interiormeshfunc,fsibound)]
            else:
                bcs += [DirichletBC(space, value, boundary)]
        info("Created bc %s"%bcname)
##        except:
##            info("No Dirichlet bc created for %s"%bcname)
        return bcs

    def create_fluid_bc(self):
        bcv = self.create_bc(self.spaces.V_F,self.problem.fluid_velocity_dirichlet_boundaries(),\
                             self.problem.fluid_velocity_dirichlet_values(),"Fluid Velocity")
        bcp = self.create_fluid_pressure_bc()
        return bcv + bcp
    
    def create_fluid_pressure_bc(self):
        return self.create_bc(self.spaces.Q_F,self.problem.fluid_pressure_dirichlet_boundaries(),\
                             self.problem.fluid_pressure_dirichlet_values(),"Fluid Pressure")

    def create_structure_bc(self):
        bcU = self.create_bc(self.spaces.C_S,self.problem.structure_dirichlet_boundaries(),\
                             self.problem.structure_dirichlet_values(),"Structure Displacement")
        
        bcP = self.create_bc(self.spaces.V_S,self.problem.structure_velocity_dirichlet_boundaries(),\
                             self.problem.structure_velocity_dirichlet_values(),"Structure Velocity")
        return bcU + bcP
    
    def create_mesh_bc(self):
        #If no Mesh BC specified assume domain boundary and fixed"
        if self.problem.mesh_dirichlet_boundaries() is None:
            #The value will be set to zero in self.create_bc
            return self.create_bc(self.self.spaces.C_F,["on_boundary"],None,"Mesh Displacement")
        #Allow the user to explicitly create no mesh bc whatsoever.
        elif self.problem.mesh_dirichlet_boundaries() == "NoBC":
            return []
        else:
            return self.create_bc(self.spaces.C_F,self.problem.mesh_dirichlet_boundaries(),\
                             self.problem.mesh_dirichlet_values(),"Mesh Displacement")
        
