__author__ = "Gabriel Balaban"
__copyright__ = "Copyright (C) 2010 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

import io
from dolfin import *
from cbc.swing.fsiproblem import MeshLoadFSI
from cbc.swing.fsinewton.solver.solver_fsinewton import FSINewtonSolver
from cbc.swing.parameters import read_parameters

application_parameters = read_parameters()
application_parameters["primal_solver"] = "Newton"

application_parameters["output_directory"] = "results_3Dtube"
application_parameters["global_storage"] = True
application_parameters["solve_dual"] = False
application_parameters["estimate_error"] = False
application_parameters["uniform_timestep"] = True
application_parameters["initial_timestep"] = 0.5 #Newton Solver
application_parameters["plot_solution"] = True
application_parameters["iteration_tolerance"] = 1.0e-6
application_parameters["FSINewtonSolver_parameters"]["optimization"]["max_reuse_jacobian"] = 40
application_parameters["FSINewtonSolver_parameters"]["optimization"]["simplify_jacobian"] = False
application_parameters["FSINewtonSolver_parameters"]["newtonitrmax"] = 180
application_parameters["FSINewtonSolver_parameters"]["plot"] = True
#Fixpoint parameters
application_parameters["fluid_solver"] = "taylor-hood"

#Presure Wave
from demo.swing.bloodvessel2d.bloodvessel2d import cpp_P_Fwave
C = 1.0

#Cell domains
FLUIDDOMAIN = 0
STRUCTUREDOMAIN = 1

#Facet Domains
FSIINTERFACE = 1 
STRUCTUREOUTERWALL = 2
RIGHTINFLOW = 3
LEFTINFLOW = 4
LEFTSTRUC = 5
RIGHTSTRUC = 6

meshdomains = {"fluid":[FLUIDDOMAIN],
               "structure":[STRUCTUREDOMAIN],
               "FSI_bound":[FSIINTERFACE],
               "strucbound":[STRUCTUREOUTERWALL],
               "donothingbound":[RIGHTINFLOW,LEFTINFLOW],
               "fluidneumannbound":[]}

class BloodVessel3D(MeshLoadFSI):
    def __init__(self):
        mesh = Mesh("mesh.xml")
        self.structure = self.__get_structure_domain(mesh)
        exit()
        self.P_Fwave = Expression(cpp_P_Fwave)
        self.P_Fwave.C = C
        MeshLoadFSI.__init__(self,mesh,meshdomains)

    def __get_structure_domain(self,mesh):
        domains = mesh.domains()
        cell_domains = domains.cell_domains(mesh)
        structure = SubDomain(mesh,cell_domains,STRUCTUREDOMAIN)
        return structure
                                   
    def update(self, t0, t1, dt):
        self.P_Fwave.t = t1

    #--- Common ---
    def end_time(self):
        return 70.00
    
    def __str__(self):
        return "Blood Vessel"

#--- Material Parameters---
    def fluid_density(self):
        return 1.0

    def fluid_viscosity(self):
        return 0.002
    
    def structure_density(self):  
        return 4.0
    
    def structure_mu(self):
        return 5.0

    def structure_lmbda(self):
        return 2.0

    def mesh_mu(self):
        return 100.0

    def mesh_lmbda(self):
        return 100.0

    #--- Fluid problem BC---
    def fluid_velocity_initial_condition(self):
        return (0.0, 0.0)

    def fluid_pressure_initial_condition(self):
       return (0.0)
    
    def fluid_pressure_dirichlet_boundaries(self):
        if application_parameters["primal_solver"] == "Newton": return ["GammaFSI"]
        else: return [noslip]

    def fluid_pressure_dirichlet_values(self):
        return [self.P_Fwave]

    def fluid_donothing_boundaries(self):
        return [BothBoundary()]

    def structure(self):
        return Structure()

    def structure_dirichlet_values(self):
        return [(0,0),(0,0)]
    
    def structure_dirichlet_boundaries(self):
        return [struc_left,struc_right]

##    #--- Mesh problem BC---
##    def mesh_dirichlet_boundaries(self):
##        return [meshbc]
    
# Define and solve problem
if __name__ == "__main__":
    problem = BloodVessel3D()
    solver = FSINewtonSolver(problem,application_parameters["FSINewtonSolver_parameters"])
    solver.solve()
    interactive()
