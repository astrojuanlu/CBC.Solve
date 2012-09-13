__author__ = "Gabriel Balaban"
__copyright__ = "Copyright (C) 2010 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

import io
from dolfin import *
from cbc.swing.fsiproblem import MeshLoadFSI
from cbc.swing.fsinewton.solver.solver_fsinewton import FSINewtonSolver
from cbc.swing.parameters import read_parameters

#parameters['form_compiler']["name"] ="sfc" 

application_parameters = read_parameters()
application_parameters["primal_solver"] = "Newton"
application_parameters["output_directory"] = "results_3Dtube"
application_parameters["global_storage"] = True
application_parameters["solve_dual"] = False
application_parameters["estimate_error"] = False
application_parameters["uniform_timestep"] = True
application_parameters["initial_timestep"] = 0.5
application_parameters["plot_solution"] = True
application_parameters["iteration_tolerance"] = 1.0e-6
application_parameters["FSINewtonSolver"]["optimization"]["max_reuse_jacobian"] = 40
application_parameters["FSINewtonSolver"]["optimization"]["simplify_jacobian"] = False
application_parameters["FSINewtonSolver"]["optimization"]["reduce_quadrature"] = 2
application_parameters["FSINewtonSolver"]["newtonitrmax"] = 180
application_parameters["FSINewtonSolver"]["plot"] = True
#Fixpoint parameters
application_parameters["fluid_solver"] = "taylor-hood"

C = 1.0
#Presure Wave
cpp_P_Fwave = """
class P_F : public Expression
{
public:

  P_F() : Expression(), C(0), t(0) {}

  void eval(Array<double>& values, const Array<double>& xx,
            const ufc::cell& cell) const
  {
    const double x = xx[0];

    if (x < 0.25*t - 0.2 ){
        values[0] = 0;
    }
    else if (x < 0.25*t + 0.2){ 
        values[0] = C*cos((x - 0.25*t)*3.141519*0.5*0.2);
    }
    else {
        values[0] = 0;
          }
  }

  double C;
  double t;

};
"""

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
        self.P_Fwave = Expression(cpp_P_Fwave,C=C, t=0)
        self.P_Fwave.C = C
        
        MeshLoadFSI.__init__(self,mesh,meshdomains)
                                   
    def update(self, t0, t1, dt):
        self.P_Fwave.t = t1

    #--- Common ---
    def end_time(self):
        return 70.00
    
    def __str__(self):
        return "Blood Vessel"
    
    #This can be done by the class FixedPointFSI after integration
    def initial_step(self):
        return application_parameters["initial_timestep"]

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
        #return (0.0,)*spatial_dimension
        return (0.0, 0.0, 0.0)

    def fluid_pressure_initial_condition(self):
       return (0.0)
    
    def fluid_pressure_dirichlet_boundaries(self):
       return ["GammaFSI"]
    
    def fluid_pressure_dirichlet_values(self):
        return [self.P_Fwave]

    def fluid_donothing_boundaries(self):
        return [LEFTINFLOW,RIGHTINFLOW]

    def structure_dirichlet_values(self):
        return [(0,0,0),(0,0,0)]
    
    def structure_dirichlet_boundaries(self):
        return [LEFTSTRUC,RIGHTSTRUC]

# Define and solve problem
if __name__ == "__main__":
    problem = BloodVessel3D()
    solver = FSINewtonSolver(problem,application_parameters["FSINewtonSolver"].to_dict())
    solver.solve()
    interactive()
