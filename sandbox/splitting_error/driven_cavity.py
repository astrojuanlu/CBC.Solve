__author__ = "Kristoffer Selim and Anders Logg"
__copyright__ = "Copyright (C) 2010 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

# Last changed: 2011-01-31

from fsiproblem import *

#   -------------------------> u = (1,0)
#   
#   |                        |
#   |                        |
#   |                        |
#   |        FLUID           |
#   |                        |
#   |     (Unit Square)      |
#   |                        |  
#   |                        |
#   __________________________ 


# Create application parameters set
application_parameters = Parameters("application_parameters")
application_parameters.add("mesh_scale", 4)
application_parameters.add("end_time", 1.0)
application_parameters.add("dt", 0.25)
application_parameters.add("TOL", 0.1)
application_parameters.add("w_h", 0.5) 
application_parameters.add("w_k", 0.5)
application_parameters.add("w_c", 0.1) # Passive in this problem!
application_parameters.add("fraction", 0.3)
application_parameters.add("solve_primal", True)
application_parameters.add("solve_dual", True)
application_parameters.add("estimate_error", True)
application_parameters.add("dorfler_marking", True)
application_parameters.add("uniform_timestep", False)
application_parameters.parse()

# Collect parameters
parameter_info = application_parameters.option_string()

# Define boundaries
inflow = "x[1] == 1.0 " 
noslip = "on_boundary && !(%s)" % inflow

# Define problem class 
class DrivenCavity(FSI):
    def __init__(self):
        
        # Define mesh based on a scale factor 
        mesh_scale = application_parameters["mesh_scale"]
        mesh =  UnitSquare(mesh_scale, mesh_scale)

        # Save original mesh
        file = File("adaptivity/mesh_0.xml")
        file << mesh

        # Report problem parameters
        mesh_size = mesh.hmin()
        f = open("adaptivity/drivencavity.txt", "w")
        f.write(str("Mesh size:  ") + (str(mesh_size)) + "\n \n")
        f.write(parameter_info)
        f.close()

        # Initialize base class
        FSI.__init__(self, mesh)


    #--- Solver options ---

    def solve_primal(self):
        return application_parameters["solve_primal"]

    def solve_dual(self):
        return application_parameters["solve_dual"]

    def estimate_error(self):
        return application_parameters["estimate_error"]
    
    def dorfler_marking(self):
        return application_parameters["dorfler_marking"]

    def uniform_timestep(self):
        return application_parameters["uniform_timestep"]


    #--- Common parameters ---

    def mesh_scale(self):
        return application_parameters["mesh_scale"]

    def end_time(self):
        return application_parameters["end_time"]

    def TOL(self):
        return application_parameters["TOL"]

    def initial_timestep(self):
        return application_parameters["dt"]

    def space_error_weight(self):
        return application_parameters["w_h"]

    def time_error_weight(self):
        return application_parameters["w_k"]

    def non_galerkin_error_weight(self):
        return application_parameters["w_c"]

    def fraction(self):
        return application_parameters["fraction"]

    def evaluate_functional(self, u, p, dx, ds, t1):
        "Evaluates the goal functional in the primal problem"
        "and defines the goal funcional in the dual problem"

        # Define the Riezs' reprsenter for the goal functional
        psi = Expression("exp(-(pow(25*(x[0] - 0.75), 2) + pow(25*(x[1] - 0.25), 2)) / 5.0)")
        
        # Define the goal functional
        goal_functional = psi*(u[0] + u[1])*dx

        return goal_functional

    def __str__(self):
        return "Driven Cavity test case"


 #--- Fluid parameters ---
    
    def density(self):
        return 1.0

    def viscosity(self):
        return 0.005

    def velocity_dirichlet_values(self):
        return [(0.0, 0.0), (1.0, 0.0)]

    def velocity_dirichlet_boundaries(self):
        return [noslip, inflow]

    def pressure_dirichlet_values(self):
        return [0]

    def pressure_dirichlet_boundaries(self):
        return [inflow]

    def velocity_initial_condition(self):
        return (0.0, 0.0)

    def pressure_initial_condition(self):
        return 0.0


# Define problem
problem = DrivenCavity()
problem.parameters["solver_parameters"]["solve_primal"] = problem.solve_primal()
problem.parameters["solver_parameters"]["solve_dual"] = problem.solve_dual() 
problem.parameters["solver_parameters"]["estimate_error"] = problem.estimate_error()
problem.parameters["solver_parameters"]["uniform_timestep"]  = problem.uniform_timestep()
problem.parameters["solver_parameters"]["tolerance"] = problem.TOL()

# Solve problem
u, p = problem.solve()


