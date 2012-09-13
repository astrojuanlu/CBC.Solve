__author__ = "Gabriel Balaban"
__copyright__ = "Copyright (C) 2010 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

import io
from dolfin import *
from cbc.swing.fsiproblem import FSI
from cbc.swing.fsinewton.solver.solver_fsinewton import FSINewtonSolver
from cbc.swing.parameters import read_parameters

application_parameters = read_parameters()
application_parameters["primal_solver"] = "Newton"
application_parameters["output_directory"] = "results_bloodvessel"
application_parameters["global_storage"] = True
application_parameters["solve_dual"] = False
application_parameters["estimate_error"] = False
application_parameters["uniform_timestep"] = True
application_parameters["initial_timestep"] = 0.5
application_parameters["plot_solution"] = False
application_parameters["iteration_tolerance"] = 1.0e-6
application_parameters["max_num_refinements"] = 0
application_parameters["FSINewtonSolver"]["optimization"]["reuse_jacobian"] = True
application_parameters["FSINewtonSolver"]["optimization"]["simplify_jacobian"] = False
application_parameters["FSINewtonSolver"]["optimization"]["reduce_quadrature"] = 0
application_parameters["FSINewtonSolver"]["jacobian"] = "manual"

#Fixpoint parameters
application_parameters["fluid_solver"] = "taylor-hood"
##application_parameters["fluid_solver"] = "ipcs" #GB this one fails to converge at some point.

# Constants related to the geometry of the problem
vessel_length  = 6.0
vessel_height  = 1.0
wall_thickness  = 0.1
fineness = 1
ny = int(vessel_height*10*fineness)
nx = int(vessel_length *10*fineness)

C = 0.05
# Define boundaries
#Exclude FSI Nodes
influid =  "x[1] >  %g + DOLFIN_EPS && \
            x[1] < %g - DOLFIN_EPS"%(wall_thickness, vessel_height - wall_thickness)

inflow = "near(x[0],0) && %s"%influid

outflow = "x[0] >= %g - DOLFIN_EPS && %s"%(vessel_length,influid)
 
struc_left = "x[0] <= DOLFIN_EPS"
struc_right = "x[0] >= %g - DOLFIN_EPS" %(vessel_length)

meshbc = "on_boundary &&" + influid
noslip = "on_boundary && !(%s) && !(%s)" % (inflow,outflow)

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

# Define structure subdomain
class Structure(SubDomain):
    def inside(self, x, on_boundary):
        #top or bottom vessel wall
        return \
            x[1] < wall_thickness + DOLFIN_EPS or \
            x[1] > vessel_height - wall_thickness - DOLFIN_EPS

#Define Fluid Neumann Boundary
class FluidNeumann(SubDomain):
    def inside(self,x,on_boundary):
        return near(x[0],0) and x[1] >= -DOLFIN_EPS + wall_thickness and \
                x[1] <= DOLFIN_EPS + vessel_height - wall_thickness
    
#Define Fluid Do Nothing Boundary
class FluidDN(SubDomain):
    def inside(self,x,on_boundary):
        return x[0] >= vessel_length - DOLFIN_EPS and \
               x[1] >= -DOLFIN_EPS +  wall_thickness and \
               x[1] <= vessel_height - wall_thickness + DOLFIN_EPS

class BothBoundary(SubDomain):
    def inside (self,x,on_boundary):
        return FluidDN().inside(x,on_boundary) or FluidNeumann().inside(x,on_boundary)

class BloodVessel2D(FSI):
    def __init__(self):
        mesh = Rectangle(0.0, 0.0, vessel_length, vessel_height, nx, ny, "crossed")
        self.P_Fwave = Expression(cpp_P_Fwave)
        self.P_Fwave.C = C
        self.endtime = 70.0
        self.rho_S =  4.0
        FSI.__init__(self,mesh,application_parameters)
        
    def update(self, t0, t1, dt):
        self.P_Fwave.t = t1

    #--- Common ---
    def end_time(self):
        return self.endtime
    
    def __str__(self):
        return "Blood Vessel"

#--- Material Parameters---
    def fluid_density(self):
        return 1.0

    def fluid_viscosity(self):
        return 0.002
    
    def structure_density(self):  
        return self.rho_S
    
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
 
# Define and solve problem
mode = "normal"
##mode = "normal"
if __name__ == "__main__":
    if mode == "normal":
        problem = BloodVessel2D()
        problem.solve(application_parameters)
        interactive()
    elif mode == "stresstest":
        #Init plotting
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages

        num_steps = 7
        iterations = {"fixpoint":[],"Full Newton":[],"Reuse Newton":[],"Simple Newton":[]}
        stepsizes = []
        densities = []
        rho_S = 2.0
        application_parameters["initial_timestep"] = 1.0
        test = "time_step"
##        test = "density"
        for i in range(num_steps):
            #solve problem with fixpoint method
            problem = BloodVessel2D()
            problem.endtime = application_parameters["initial_timestep"]
            problem.rho_S = rho_S
            application_parameters["primal_solver"] = "fixpoint"
            problem.solve(application_parameters) 
            g_numiter = problem.solver.primalsolver.g_numiter
            iterations["fixpoint"].append(g_numiter)

####          #solve problem with Full Newton method
            application_parameters["primal_solver"] = "Newton"
            application_parameters["FSINewtonSolver"]["optimization"]["reuse_jacobian"] = False
            application_parameters["FSINewtonSolver"]["optimization"]["simplify_jacobian"] = False
            problem = BloodVessel2D()
            problem.endtime = application_parameters["initial_timestep"]
            problem.rho_S = rho_S
            problem.solve(application_parameters) 
            #extract the number of iterations
            g_numiter = problem.solver.primalsolver.g_numiter
            iterations["Full Newton"].append(g_numiter) 
##
####          #solve problem with Reuse Newton method
            application_parameters["primal_solver"] = "Newton"
            application_parameters["FSINewtonSolver"]["optimization"]["reuse_jacobian"] = True
            application_parameters["FSINewtonSolver"]["optimization"]["simplify_jacobian"] = False
            problem = BloodVessel2D()
            problem.endtime = application_parameters["initial_timestep"]
            problem.rho_S = rho_S
            problem.solve(application_parameters) 
            #extract the number of iterations
            g_numiter = problem.solver.primalsolver.g_numiter
            iterations["Reuse Newton"].append(g_numiter) 

##          #solve problem with Simplify Newton method
##            application_parameters["primal_solver"] = "Newton"
##            application_parameters["FSINewtonSolver"]["optimization"]["reuse_jacobian"] = False
##            application_parameters["FSINewtonSolver"]["optimization"]["simplify_jacobian"] = True
##            problem = BloodVessel2D()
##            problem.endtime = application_parameters["initial_timestep"]
##            problem.rho_S = rho_S
##            problem.solve(application_parameters) 
##            #extract the number of iterations
##            g_numiter = problem.solver.primalsolver.g_numiter
##            iterations["Simple Newton"].append(g_numiter) 

            stepsizes.append(application_parameters["initial_timestep"])
            densities.append(rho_S)
##                
            #Plotting
            plt.figure()
            pdf = PdfPages("Itervsstep")
            if test == "time_step":
                plt.xlabel("Time Step  Size")
            elif test == "density":
                plt.xlabel("Densities")
            ax = plt.gca()
            ax.grid()
            plt.ylabel("Number of iterations")

            if test == "density":
                ax.set_xlim(ax.get_xlim()[::-1]) #reverse axis
                plt.plot(densities,iterations["fixpoint"],'bD',label = "Fixpoint",linestyle = '-')
                plt.plot(densities,iterations["Full Newton"],'gp',label = "Newton",linestyle = '-')
            elif test == "time_step":
                plt.plot(stepsizes,iterations["fixpoint"],'bD',label = "Fixpoint",linestyle = '-')
                plt.plot(stepsizes,iterations["Full Newton"],'gp',label = "Full Newton",linestyle = '-')
                plt.plot(stepsizes,iterations["Reuse Newton"],'k2',label = "Reuse Newton",linestyle = '-')
##                plt.plot(stepsizes,iterations["Simple Newton"],'r*',label = "Simplified Newton",linestyle = '-')
                 
            plt.title("2D Blood Vessel solver stress test, "+r'$\rho_S = 2$')
            plt.legend(loc=0)
            plt.savefig(pdf, format ='pdf')
            pdf.close()

            if test == "time_step":
                application_parameters["initial_timestep"] += 0.5          
            elif test == "density":
                rho_S = rho_S - 0.5
