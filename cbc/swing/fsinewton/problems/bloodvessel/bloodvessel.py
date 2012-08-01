__author__ = "Gabriel Balaban"
__copyright__ = "Copyright (C) 2010 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

import io
from dolfin import *
from fsinewton.problems.base import FsiNewtonTest
from fsinewton.solver.solver_fsinewton import FSINewtonSolver

# Constants related to the geometry of the problem
vessel_length  = 6.0
vessel_height  = 1.0
wall_thickness  = 0.1
fineness = 1
ny = int(vessel_height*10*fineness)
nx = int(vessel_length *10*fineness)

C = 10.0
# Define boundaries
#Exclude FSI Nodes
influid =  "x[1] >  %g + DOLFIN_EPS && \
            x[1] < %g - DOLFIN_EPS"%(wall_thickness, vessel_height - wall_thickness)

inflow = "near(x[0],0) && %s"%influid

outflow = "x[0] >= %g - DOLFIN_EPS && %s"%(vessel_length,influid)
 
struc_left = "x[0] <= DOLFIN_EPS"
struc_right = "x[0] >= %g - DOLFIN_EPS" %(vessel_length)

meshbc = "on_boundary &&" + influid

cpp_G_F =  """
class G_F : public Expression
{
public:

  G_F() : Expression(2), C(0), t(0) {}

  void eval(Array<double>& values, const Array<double>& xx,
            const ufc::cell& cell) const
  {
    const double X = xx[0];
    const double Y = xx[1];

    if (t < 1.0 ){
        values[0] = C*(Y - 0.1)*(0.9 - Y)*sin(2.0*t/3.14);
        values[1] = 0.0;
    }
    else {
        values[0] = 1.0;
        values[1] = 0.0;
    }
  }
  double C;
  double t;
};
"""

#Time variable pressure on the left boundary
cpp_P_F = """
class P_F : public Expression
{
public:

  P_F() : Expression(), C(0), t(0) {}

  void eval(Array<double>& values, const Array<double>& xx,
            const ufc::cell& cell) const
  {
    const double x = xx[0];

    if (t < 5.0 ){
        values[0] = 1.0;
    }
    else if (t < 15.0){ 
        values[0] = 1.0 - (t - 5.0)/11.0;
    }
    else {
        values[0] = 1.0/11.0;
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

class BloodVessel(FsiNewtonTest):
    def __init__(self):
        mesh = Rectangle(0.0, 0.0, vessel_length, vessel_height, nx, ny, "crossed")
        self.G_F = Expression(cpp_G_F)
        self.P_F = Expression(cpp_P_F)
        self.G_F.C = C
        self.P_F.C = C
        FsiNewtonTest.__init__(self,mesh,Structure())
        
    def update(self, t0, t1, dt):
        self.P_F.t = t1

    #--- Common ---
    def initial_step(self):
        return 0.5

    def end_time(self):
        return 20.0

    def __str__(self):
        return "Blood Vessel"

#--- Material Parameters ---
    def fluid_density(self):
        return 1.0

    def fluid_viscosity(self):
        return 0.2
    
    def structure_density(self):
        return 20.0

    def structure_mu(self):
        return 40.0

    def structure_lmbda(self):
        return 2.0

    def mesh_mu(self):
        return 100.8461

    def mesh_lmbda(self):
        return 100.76

    #--- Fluid problem BC---
    def fluid_pressure_dirichlet_boundaries(self):
        return [inflow,outflow]
##    
##    def fluid_pressure_dirichlet_values(self):
##        return [1.0,0.0]
    def fluid_pressure_dirichlet_values(self):
        return [self.P_F,0.0]
    
    def fluid_donothing_boundaries(self):
        return BothBoundary()

##    def fluid_velocity_neumann_boundaries(self):
##        return FluidNeumann()
##
##    def fluid_velocity_neumann_values(self):
##        return self.G_F
##        
##    def fluid_donothing_boundaries(self):
##        return FluidDN()

    #--- Structure problem BC---

    def structure(self):
        return Structure()

    def structure_dirichlet_values(self):
        return [(0,0),(0,0)]
    
    def structure_dirichlet_boundaries(self):
        return [struc_left,struc_right]

    #--- Mesh problem BC---
    def mesh_dirichlet_boundaries(self):
        return [meshbc]
    
# Define and solve problem
if __name__ == "__main__":
    problem = BloodVessel()
    from fsinewton.solver.default_params import solver_params
    storefolder = "fastresults"
    solver_params["plot"]= True
    solver_params["store"] = storefolder
    solver_params["runtimedata"]["newtonsolver"] = storefolder
    solver_params["runtimedata"]["fsisolver"] = storefolder
    solver_params["reuse_jacobian"] = True
    solver_params["jacobian"] = "auto"
    solver_params["max_reuse_jacobian"] = 60
#    solver_params["solve"] = False
    solver_params["newtonitrmax"] = 180
    
    solver_params["newtonsoltol"] = 1.0e-6
    
    solver = FSINewtonSolver(problem,solver_params)
    solver.solve()
