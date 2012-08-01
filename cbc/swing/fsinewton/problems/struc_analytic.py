"""
An analytic test problem to test the structure block of the FSINewtonSolver
A 1 by pi bar is bent into a bow shape with displacement (0,sinx).
"""

__author__ = "Gabriel Balaban"
__copyright__ = "Copyright (C) 2012 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

from dolfin import *
from fsinewton.problems.base import FsiNewtonTest
import numpy as np
import cbc.twist as twi

meshlength = np.pi
nx = 15
meshheight = 12/float(nx)
ny = 4
strucheight = meshheight/2.0

#Wierd that formulating xright with near doesnt pass the setup test.
xright = "x[0] > %g - DOLFIN_EPS"%(meshlength)
xleft = "near(x[0],0.0)"

#Define Structure Subdomain
class Structure(SubDomain):
    def inside(self, x, on_boundary):
        return x[1] < strucheight + DOLFIN_EPS

#Structure Body Force
cpp_f_S = """
class f_S : public Expression
{
public:

  f_S() : Expression(2), t(0) {}

  void eval(Array<double>& values, const Array<double>& xx,
            const ufc::cell& cell) const
  {
    const double X = xx[0];
    const double Y = xx[1];
    const double a = sin(t);
    const double s = sin(X);
    const double c = cos(X);

    values[0] = (pow(a,3)/64.0)*(2.0*c*s+5.0*s);
    values[1] = (pow(a,3)/64.0)*(3.0*pow(c,2)*s+10.0*c*s+s) -a*s*0.75;
  }

  double t;

};
"""

class StrucAnalytic(FsiNewtonTest):
    """Analytic Structure Problem"""
    def __init__(self):
        mesh = Rectangle(0.0,0.0,meshlength,meshheight,nx,ny)
        self.cpp_f_S = cpp_f_S
        FsiNewtonTest.__init__(self,mesh,Structure())

    def __str__(self):
        return """Analytic Structure Problem"""

    #--- Structure problem ---
    def structure(self):
        return Structure()

    def structure_density(self):
        return 1.0

    def structure_dirichlet_boundaries(self):
        return [xleft,xright]

    def structure_lmbda(self):
        return 10.0
        
    def initial_step(self):
        return 0.05

class TimeStrucAnalyticTwist(twi.Hyperelasticity,StrucAnalytic):
    def __init__(self):
        ##TimeStrucAnalytic.__init__(self)
        twi.Hyperelasticity.__init__(self)

        
    def analytical_solution(self):
        def anaf(t,x):
            return (0.0, sin(t)*sin(x) *0.25)
        return anaf
        
    def body_force(self):
        return Expression(self.cpp_f_S)

    def time_step(self):
        return self.initial_step()
    
    def time_stepping(self):
        return "CG1"

    def end_time(self):
        return 6.5

    def mesh(self):
        return Rectangle(0.0,0.0,meshlength,meshheight/2,nx,ny)

    def dirichlet_boundaries(self):
        return self.structure_dirichlet_boundaries()
    
    def dirichlet_values(self):
        return [("0.0", "0.0"),("0.0", "0.0")]

    def material_model(self):
        mu = self.structure_mu()
        lmbda = self.structure_lmbda()
        return twi.StVenantKirchhoff([mu, lmbda])
    #Used for the time dependant problem
    def reference_density(self):
        return self.structure_density()

class TimeStrucAnalytic(TimeStrucAnalyticTwist):
    def __init__(self):
        StrucAnalytic.__init__(self)

    def structure_body_force(self):
        return self.body_force()

if __name__ == "__main__":
    # Setup the problem
    twist = TimeStrucAnalyticTwist()

    # Solve the problem
    print twist
    u = twist.solve()
    print "Analyical solution at (pi/2)"
    anasol = twist.analytical_solution()
##    print anasol(twist.end_time(),np.pi/2.0)
##    print "FEM solution at (pi/2)"
##    print self.solver.U1_S(np.pi/2.0,0.0)
