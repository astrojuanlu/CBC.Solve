__author__ = "Kent-Andre Mardal"
__copyright__ = "Copyright (C) 2011 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

from scipy.interpolate import splrep, splev

from cbc.flow import *

class Inflow(Expression):

    def __init__(self, V, problem):

        self.problem = problem
        self.V = problem.V
        self.mesh = problem.V.mesh()
        self.t_period = 1

	t  = array([0., 27., 42., 58., 69., 88., 110., 130.,
                    136., 168., 201., 254., 274., 290., 312., 325.,
                    347., 365., 402., 425., 440., 491., 546., 618.,
                    703., 758., 828., 897., 1002.])/(75/60.0)/1000

	scale = 750

	# Create interpolated mean velocity in time
	v = array([ 390.,         398.76132931, 512.65861027, 642.32628399,
                    710.66465257, 770.24169184, 779.00302115, 817.55287009,
                    877.12990937, 941.96374622, 970.        , 961.2386707 ,
                    910.42296073, 870.12084592, 843.83685801, 794.7734139 ,
                    694.89425982, 714.16918429, 682.62839879, 644.07854985,
                    647.58308157, 589.75830816, 559.96978852, 516.16314199,
                    486.37462236, 474.10876133, 456.58610272, 432.05438066,
                    390.])/574.211239628*scale

        self.inflow = splrep(t, v)

    def eval_cell(self, values, x, ufc_cell):
        cell = Cell(self.mesh, ufc_cell.index)
        n = cell.normal(ufc_cell.local_facet)

	t = self.problem.t
	val = splev(t - int(t/self.t_period)*self.t_period, self.inflow)
	values[0] = -n.x()*val
	values[1] = -n.y()*val
	values[2] = -n.z()*val

    def value_shape(self):
        return (3,)

class Aneurysm(NavierStokes):

    def mesh(self):
        return Mesh("logg.xml.gz")

    def viscosity(self):
        return 3.5
#        return 0.00345

    def velocity_dirichlet_values(self):
        self.g = Inflow(self)
        return [self.g]

    def velocity_dirichlet_boundaries(self):
        return [3]

    def velocity_initial_condition(self):
        return (0, 0, 0)

    def pressure_dirichlet_values(self):
        return 0

    def pressure_dirichlet_boundaries(self):
        return [2, 4]


    def pressure_initial_condition(self):
        return 0

    def end_time(self):
        return 1.0

    def functional(self, u, p):
        return u((0.75, 0.75))[0]

    def reference(self, t):
        return -0.0780739691918

    def __str__(self):
        return "Aneurysm (3D)"

# Solve problem
problem = Aneurysm()
problem.parameters["solver_parameters"]["plot_solution"] = True
problem.parameters["solver_parameters"]["save_solution"] = False
u, p = problem.solve()

# Check error
e = problem.functional(u, p) - problem.reference(0.5)
print "Error is", e
