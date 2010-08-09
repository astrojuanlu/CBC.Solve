__author__ = "Kristoffer Selim and Anders Logg"
__copyright__ = "Copyright (C) 2010 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

# Last changed: 2010-08-09

from dolfin import *
from numpy import array, append
from cbc.common import CBCProblem

from fsisolver import FSISolver

class FSI(CBCProblem):
    "Base class for all FSI problems"

    def __init__(self, parameters=None):
        "Create FSI problem"

        # Initialize base class
        CBCProblem.__init__(self)

        # Create solver
        self.solver = FSISolver(self)

        # Set up parameters
        self.parameters = Parameters("problem_parameters")
        self.parameters.add(self.solver.parameters)

        # Create mappings between submeshes
        self.init_mappings()

    def solve(self):
        "Solve and return computed solution (u_F, p_F, U_S, P_S, U_M, P_M)"

        # Update solver parameters
        self.solver.parameters.update(self.parameters["solver_parameters"])

        # Call solver
        return self.solver.solve()

    def init_mappings(self):
        "Create mappings between submeshes"

        info("Computing mappings between submeshes")

        # Get meshes
        Omega_F = self.initial_fluid_mesh()
        Omega_S = self.structure_mesh()

        # Extract matching indices for fluid and structure
        structure_to_fluid = compute_vertex_map(Omega_S, Omega_F)

        # Extract matching indices for fluid and structure
        fluid_indices = array([i for i in structure_to_fluid.itervalues()])
        structure_indices = array([i for i in structure_to_fluid.iterkeys()])

        # Extract matching dofs for fluid and structure (for vector P1 elements)
        self.fdofs = append(fluid_indices, fluid_indices + Omega_F.num_vertices())
        self.sdofs = append(structure_indices, structure_indices + Omega_S.num_vertices())

    def add_f2s(self, xs, xf):
        "Compute xs += xf for corresponding indices"
        xs_array = xs.array()
        xf_array = xf.array()
        xs_array[self.sdofs] += xf_array[self.fdofs]
        xs[:] = xs_array

    def add_s2f(self, xf, xs):
        "Compute xf += xs for corresponding indices"
        xf_array = xf.array()
        xs_array = xs.array()
        xf_array[self.fdofs] += xs_array[self.sdofs]
        xf[:] = xf_array
