__author__ = "Kristoffer Selim andAnders Logg"
__copyright__ = "Copyright (C) 2010 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

# Last changed: 2011-02-12

from dolfin import *
from numpy import array, append
from cbc.common import CBCProblem

from fsisolver import FSISolver
from parameters import default_parameters

class FSI(CBCProblem):
    "Base class for all FSI problems"

    def __init__(self, mesh):
        "Create FSI problem"

        # Initialize base class
        CBCProblem.__init__(self)

        # Store original mesh
        self._original_mesh = mesh
        self.Omega = None

    def solve(self, parameters=default_parameters()):
        "Solve and return computed solution (u_F, p_F, U_S, P_S, U_M, P_M)"

        # Create submeshes and mappings (only first time)
        if self.Omega is None:
            self.init_meshes(self._original_mesh, parameters)

        # Create solver
        solver = FSISolver(self)

        # Solve
        return solver.solve(parameters)

    def init_meshes(self, Omega, parameters):
        "Create mappings between submeshes"

        info("Exracting fluid and structure submeshes")

        # Set global mesh
        self.Omega = Omega

        # Create cell markers (0 = fluid, 1 = structure)
        D = Omega.topology().dim()
        cell_domains = MeshFunction("uint", self.Omega, D)
        cell_domains.set_all(0)
        structure = self.structure()
        structure.mark(cell_domains, 1)

        # Extract submeshes for fluid and structure
        Omega_F = SubMesh(self.Omega, cell_domains, 0)
        Omega_S = SubMesh(self.Omega, cell_domains, 1)

        info("Computing mappings between submeshes")

        # Extract matching indices for fluid and structure
        structure_to_fluid = compute_vertex_map(Omega_S, Omega_F)

        # Extract matching indices for fluid and structure
        fluid_indices = array([i for i in structure_to_fluid.itervalues()])
        structure_indices = array([i for i in structure_to_fluid.iterkeys()])

        # Extract matching dofs for fluid and structure (for vector P1 elements)
        structure_element_degree = parameters["structure_element_degree"]
        if structure_element_degree == 1:
            fdofs = append(fluid_indices, fluid_indices + Omega_F.num_vertices())
            sdofs = append(structure_indices, structure_indices + Omega_S.num_vertices())
        elif structure_element_degree == 2:
            error("Not implemented yet.")
        else:
            error("Only know how to map dofs for P1 and P2 elements.")

        info("Computing FSI boundary and orientation markers")

        # Initialize FSI boundary and orientation markers
        Omega.init(D - 1, D)
        fsi_boundary = MeshFunction("uint", Omega, D - 1)
        fsi_boundary.set_all(0)
        fsi_orientation = Omega.data().create_mesh_function("facet orientation", D - 1)
        fsi_orientation.set_all(0)

        # Compute FSI boundary and orientation markers
        for facet in facets(Omega):

            # Skip facets on the boundary
            cells = facet.entities(D)
            if len(cells) == 1:
                continue
            elif len(cells) != 2:
                error("Strange, expecting two facets!")

            # Create the two cells
            c0, c1 = cells
            cell0 = Cell(Omega, c0)
            cell1 = Cell(Omega, c1)

            # Get the two midpoints
            p0 = cell0.midpoint()
            p1 = cell1.midpoint()

            # Check if the points are inside
            p0_inside = structure.inside(p0, False)
            p1_inside = structure.inside(p1, False)

            # Just set c0, will be set only for FSI facets below
            fsi_orientation[facet.index()] = c0

            # Look for points where exactly one is inside the structure
            if p0_inside and not p1_inside:
                fsi_boundary[facet.index()] = 2
                fsi_orientation[facet.index()] = c1
            elif p1_inside and not p0_inside:
                fsi_boundary[facet.index()] = 2
                fsi_orientation[facet.index()] = c0
            elif p0_inside and p1_inside:
                fsi_boundary[facet.index()] = 1
            else:
                fsi_boundary[facet.index()] = 0

        # Store data
        self.Omega_F = Omega_F
        self.Omega_S = Omega_S
        self.cell_domains = cell_domains
        self.fdofs = fdofs
        self.sdofs = sdofs
        self.fsi_boundary = fsi_boundary
        self.fsi_orientation = fsi_orientation

    def mesh(self):
        "Return mesh for full domain"
        return self.Omega

    def fluid_mesh(self):
        "Return mesh for fluid domain"
        return self.Omega_F

    def structure_mesh(self):
        "Return mesh for structure domain"
        return self.Omega_S

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
