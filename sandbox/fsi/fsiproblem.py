__author__ = "Kristoffer Selim and Anders Logg"
__copyright__ = "Copyright (C) 2010 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

# Last changed: 2010-09-13

from dolfin import *
from numpy import array, append
from cbc.common import CBCProblem

from fsisolver import FSISolver

class FSI(CBCProblem):
    "Base class for all FSI problems"

    def __init__(self, mesh):
        "Create FSI problem"

        # Initialize base class
        CBCProblem.__init__(self)

        # Create solver
        self.solver = FSISolver(self)

        # Set up parameters
        self.parameters = Parameters("problem_parameters")
        self.parameters.add(self.solver.parameters)

        # Create submeshes and mappings
        self.init_meshes(mesh)

    def solve(self):
        "Solve and return computed solution (u_F, p_F, U_S, P_S, U_M, P_M)"

        # Update solver parameters
        self.solver.parameters.update(self.parameters["solver_parameters"])

        # Call solver
        return self.solver.solve()

    def init_meshes(self, Omega):
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
        fdofs = append(fluid_indices, fluid_indices + Omega_F.num_vertices())
        sdofs = append(structure_indices, structure_indices + Omega_S.num_vertices())

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
            if facet.num_entities(D) == 1:
                continue

            # Create the two cells
            c0, c1 = facet.entities(D)
            cell0 = Cell(Omega, c0)
            cell1 = Cell(Omega, c1)

            # Get the two midpoints
            p0 = cell0.midpoint()
            p1 = cell1.midpoint()

            # Check if the points are inside
            p0_inside = structure.inside(p0, False)
            p1_inside = structure.inside(p1, False)

            # Look for points where exactly one is inside the structure
            if p0_inside and not p1_inside:
                fsi_boundary[facet.index()] = 1
                fsi_orientation[facet.index()] = c1
            elif p1_inside and not p0_inside:
                fsi_boundary[facet.index()] = 1
                fsi_orientation[facet.index()] = c0
            else:
                # Just set c0, will not be used
                fsi_orientation[facet.index()] = c0

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
