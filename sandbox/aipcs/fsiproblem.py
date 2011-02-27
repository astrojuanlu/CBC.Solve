__author__ = "Kristoffer Selim andAnders Logg"
__copyright__ = "Copyright (C) 2010 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

# Last changed: 2011-02-27

from dolfin import *
from numpy import array, append
from cbc.common import CBCProblem

from fsisolver import FSISolver
from parameters import default_parameters, read_parameters

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

            # Refine original mesh
            mesh = self._original_mesh
            for i in range(parameters["num_initial_refinements"]):
                mesh = refine(mesh)

            # Initialize meshes
            self.init_meshes(mesh, parameters)

        # Create solver
        solver = FSISolver(self)

        # Solve
        return solver.solve(parameters)

    def init_meshes(self, Omega, parameters):
        "Create mappings between submeshes"

        info("Extracting fluid and structure submeshes")

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

        info("Computing mappings between submeshes")

        # Extract map from vertices in Omega to vertices in Omega_F
        vertex_map_to_fluid = {}
        vertex_map_from_fluid = Omega_F.data().mesh_function("global vertex indices")
        for i in range(vertex_map_from_fluid.size()):
            vertex_map_to_fluid[vertex_map_from_fluid[i]] = i

        info("Computing FSI boundary and orientation markers")

        # Initialize FSI boundary and orientation markers on Omega
        Omega.init(D - 1, D)
        fsi_boundary = FacetFunction("uint", Omega, D - 1)
        fsi_boundary.set_all(0)
        fsi_orientation = Omega.data().create_mesh_function("facet orientation", D - 1)
        fsi_orientation.set_all(0)

        # Initialize FSI boundary on Omega_F
        Omega_F.init(D - 1, D)
        Omega_F.init(0, 1)
        fsi_boundary_F = MeshFunction("uint", Omega_F, D - 1)
        fsi_boundary_F.set_all(0)

        # Compute FSI boundary and orientation markers on Omega
        for facet in facets(Omega):

            # Skip facets on the boundary
            cells = facet.entities(D)
            if len(cells) == 1:
                continue
            elif len(cells) != 2:
                error("Strange, expecting one or two facets!")

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

            # Markers:
            #
            # 0 = fluid
            # 1 = structure
            # 2 = FSI boundary

            # Look for points where exactly one is inside the structure
            facet_index = facet.index()
            if p0_inside and not p1_inside:
                fsi_boundary[facet_index] = 2
                fsi_orientation[facet_index] = c1
            elif p1_inside and not p0_inside:
                fsi_boundary[facet_index] = 2
                fsi_orientation[facet_index] = c0
            elif p0_inside and p1_inside:
                fsi_boundary[facet_index] = 1
            else:
                fsi_boundary[facet_index] = 0

        # Initialize global edge indices (used in read_primal_data)
        init_global_edge_indices(Omega_F, Omega)

        # Store data
        self.Omega_F = Omega_F
        self.cell_domains = cell_domains
        self.fsi_boundary = fsi_boundary
        self.fsi_orientation = fsi_orientation
        self.fsi_boundary_F = fsi_boundary_F

    def mesh(self):
        "Return mesh for full domain"
        return self.Omega

    def fluid_mesh(self):
        "Return mesh for fluid domain"
        return self.Omega_F
