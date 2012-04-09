__author__ = "Kristoffer Selim andAnders Logg"
__copyright__ = "Copyright (C) 2010 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

# Last changed: 2012-04-09

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
        Omega_S = SubMesh(self.Omega, cell_domains, 1)

        info("Computing mappings between submeshes")

        # Extract matching indices for fluid and structure
        fluid_to_structure_v = compute_vertex_map(Omega_F, Omega_S)
        fluid_to_structure_e = compute_edge_map(Omega_F, Omega_S)

        # Extract matching vertex indices for fluid and structure
        v_F = array([i for i in fluid_to_structure_v.iterkeys()])
        v_S = array([i for i in fluid_to_structure_v.itervalues()])

        # Extract matching edge indices for fluid and structure
        e_F = array([i for i in fluid_to_structure_e.iterkeys()])
        e_S = array([i for i in fluid_to_structure_e.itervalues()])

        # Extract matching dofs for fluid and structure
        structure_element_degree = parameters["structure_element_degree"]
        Nv_F = Omega_F.num_vertices()
        Nv_S = Omega_S.num_vertices()
        Ne_F = Omega_F.num_edges()
        Ne_S = Omega_S.num_edges()
        if structure_element_degree == 1:
            fdofs = append(v_F, v_F + Nv_F)
            sdofs = append(v_S, v_S + Nv_S)
        elif structure_element_degree == 2:
            fdofs = append(append(v_F, Nv_F + e_F), append((Nv_F + Ne_F) + v_F, (Nv_F + Ne_F + Nv_F) + e_F))
            sdofs = append(append(v_S, Nv_S + e_S), append((Nv_S + Ne_S) + v_S, (Nv_S + Ne_S + Nv_S) + e_S))
        else:
            error("Only know how to map dofs for P1 and P2 elements.")

        # Extract map from vertices in Omega to vertices in Omega_F
        vertex_map_to_fluid = {}
        vertex_map_from_fluid = Omega_F.data().mesh_function("parent_vertex_indices")
        for i in range(vertex_map_from_fluid.size()):
            vertex_map_to_fluid[vertex_map_from_fluid[i]] = i

        # Extract map from vertices in Omega to vertices in Omega_S
        vertex_map_to_structure = {}
        vertex_map_from_structure = Omega_S.data().mesh_function("parent_vertex_indices")
        for i in range(vertex_map_from_structure.size()):
            vertex_map_to_structure[vertex_map_from_structure[i]] = i

        info("Computing FSI boundary and orientation markers")

        # Initialize FSI boundary and orientation markers on Omega
        Omega.init(D - 1, D)
        fsi_boundary = FacetFunction("uint", Omega, D - 1)
        fsi_boundary.set_all(0)
        fsi_orientation = Omega.data().create_mesh_function("facet_orientation", D - 1)
        fsi_orientation.set_all(0)

        # Initialize FSI boundary on Omega_F
        Omega_F.init(D - 1, D)
        Omega_F.init(0, 1)
        fsi_boundary_F = MeshFunction("uint", Omega_F, D - 1)
        fsi_boundary_F.set_all(0)

        # Initialize FSI boundary on Omega_S
        Omega_S.init(D - 1, D)
        Omega_S.init(0, 1)
        fsi_boundary_S = MeshFunction("uint", Omega_S, D - 1)
        fsi_boundary_S.set_all(0)

        # Compute FSI boundary and orientation markers on Omega
        for facet in facets(Omega):

            # Handle facets on the boundary
            cells = facet.entities(D)
            if len(cells) == 1:

                # Create cell and midpoint
                c = cells[0]
                cell = Cell(Omega, c)
                p = cell.midpoint()

                # Check whether point is inside structure domain
                facet_index = facet.index()
                if structure.inside(p0, True):

                    # On structure boundary
                    fsi_boundary[facet_index] = 1
                    fsi_orientation[facet_index] = c

                else:

                    # On fluid boundary
                    fsi_boundary[facet_index] = 0
                    fsi_orientation[facet_index] = c

                continue

            # Sanity check
            if len(cells) != 2:
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

                # On FSI boundary
                fsi_boundary[facet_index] = 2
                fsi_orientation[facet_index] = c1
                fsi_boundary_F[_map_to_facet(facet_index, Omega, Omega_F, vertex_map_to_fluid)] = 2
                fsi_boundary_S[_map_to_facet(facet_index, Omega, Omega_S, vertex_map_to_structure)] = 2
            elif p1_inside and not p0_inside:

                # On FSI boundary
                fsi_boundary[facet_index] = 2
                fsi_orientation[facet_index] = c0
                fsi_boundary_F[_map_to_facet(facet_index, Omega, Omega_F, vertex_map_to_fluid)] = 2
                fsi_boundary_S[_map_to_facet(facet_index, Omega, Omega_S, vertex_map_to_structure)] = 2
            elif p0_inside and p1_inside:

                # Inside structure domain
                fsi_boundary[facet_index] = 1
            else:

                # Inside fluid domain
                fsi_boundary[facet_index] = 0

        # Initialize global edge indices (used in read_primal_data)
        init_parent_edge_indices(Omega_F, Omega)
        init_parent_edge_indices(Omega_S, Omega)

        # Store data
        self.Omega_F = Omega_F
        self.Omega_S = Omega_S
        self.cell_domains = cell_domains
        self.fdofs = fdofs
        self.sdofs = sdofs
        self.fsi_boundary = fsi_boundary
        self.fsi_orientation = fsi_orientation
        self.fsi_boundary_F = fsi_boundary_F
        self.fsi_boundary_S = fsi_boundary_S

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

    #--- Functions

    def update(self, t0, t1, dt):
        pass

    def fluid_body_force(self):
        return []

    def structure_body_force(self):
        return []

    def structure_boundary_traction_extra(self):
        return Constant((0, 0))

    def mesh_right_hand_side(self):
        return Constant((0, 0))

def _map_to_facet(facet_index, Omega, Omega_X, vertex_map):
    "Map facet index in Omega to facet index in Omega_X"

    # Get the two vertices in Omega
    facet = Facet(Omega, facet_index)
    v0 = facet.entities(0)[0]
    v1 = facet.entities(0)[1]

    # Get the two vertices in Omega_X
    v0 = Vertex(Omega_X, vertex_map[v0])
    v1 = Vertex(Omega_X, vertex_map[v1])

    # Get the facets of the two vertices in Omega_X
    f0 = v0.entities(1)
    f1 = v1.entities(1)

    # Get the common facet index
    common_facets = set(f0).intersection(set(f1))

    # Check that we get exactly one facet
    if not len(common_facets) == 1:
        error("Unable to find facet in fluid mesh.")

    return int(list(common_facets)[0])
