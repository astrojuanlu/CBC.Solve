# Initialize mesh conectivity
mesh.init(D-1, D)


# Create facet marker for the entire mesh
interior_facet_domains = MeshFunction("uint", Omega, D-1)
interior_facet_domains.set_all(0)

# Create facet marker for inflow (used in the Neumann BCs for the dual fluid)
left = compile_subdomains("x[0] == 0.0")
exterior_boundary = MeshFunction("uint", Omega, D-1)
left.mark(exterior_boundary, 2)

# Create facet marker for outflow (used in the Neumann BCs for the dual fluid)
right = compile_subdomains("x[0] == channel_length")
exterior_boundary = MeshFunction("uint", Omega, D-1)
right.mark(exterior_boundary, 3)

# Create facet orientation for the entire mesh
facet_orientation = mesh.data().create_mesh_function("facet orientation", D - 1)
facet_orientation.set_all(0)

