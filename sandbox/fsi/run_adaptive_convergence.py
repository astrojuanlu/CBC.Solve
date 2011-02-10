from fsirun import *

# Set parameters
p = default_parameters()
p["solve_primal"] = True
p["solve_dual"] = True
p["estimate_error"] = True
p["uniform_timestep"] = True
p["tolerance"] = 1e-6
p["initial_timestep"] = 0.025
#p["dorfler_marking"] = False
#p["marking_fraction"] = 0.5
#p["refinement_algorithm"] = "recursive_bisection"
p["refinement_algorithm"] = "regular_cut"
p["description"] = "adaptive, k = 0.025 (regular)"

# Run problem
run_local("modified_pressure_driven_cavity", p)
