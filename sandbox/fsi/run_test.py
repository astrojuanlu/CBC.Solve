from fsirun import *

# Set common parameters
p = default_parameters()
p["solve_primal"] = True
p["solve_dual"] = True
p["estimate_error"] = True
p["uniform_timestep"] = True
p["tolerance"] = 1e-8
p["initial_timestep"] = 0.05
p["num_initial_refinements"] = 0
p["structure_element_degree"] = 1

p["refinement_algorithm"] = "regular_cut"
p["dorfler_marking"] = False
p["marking_fraction"] = 0.3

#p["plot_solution"] = True

problem = "channel_with_flap"

# Run problem
run_local(problem, parameters=p, case="test")
