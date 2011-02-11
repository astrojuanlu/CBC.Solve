from fsirun import *

# Set parameters
p = default_parameters()
p["solve_primal"] = True
p["solve_dual"] = True
p["estimate_error"] = True
p["uniform_timestep"] = True
p["tolerance"] = 1e-6
p["initial_timestep"] = 0.05

# test case 1
p["refinement_algorithm"] = "regular_cut"
p["description"] = "adaptive, k = 0.05 (regular)"
p["output_directory"] = "results-channel_with_flap_1"

# test case 2
#p["refinement_algorithm"] = "recursive_bisection"
#p["description"] = "adaptive, k = 0.05 (bisection)"
#p["output_directory"] = "results-channel_with_flap_2"

# test case 3
#p["dorfler_marking"] = False
#p["refinement_algorithm"] = "regular_cut"
#p["description"] = "adaptive, k = 0.05 (regular, fixed fraction)"
#p["output_directory"] = "results-channel_with_flap_3"

# Run problem
run_local("channel_with_flap", p)
