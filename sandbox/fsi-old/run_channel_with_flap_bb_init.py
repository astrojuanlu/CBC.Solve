# Script to run before the run_channel_with_flap_bb.py script to get
# all forms into the Instant cache.

from fsirun import *

p = default_parameters()
p["solve_primal"] = True
p["solve_dual"] = True
p["estimate_error"] = True
p["uniform_timestep"] = True
p["tolerance"] = 1e-8
p["fixedpoint_tolerance"] = 1e-8
p["max_num_refinements"] = 0
p["initial_timestep"] = 0.05

p["structure_element_degree"] = 1
p["output_directory"] = "unspecified"
run_local("channel_with_flap", parameters=p, case="init_1")

p["structure_element_degree"] = 2
p["output_directory"] = "unspecified"
run_local("channel_with_flap", parameters=p, case="init_2")
