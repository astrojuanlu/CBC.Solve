from fsirun import *

# Set common parameters
p = default_parameters()
p["solve_primal"] = True
p["solve_dual"] = False
p["estimate_error"] = False
p["uniform_timestep"] = True
p["tolerance"] = 1e-6
p["initial_timestep"] = 0.05
p["num_initial_refinements"] = 0
#p["plot_solution"] = True

p["structure_element_degree"] = 2

#problem = "channel_with_flap"
problem = "modified_pressure_driven_cavity"

# Run problem
run_local(problem, parameters=p, case="test")

# Increment = 3.29897e-10 (tolerance = 1e-08), converged after 5 iterations
# Value of goal functional at t = 0.5: 0.0162199
# Value of goal functional at T: 0.0162199
# Primal solution computed in 19.8312 seconds.

# P1 structure: 0.01621987919642476
# P2 structure: 0.005281059097190604
