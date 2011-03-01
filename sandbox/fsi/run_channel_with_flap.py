from fsirun import *

# Set common parameters
p = default_parameters()
p["tolerance"] = 1e-6
p["initial_timestep"] = 0.05

# Run problem
run_local("channel_with_flap", parameters=p, case=0)
