from fsirun import *

# Set common parameters
p = default_parameters()
p["tolerance"] = 5e-8
p["initial_timestep"] = 0.025

# Run problem
run_local("channel_with_flap", parameters=p, case=0)
