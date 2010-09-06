"This module implements functionality for adaptivity."

__author__ = "Kristoffer Selim and Anders Logg"
__copyright__ = "Copyright (C) 2010 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

# Last changed: 2010-09-06

from dolfin import info

def estimate_error():
    "Estimate error and compute error indicators"

    # Compute error indicators
    indicators = []

    # Compute error estimate
    error = 1.0

    return error, indicators

def compute_timestep(R, S, TOL, dt, t, T):
    """Compute new time step based on residual R, stability factor S,
    tolerance TOL, and the previous time step dt. The time step is
    adjusted so that we will not step beyond the given end time."""

    # Parameters for adaptive time-stepping
    C = 1.0               # interpolation constant
    safety_factor = 0.9   # safety factor for time step selection
    snap = 0.9            # snapping to end time when close
    conservation = 1.0    # time step conservation (high value means small change)

    # Compute new time step
    dt_new = safety_factor * TOL / (C*S*R)

    # FIXME: Temporary until we get real input
    dt_new  = dt

    # Modify time step to avoid oscillations
    dt_new = (1.0 + conservation) * dt * dt_new / (dt + conservation * dt_new)

    # Modify time step so we don't step beoynd end time
    at_end = False
    if dt_new > snap * (T - t):
        print "Close to t = T, snapping time step to end time: %g --> %g" % (dt_new, T - t)
        dt_new = T - t
        at_end = True

    info("Changing time step: %g --> %g" % (dt, dt_new))

    return dt_new, at_end

def refine_mesh(mesh, indicators):
    return mesh
