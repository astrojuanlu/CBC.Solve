__author__ = "Anders Logg"
__copyright__ = "Copyright (C) 2009 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

# Last changed: 2010-06-30

from time import time
from dolfin import info, error, Progress

class CBCSolver:
    "Base class for all solvers"

    def __init__(self):
        "Constructor"
        self._time_step = 1
        self._progress = None
        self._cpu_time = time()

    #--- Functions that must be overloaded by subclasses ---

    def solve():
        error("solve() function not implemented by solver.")

    def __str__():
        error("__str__ not implemented by solver.")

    #--- Useful functions for solvers ---

    def _end_time_step(self, t, T):
        "Call at end of time step"

        # Record CPU time
        cpu_time = time()
        elapsed_time = cpu_time - self._cpu_time
        self._cpu_time = cpu_time

        # Write some useful information
        s = "Time step %d (t = %g) finished in %g seconds." % (self._time_step, t, elapsed_time)
        info("\n" + s + "\n" + len(s)*"-" + "\n")

        # Update progress bar
        if self._progress is None:
            self._progress = Progress("Time-stepping")
        self._progress.update(t / T)

        # Increase time step counter
        self._time_step += 1

        ## Store solution
        #if self.parameters["save_solution"]:
        #    self.velocity_series.store(self.u1.vector(), t)
        #    self.pressure_series.store(self.p1.vector(), t)
