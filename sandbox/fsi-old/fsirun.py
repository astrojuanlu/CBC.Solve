"""This module provides utilities for running problems with various
parameters, either on a local machine or on BigBlue."""

__author__ = "Kristoffer Selim and Anders Logg"
__copyright__ = "Copyright (C) 2010 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

import os
from dolfin_utils.pjobs import *
from parameters import *

def run_local(problem, parameters, case=None):
    "Run problem on local machine with given parameters."

    # Store parameters to file
    filename = store_parameters(parameters, problem, case)

    # Submit job
    os.system("python %s.py %s" % (problem, filename))

def run_bb(problem, parameters, case=None):
    "Run problem on bigblue with given parameters."

    # Store parameters to file
    filename = store_parameters(parameters, problem, case)

    # Create name
    if case is None:
        name = problem
    else:
        name = problem + "_" + str(case)

    # Submit job
    submit("python %s.py %s" % (problem, filename),
           nodes=1, ppn=8, keep_environment=True, walltime=24*1000, name=name)
