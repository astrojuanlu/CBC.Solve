"""This module provides utilities for running problems with various
parameters, either on a local machine or on BigBlue."""

__author__ = "Kristoffer Selim and Anders Logg"
__copyright__ = "Copyright (C) 2010 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

import os
from dolfin_utils.pjobs import *
from parameters import *

def run_local(problem, parameters):
    "Run problem on local machine with given parameters."

    # Store parameters to file
    store_parameters(parameters)

    # Submit job
    os.system("python %s.py" % problem)

def run_bb(problem, parameters):
    "Run problem on bigblue with given parameters."

    # Store parameters to file
    store_parameters(parameters)

    # Submit job
    submit("python %s.py" % problem, nodes=1, ppn=8,  keep_environment=True, walltime=24*1000)
