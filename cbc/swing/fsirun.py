"""This module provides utilities for running problems with various
parameters, either on a local machine or on BigBlue."""

__author__ = "Kristoffer Selim and Anders Logg"
__copyright__ = "Copyright (C) 2010 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

import os
from dolfin_utils.pjobs import *
from dolfin_utils.commands import *
from utils import date
from parameters import *

def run_local(problem, parameters, case=None):
    "Run problem on local machine with given parameters."

    # Set output directory
    set_output_directory(parameters, problem, case)

    # Store parameters to file
    filename = store_parameters(parameters)

    # Set name of logfile
    if case is None:
        logfile = "output-%s-%s.log" % (problem, date())
    else:
        logfile = "output-%s-%s.log" % (problem, str(case))
    logfile = os.path.join(parameters["output_directory"], logfile)

    # Submit job
    status, output = getstatusoutput("python %s.py %s | tee %s" % (problem,
                                                                   filename,
                                                                   logfile))

    return status, output

def run_bb(problem, parameters, case=None):
    "Run problem on bigblue with given parameters."

    # Store parameters to file
    filename = store_parameters(parameters)

    # Create name
    if case is None:
        name = problem
    else:
        name = problem + "_" + str(case)

    # Submit job
    submit("python %s.py %s" % (problem, filename),
           nodes=1, ppn=8, keep_environment=True, walltime=24*1000, name=name)
