"""
This Module contains default parameters for the fsi newton solver
    - reuse_jacobian
        Prevents the jacobian from being reassembled at every newton iteration.
    - max_reuse_jacobian
        Forces reassembly of jacobian after every nth iteration.
            Note: At the start of a time step a reassembly happens anyway.
    - jacobian_forms
        "auto"   - Use the dolfin Derivative() to the symbolic derivative of the residual forms.
        "manual" - Use a combined Jacobian from jacobianforms_buffered.py and jacobianforms_step.py
        "buff"   - Buffer the jacobian from jacobianforms_buffered.py and add the jacobian from
                   jacobianforms_step.py during time steping.
    - bigblue
        If running on bigblue plots should not be generated with matplotlib, instead
        data should be pickled.
    - linear_solve
        np (better condition numbers) or PETSc (better memoory management)
    - Stress coupling
        'forward' structure gets stress from fluid
        'backward' fluid gets stress from structure
    -newtonsoltol
        newtonsolver tolerance
"""
__author__ = "Gabriel Balaban"
__copyright__ = "Copyright (C) 2010 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

from dolfin import parameters
# Set DOLFIN parameters
parameters["form_compiler"]["cpp_optimize"] = True
#parameters["form_compiler"]["log_level"] = DEBUG
parameters["form_compiler"]["representation"] = "quadrature"
##parameters["form_compiler"]["optimize"] = True

solver_params = { "V_F":{"deg":2,"elem":"CG"},
                  "Q_F":{"deg":1,"elem":"CG"},
                  "L_F":{"deg":0,"elem":"DG"},
                  "V_S":{"deg":1,"elem":"CG"},
                  "Q_S":{"deg":1,"elem":"CG"},
                  "V_M":{"deg":1,"elem":"CG"},
                  "L_M":{"deg":0,"elem":"DG"},
                  "stress_coupling":"forward",
                  "solve":True,
                  "store":False,
                  "plot":False,
                  "linear_solve":"PETSc",
                  "runtimedata":{"fsisolver":False,
                                 "newtonsolver":False},
                  "jacobian":"auto",
                  "reuse_jacobian":True,
                  "max_reuse_jacobian":20,
                  "newtonitrmax":100,
                  "newtonsoltol":1.0e-13,
                  "bigblue":False
                  }
