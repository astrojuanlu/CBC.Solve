from dolfin import *
from kinematics import *
from material_models import LinearElastic, StVenantKirchhoff, MooneyRivlin, neoHookean, Isihara, Biderman, GentThomas
from problem_definitions import StaticHyperelasticity, Hyperelasticity

# Optimise compilation of forms
parameters.optimize = True
