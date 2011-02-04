from dolfin import *

application_parameters = Parameters("application_parameters")

#application_parameters.add("end_time", 0.25)
#application_parameters.add("dt", 0.01)
#application_parameters.add("mesh_scale", 1)
#application_parameters.add("TOL", 1e-12)
#application_parameters.add("w_h", 0.45)
#application_parameters.add("w_k", 0.45)
#application_parameters.add("w_c", 0.1)

file = File("application_parameters.xml")
file >> application_parameters

info(application_parameters, True)
