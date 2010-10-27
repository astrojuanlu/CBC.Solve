from dolfin import error, info

class Heart:

    def __init__(self, cell_model):

        self._cell_model = cell_model

    # Mandatory stuff

    def mesh(self):
        error("Need to prescribe domain")

    def conductivities(self):
        error("Need to prescribe conducitivites")

    # Optional stuff

    def applied_current(self):
        return None

    def end_time(self):
        info("Using default end time (T = 1.0)")
        return 1.0

    def essential_boundaries(self):
        return None

    def essential_boundary_values(self):
        return None

    def initial_conditions(self):
        return None

    def neumann_boundaries(self):
        return None

    def boundary_current(self):
        return None

    # Peculiar stuff (for now)
    def is_dynamic(self):
        return True

    # Helper functions
    def cell_model(self):
        return self._cell_model
