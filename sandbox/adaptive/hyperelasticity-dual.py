from dolfin import *

dt = 0.01
T = 2.0
t = 0

displacement_series = TimeSeries("displacement")
velocity_series = TimeSeries("velocity")

mesh = UnitCube(8, 8, 8)
vector = VectorFunctionSpace(mesh, "CG", 1)

u_h = Function(vector)
v_h = Function(vector)

while t < T:
    t = t + dt

    displacement_series.retrieve(u_h.vector(), t)
    velocity_series.retrieve(v_h.vector(), t)

    plot(u_h, mode ='displacement')
#    plot(v_h)

interactive()
