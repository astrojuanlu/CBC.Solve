from dolfin import *

def plot_interface(n, length, lamda, delta_A):
    import pylab

    xs = pylab.linspace(0, length, n)
    ys = [- delta_A*pylab.sin(2*pi/lamda*x) for x in xs]

    pylab.plot(xs, ys)
    pylab.show()

class Density(Expression):

    def __init__(self, amplitude, wavelength, rho0, rho1):
        self.amplitude = amplitude
        self.wavelength = wavelength
        self.rho0 = rho0
        self.rho1 = rho1

    def eval(self, values, x):
        k = 2*pi/self.wavelength
        interface = - self.amplitude*sin(k*x[0])
        if x[1] >= interface:
            values[0] = self.rho1
        else:
            values[0] = self.rho0

def snap_mesh_to_interface(mesh, amplitude, wavelength):

    # Extract mesh coordinates
    x = mesh.coordinates()

    k = 2*pi/wavelength

    # Update mesh coordinates based on v
    num_vertices = mesh.num_vertices()

    for i in range(num_vertices):

        # Distance to interface:
        y = amplitude*sin(k*x[i][0])

        distance = x[i][1] - y

        if x[i][1] <= 0.0:
            f = pow((x[i][1]/(-h1) - 1), 2)
            x[i][1] += f*distance
        else:
            f = pow((x[i][1]/(h2) - 1), 2)
            x[i][1] += f*distance


if __name__ == "__main__":

    length = 1

    # Height from bottom to interface at x[0] = 0
    h1 = 0.4

    # Height from interface to top at x[0] = 0
    h2 = 0.6

    # Wavelength
    wavelength = 2.0/3

    # Amplitude
    amplitude = 0.1

    # Density at bottom
    rho0 = 1.0

    # Density at top:
    rho1 = 2.0

    # Number of cells (in each direction)
    n = 50

    mesh = Rectangle(0, -h1, length, h2, n, n)

    snap_mesh_to_interface(mesh, amplitude, wavelength)

    plot(mesh, title="snapped", interactive=True)

    rho = Density(amplitude, wavelength, rho0, rho1)
    plot(rho, mesh=mesh, interactive=True)


