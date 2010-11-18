from dolfin import *

def plot_interface(n, length, lamda, delta_A):
    import pylab

    xs = pylab.linspace(0, length, n)
    ys = [- delta_A*pylab.sin(2*pi/lamda*x) for x in xs]

    pylab.plot(xs, ys)
    pylab.show()

class Density(Expression):

    def __init__(self, amplitude, wavelength, nu0, nu1):
        self.amplitude = amplitude
        self.wavelength = wavelength
        self.nu0 = nu0
        self.nu1 = nu1

    def eval(self, values, x):
        interface = - self.amplitude*sin(2*pi/self.wavelength*x[0])
        if x[1] > interface:
            values[0] = self.nu1
        else:
            values[0] = self.nu0

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
    nu0 = 1.0

    # Density at top:
    nu1 = 2.0

    # Number of mesh points
    n = 100

    mesh = Rectangle(0, -h1, length, h2, n, n)

    nu = Density(amplitude, wavelength, nu0, nu1)

    plot(nu, mesh=mesh, interactive=True)


