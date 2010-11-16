from dolfin import *
from pylab import array

def strain(v):
    return sym(grad(v))

def spin(v):
    return 0.5*(grad(v) - grad(v).T)

def move(mesh, velocity, timestep):
    """
    Move mesh according to timestep*velocity.

    Note that mesh is changed.
    """

    # Extract mesh coordinates
    x = mesh.coordinates()

    # Interpolate velocity onto P1
    P1 = VectorFunctionSpace(mesh, "CG", 1)
    v = interpolate(velocity, P1)
    dofs = v.vector().array()

    # Update mesh coordinates based on v
    num_vertices = mesh.num_vertices()
    for i in range(num_vertices):
        for k in range(mesh.geometry().dim()):
            x[i][k] += timestep*dofs[k*num_vertices + i]


# -----------------------------------------------------------------------------

def power_law_viscosity(v, p, T, n, E, V, R):
    """

    E:   activation energy (given)
    V:   activation volume (given)
    R:   gas constant      (given)
    n:   material dependent(given)

    v:   velocity
    p:   pressure
    T:   temperature
    """


    # Strain (rate)
    D = strain(v)

    # Second invariant of strain rate:
    J2 = second_invariant(D)

    # Constant
    F = 2**((1.0 - 2.0*n)/n)

    # Power-law viscocity
    eta2 = J2**((n - 1.0)/n)*F**(-1/n)*exp((E + V*p)/(n*R*T))

    return eta2

def viscosity(v):
    power_law_viscosity(v)

def Dt(tau, tau0, timestep):

    return (tau - tau0)/timestep


def JaumannDerivative(tau, W0, tau0):
    """

    triangle tau = d tau/ dt + tau'

    where

    tau' = grad tau * v + tau*W - W*tau

    where

    W = 0.5*(grad(v) - grad(v).T)

    and for now we skip grad tau part
    """
    return Dt(tau, tau0) + tau0*W0 - W0*tau0

def second_invariant(A):
    return 0.5*(tr(A)**2 - tr(A*A))

