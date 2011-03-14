"Simple extrapolation of converging sequence"

__author__ = "Anders Logg"
__copyright__ = "Copyright (C) 2010 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"
__date__ = "2011-03-14"

from numpy import array, polyfit, log, sum, sqrt, exp
import pylab

def extrapolate(n, y, tolerance=1e-15, plot=False, call_show=True):
    "Extrapolate functional value Y from sequence of values (n, y)."

    # Make sure we have NumPy arrays
    n = array(n)
    y = array(y)

    # Create initial "bound"
    Y0 = 0.99*y[-1]
    Y1 = 1.01*y[-1]

    # Compute initial interior points
    phi = (sqrt(5.0) + 1.0) / 2.0
    Y2 = Y1 - (Y1 - Y0) / phi
    Y3 = Y0 + (Y1 - Y0) / phi

    # Compute initial values
    F0, e, nn, ee, yy = _eval(n, y, Y0)
    F1, e, nn, ee, yy = _eval(n, y, Y1)
    F2, e, nn, ee, yy = _eval(n, y, Y2)
    F3, e, nn, ee, yy = _eval(n, y, Y3)

    # Solve using direct search (golden ratio fraction)
    while Y1 - Y0 > tolerance:

        if F2 < F3:
            Y1, F1 = Y3, F3
            Y3, F3 = Y2, F2
            Y2 = Y1 - (Y1 - Y0) / phi
            F2, e, nn, ee, yy = _eval(n, y, Y2)
        else:
            Y0, F0 = Y2, F2
            Y2, F2 = Y3, F3
            Y3 = Y0 + (Y1 - Y0) / phi
            F3, e, nn, ee, yy = _eval(n, y, Y3)

        print Y0, Y1

    # Compute reference value
    Y = 0.5*(Y0 + Y1)

    # Print results
    print
    print "Reference value:", Y

    # Plot result
    if plot:
        pylab.figure()
        pylab.subplot(2, 1, 1)
        pylab.title("Reference value: %g" % Y)
        pylab.semilogx(n, y, 'b-o')
        pylab.semilogx(nn, yy, 'g--')
        pylab.subplot(2, 1, 2)
        pylab.loglog(n, e, 'b-o')
        pylab.loglog(nn, ee, 'g--')
        pylab.grid(True)
        if call_show:
            pylab.show()

    return Y

def _eval(n, y, Y):
    """Evaluate how well we can fit a straight line to the logarithmic
    plot of the error for given reference value Y."""

    # Compute error
    e = abs(Y - y)

    # Polyfit straight line
    k, m = polyfit(log(n), log(e), 1)
    C = exp(m)
    alpha = m

    # Compute error
    nn = array(list(n) + [2*n[-1], 4*n[-1], 8*n[-1], 16*n[-1]])
    ee = C*(nn**alpha)

    # Compute extrapolation sequence
    if y[0] > y[-1]:
        yy = Y + ee
    else:
        yy = Y - ee

    # Compute quality of fit
    E = sqrt(sum((log(ee[:len(e)]) - log(e))**2))

    #pylab.subplot(2, 1, 1)
    #pylab.title("Reference value: %g" % Y)
    #pylab.semilogx(n, y, 'b-o')
    #pylab.semilogx(nn, yy, 'g--')
    #pylab.subplot(2, 1, 2)
    #pylab.loglog(n, e, 'b-o')
    #pylab.loglog(nn, ee, 'g--')

    return E, e, nn, ee, yy

if __name__ == "__main__":

    n = [20570, 30795, 46236, 69177, 104650, 157801, 237263, 358402]
    y = [0.02038030, 0.02046210, 0.02028142, 0.02030110, 0.02019661, 0.02018782, 0.02013778, 0.02010004]
    extrapolate(n, y, plot=True)

    #n = [2028, 4104, 8381, 17392, 35274, 69944, 141297]
    #y = [0.48258016, 0.48458517, 0.48552337, 0.48566151, 0.48563823, 0.48561884, 0.48561005]
    #extrapolate(n, y, plot=True)


