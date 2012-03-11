__author__ = "Anders Logg"
__copyright__ = "Copyright (C) 2012 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

# First added:  2012-03-04
# Last changed: 2012-03-09

from dolfin import *
from time import sleep

cpp_u_F = """
class u_F : public Expression
{
public:

  u_F() : Expression(2), C(0), t(0) {}

  void eval(Array<double>& values, const Array<double>& xx,
            const ufc::cell& cell) const
  {
    const double x = xx[0];

    values[0] = 0.0;
    values[1] = 2.0*pi*C*x*(1.0 - x)*sin(pi*t)*cos(pi*t);
  }

  double C;
  double t;

};
"""

cpp_p_F = """
class p_F : public Expression
{
public:

  p_F() : Expression(), C(0), t(0) {}

  void eval(Array<double>& values, const Array<double>& xx,
            const ufc::cell& cell) const
  {
    const double x = xx[0];
    const double Y = xx[1];

    const double a = sin(pi*t);
    const double b = cos(pi*t);

    values[0] = -2.0*pow(C, 2)*pow(1.0 - 2.0*x, 2)*pow(a, 3)*(a + pi*b);
  }

  double C;
  double t;

};
"""

cpp_U_S = """
class U_S : public Expression
{
public:

  U_S() : Expression(2), C(0), t(0) {}

  void eval(Array<double>& values, const Array<double>& xx,
            const ufc::cell& cell) const
  {
    const double X = xx[0];
    const double Y = xx[1];

    values[0] = 0.0;
    values[1] = C*X*(1 - X)*sin(pi*Y)*pow(sin(pi*t), 2);
  }

  double C;
  double t;

};
"""

cpp_U_M = """
class U_M : public Expression
{
public:

  U_M() : Expression(2), C(0), t(0) {}

  void eval(Array<double>& values, const Array<double>& xx,
            const ufc::cell& cell) const
  {
    const double X = xx[0];
    const double Y = xx[1];

    values[0] = 0.0;
    values[1] = C*X*(1 - X)*sin(pi*Y)*pow(sin(pi*t), 2);
  }

  double C;
  double t;

};
"""

cpp_f_F = """
class f_F : public Expression
{
public:

  f_F() : Expression(2), C(0), t(0) {}

  void eval(Array<double>& values, const Array<double>& xx,
            const ufc::cell& cell) const
  {
    const double x = xx[0];

    const double A = 1.0;
    const double B = 2.0;
    const double D = 4.0;
    const double E = 8.0;
    const double a = sin(pi*t);
    const double b = cos(pi*t);
    const double fx = E*pow(C, 2)*(A - B*x)*pow(a, 3)*(a + pi*b);
    const double fy = B*pow(pi, 2)*C*x*(A - x)*(pow(b, 2) - pow(a, 2)) + D*pi*C*a*b;

    values[0] = fx;
    values[1] = fy;
  }

  double C;
  double t;

};
"""

cpp_F_S = """
class F_S : public Expression
{
public:

  F_S() : Expression(2), C(0), t(0) {}

  void eval(Array<double>& values, const Array<double>& xx,
            const ufc::cell& cell) const
  {
    const double X = xx[0];
    const double Y = xx[1];

    const double rho_S = 100.0;

    const double A = 1.0;
    const double B = 2.0;
    const double D = 3.0;
    const double E = 6.0;
    const double F = 8.0;
    const double G = 16.0;
    const double H = rho_S;
    const double a = sin(pi*t);
    const double b = cos(pi*t);
    const double c = sin(pi*Y);
    const double d = cos(pi*Y);
    const double e = pow(a, 2);
    const double f = pow(pi, 2);
    const double g = pow(X, 2);
    const double h = pow(c, 2);
    const double i = pow(d, 2);
    const double j = pow(b, 2);
    const double k = pow(a, 4);
    const double l = pow(C, 2);
    const double m = pow(X, 3);
    const double fx = C*e*(D*pi*d*(B*X - A) + C*e*(h*(B*f*X*g - D*f*g \
                    - (G - f)*X + F) - D*f*X*i*(B*g - D*X + A)));
    const double fy = B*C*e*c - C*pi*e*d - l*pi*k*d*c + B*pi*C*X*e*d \
                    - 2.0*H*C*f*(g - X)*(j - e)*c - E*l*pi*(g - X)*k*d*c \
                    + l*f*(D*g - B*m - X)*k*(i - h);

    values[0] = fx;
    values[1] = fy;
  }

  double C;
  double t;

};
"""

cpp_F_M = """
class F_M : public Expression
{
public:

  F_M() : Expression(2), C(0), t(0) {}

  void eval(Array<double>& values, const Array<double>& xx,
            const ufc::cell& cell) const
  {
    const double X = xx[0];
    const double Y = xx[1];

    const double A = 1.0;
    const double B = 2.0;
    const double D = 3.0;
    const double a = sin(pi*t);
    const double b = cos(pi*t);
    const double c = sin(pi*Y);
    const double d = cos(pi*Y);
    const double fx = D*C*pi*d*pow(a, 2)*(B*X - A);
    const double fy = C*a*(B*c*a + pi*d*a*(B*X - A) - B*X*pi*c*b*(X - A));

    values[0] = fx;
    values[1] = fy;
  }

  double C;
  double t;

};
"""

cpp_g_0 = """
class g_0 : public Expression
{
public:

  g_0() : Expression(2), C(0), t(0) {}

  void eval(Array<double>& values, const Array<double>& xx,
            const ufc::cell& cell) const
  {
    const double x = xx[0];

    const double A = 1.0;
    const double B = 2.0;
    const double a = sin(pi*t);
    const double b = cos(pi*t);
    const double p = -B*pow(C, 2)*pow(A - B*x, 2)*pow(a, 3)*(a + pi*b);
    const double gx = C*(A - B*x)*a*((A - p)*a - B*pi*b)
                      / sqrt(A + pow(C, 2)*pow(A - B*x, 2)*pow(a, 4));
    const double gy = 0.0;

    values[0] = gx;
    values[1] = gy;
  }

  double C;
  double t;

};
"""

cpp_G_0 = """
class G_0 : public Expression
{
public:

  G_0() : Expression(2), C(0), t(0) {}

  void eval(Array<double>& values, const Array<double>& xx,
            const ufc::cell& cell) const
  {
    const double X = xx[0];

    const double A = 1.0;
    const double B = 2.0;
    const double a = sin(pi*t);
    const double b = cos(pi*t);
    const double p = -B*pow(C, 2)*pow(A - B*X, 2)*pow(a, 3)*(a + pi*b);
    const double Gx = C*(A - B*X)*a*((A - p)*a - B*pi*b)
                        / (A + pow(C, 2)*pow(A - B*X, 2)*pow(a, 4));
    const double Gy = 0.0;

    values[0] = Gx;
    values[1] = Gy;
  }

  double C;
  double t;

};
"""

if __name__ == "__main__":

    # Instantiate expressions
    C = 0.2
    u_F = Expression(cpp_u_F)
    p_F = Expression(cpp_p_F)
    U_S = Expression(cpp_U_S)
    U_M = Expression(cpp_U_M)
    f_F = Expression(cpp_f_F)
    F_S = Expression(cpp_F_S)
    F_M = Expression(cpp_F_M)
    g_0 = Expression(cpp_g_0)
    G_0 = Expression(cpp_G_0)
    u_F.C = C
    p_F.C = C
    U_S.C = C
    U_M.C = C
    f_F.C = C
    F_S.C = C
    F_M.C = C
    g_0.C = C
    G_0.C = C

    # Functions used for plotting expressions
    n = 16
    omega_F = Rectangle(0.0, 0.5, 1.0, 1.0, n, n)
    Omega_M = Rectangle(0.0, 0.5, 1.0, 1.0, n, n)
    Omega_S = Rectangle(0.0, 0.0, 1.0, 0.5, n, n)
    V_F = VectorFunctionSpace(omega_F, "Lagrange", 1)
    Q_F = FunctionSpace(omega_F, "Lagrange", 1)
    V_S = VectorFunctionSpace(Omega_S, "Lagrange", 1)
    V_M = VectorFunctionSpace(Omega_M, "Lagrange", 1)
    _u_F = Function(V_F)
    _p_F = Function(Q_F)
    _U_S = Function(V_S)
    _U_M = Function(V_M)
    _f_F = Function(V_F)
    _F_S = Function(V_S)
    _F_M = Function(V_S)
    _g_0 = Function(V_F)
    _G_0 = Function(V_M)

    # Animate solutions
    T = 0.5
    t = 0.0
    dt = 0.01

    while t < T:

        print t

        u_F.t = t
        p_F.t = t
        U_S.t = t
        U_M.t = t
        f_F.t = t
        F_S.t = t
        F_M.t = t
        g_0.t = t
        G_0.t = t

        _u_F.interpolate(u_F)
        _p_F.interpolate(p_F)
        _U_S.interpolate(U_S)
        _U_M.interpolate(U_M)
        _f_F.interpolate(f_F)
        _F_S.interpolate(F_S)
        _F_M.interpolate(F_M)
        _g_0.interpolate(g_0)
        _G_0.interpolate(G_0)

        plot(_u_F, title="u_F", autoposition=False)
        plot(_p_F, title="p_F", autoposition=False)
        plot(_U_S, title="U_S", autoposition=False, mode="displacement")
        plot(_U_M, title="U_M", autoposition=False, mode="displacement")
        plot(_f_F, title="f_F", autoposition=False)
        plot(_F_S, title="F_S", autoposition=False)
        plot(_F_M, title="F_M", autoposition=False)
        plot(_g_0, title="g_0", autoposition=False)
        plot(_G_0, title="G_0", autoposition=False)

        t += dt

        sleep(0.1)

    interactive()
