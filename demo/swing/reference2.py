from sympy import *

print("-"*72)
print("Structure problem in the reference configuration (S)")
print("-"*72)

# Declare useful symbols
X, Y, x, y, t = symbols('X, Y, x, y, t')
A, rho_0_S, rho_f, mu, lam, eta = symbols('A rho_0_S rho_f mu lambda eta')

# Define a suitable solution field for the solid displacement
U_S = Matrix([0,
              A*X*(1 - X)*Y*sin(t)])
print("U_S =\n%s" % U_S)

# The kinematics of the motion are then described as follows
I = eye(2)
Grad_U_S = Matrix([[simplify(diff(U_S[0], X)), simplify(diff(U_S[0], Y))],
                   [simplify(diff(U_S[1], X)), simplify(diff(U_S[1], Y))]])
F_S = I + Grad_U_S
J_S = F_S.det()
C = F_S.T*F_S
E = Rational(1, 2)*(C - I)

# Compute the terms in (S) defined by U_S
Sigma_S = F_S*(2*mu*E + lam*E.trace()*I)
Div_Sigma_S = Matrix([simplify(diff(Sigma_S[0], X) + diff(Sigma_S[1], Y)),
                     simplify(diff(Sigma_S[2], X) + diff(Sigma_S[3], Y))])
d2U_S_dt2 = Matrix([simplify(diff(U_S[0], t, 2)), simplify(diff(U_S[1], t, 2))])

# Use (S) to determine B_S
B_S = rho_0_S*d2U_S_dt2 - Div_Sigma_S
print("B_S =\n%s" % B_S)

# Check if (S) is satisfied
check_S = rho_0_S*d2U_S_dt2 - Div_Sigma_S - B_S
print("rho_0_S*d2U_S_dt2 - Div_Sigma_S - B_S =\n%s" %
      Matrix([simplify(check_S[0]), simplify(check_S[1])]))

print("-"*72)
print("Mesh problem in the reference configuration (M)")
print("-"*72)

# Define a suitable solution field for the mesh displacement
U_M = Matrix([0,
              2*A*(1 - Y)*X*(1 - X)*sin(t)])
print("U_M =\n%s" % U_M)

# Check whether U_S = U_M on the interface
check_M = U_S.subs(Y, Rational(1, 2)) - U_M.subs(Y, Rational(1, 2))
print("U_S(0, 1/2) - U_M(0, 1/2) =\n%s" % check_M)

print("-"*72)
print("Fluid problem in the current configuration (f)")
print("-"*72)

# Define the solution field for the fluid velocity based on the time
# derivative of the solid velocity on the interface to enforce the
# no-slip condition
V_F = Matrix([diff(U_S[0].subs(Y, Rational(1, 2)), t),
              diff(U_S[1].subs(Y, Rational(1, 2)), t)])

# Check whether the reference fluid velocity matches the solid
# velocity at the interface
check_V_F = Matrix([
        diff(U_S[0].subs(Y, Rational(1, 2)), t) - V_F[0].subs(Y, Rational(1, 2)),
        diff(U_S[0].subs(Y, Rational(1, 2)), t) - V_F[0].subs(Y, Rational(1, 2))])
print("V_S(0, 1/2) - V_F(0, 1/2) =\n%s" % check_V_F)

# Cast the reference fluid coordinates in terms of the current fluid
# coordinates
X_x = solve(x - (X + U_M[0]), X)[0]
Y_y = solve(y - (Y + U_M[1]), Y)[0]

# Define the current fluid velocity
v_f = Matrix([V_F[0].subs({Y:Y_y}).subs({X:X_x}),
             V_F[1].subs({Y:Y_y}).subs({X:X_x})])
print("v_f =\n%s" % v_f)

# Check whether this flow field is divergence free
div_v_f = simplify(diff(v_f[0], x) + diff(v_f[1], y))
print("div(v_f) =\n%s" % div_v_f)

# Construct the fluid stress in terms of a currently unspecified
# pressure
p_f = symbols('p_f')
grad_v_f = Matrix([[simplify(diff(v_f[0], x)), simplify(diff(v_f[0], y))],
                   [simplify(diff(v_f[1], x)), simplify(diff(v_f[1], y))]])
sigma_f = eta*(grad_v_f + grad_v_f.T) - p_f(x, y)*I

# Construct terms of the Navier-Stokes equations starting with the
# time derivative of the velocity
dv_f_dt = Matrix([simplify(diff(v_f[0], t)), simplify(diff(v_f[1], t))])

# Divergence of the stress field
div_sigma_f = Matrix([simplify(diff(sigma_f[0], x) + diff(sigma_f[1], y)),
                      simplify(diff(sigma_f[2], x) + diff(sigma_f[3], y))])

# Use (f) to determine b_f
b_f = rho_f*(dv_f_dt + grad_v_f*v_f) - div_sigma_f
print("b_f =\n%s" % b_f)

# Check if (f) is satisfied
check_F_3 = rho_f*dv_f_dt + rho_f*grad_v_f*v_f - div_sigma_f - b_f
print("rho_f*dv_f_dt + rho_f*grad_v_f*v_f - div_sigma_f - b_f =\n%s" %
      Matrix([simplify(check_F_3[0]), simplify(check_F_3[1])]))


# # Define a functional
# N_F = Matrix([0.0, 1.0])
# V_F_dot_N_F = (V_F.T*N_F)[0]
# print(integrate(integrate(V_F_dot_N_F, (X, 0, 1)), (t, 0, 0.5)))

#sigma_f_int_dot_n = (sigma_f*n_f)

#sigma_S = (1/J_S)*Sigma_S*F_S.transpose()
#sigma_s = sigma_S.subs({X:x})
#sigma_s_int_dot_n = sigma_s*n_f

#print(simplify((sigma_f*n_f - sigma_s*n_f)[0]))

# Find the current normal direction at the interface
# FIXME: Check this super carefully
# U_FSI = U_M.subs({Y:Rational(1, 2)}).subs({X:x})
# n_f = Matrix([diff(U_FSI[1], x), -1]) / sqrt(diff(U_FSI[1], x)**2 + 1)


# # Solve for the fluid pressure to satisfy the condition that the solid
# # and fluid stresses are equal on the boundary
# p_f_sol_1 = solve(Eq(sigma_f_int_dot_n[0], sigma_s_int_dot_n[0]), p_f)[0]
# p_f_sol_2 = solve(Eq(sigma_f_int_dot_n[1], sigma_s_int_dot_n[1]), p_f)[0]

# # Insert one of these pressure fields into the definition of the fluid
# # stress
# print("p_f =\n%s" % p_f_sol_1)
# print("p_f =\n%s" % p_f_sol_2)
# sigma_f = sigma_f.subs(p_f, p_f_sol_2)

# print("check_f_2 =")
# print(simplify((sigma_f*n_f - sigma_s*n_f)[0]))
# print(simplify((sigma_f*n_f - sigma_s*n_f)[1]))
