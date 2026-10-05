import sympy as sp
import math

# define symbolic variables for the question
x, y, theta = sp.symbols('x y theta', real=True)
l_B, l_1, l_2 = sp.symbols('l_B l_1 l_2', positive=True)
alpha, gamma, psi = sp.symbols('alpha gamma psi', real=True)

# helper function for creating homogeneous transformation matrix
def make_hom_transform(angle, x, y):
    c = sp.cos(angle)
    s = sp.sin(angle)
    return sp.Matrix([
        [c, -s, x],
        [s, c, y],
        [0, 0, 1]
    ])

# QUESTION 1 A)
print("-----QUESTION 1 a)-----")

# transformation from reference frame {E} to {2}
G_E_2 = make_hom_transform(psi, l_2, 0)
print("G_E_2: ")
sp.pprint(G_E_2)

# transformation from reference frame {2} to {1}
G_2_1 = make_hom_transform(gamma, l_1, 0)
print("\nG_2_1: ")
sp.pprint(G_2_1)

# transformation from reference frame {1} to reference frame {B}
G_1_B = make_hom_transform(alpha, l_B, 0)
print("\nG_1_B: ")
sp.pprint(G_1_B)

# transformation from reference frame {B} to reference frame {s}
G_B_s = make_hom_transform(theta, x, y)
print("\nG_B_s: ")
sp.pprint(G_B_s)

# Compute the total transformation matrix from {E} tp {s}
G_E_s = sp.simplify(G_B_s * G_1_B * G_2_1 * G_E_2)
print("\nTotal transformation G_E_s")
sp.pprint(G_E_s)

# QUESTION 1 B)
print("\n-----QUESTION 1 b)-----")

# homogenous coordinates for points p1 and p2 in {E}
p_1_E = sp.Matrix([0, 0, 1])
p_2_E = sp.Matrix([0.5, 1, 1])

# values for the symbols
values = {
    psi : -sp.pi/2,
    theta: -sp.pi/2,
    alpha : sp.pi/6,
    gamma : sp.pi/4,
    l_1 : 1,
    l_2 : 1,
    l_B : sp.Rational(1,2),
    x : sp.Rational(4, 5), #0.8
    y : sp.Rational(4, 5)
    }

p_1_s = (G_E_s * p_1_E).subs(values)
p_2_s = (G_E_s * p_2_E).subs(values)

print("\np_1_s=")
sp.pprint(sp.N(p_1_s))

print("\np_2_s=")
sp.pprint(sp.N(p_2_s))

# QUESTION 1 C)
print("\n-----QUESTION 1 c)-----")

# transformation matrix from {B} to {E}
G_E_2_num = G_E_2.subs(values)
G_2_1_num = G_2_1.subs(values)
G_1_B_num = G_1_B.subs(values)

G_B_E = (
    G_E_2_num.inv()
    * G_2_1_num.inv()
    * G_1_B_num.inv()
)

# homogenous coordinates for q in {B}
q_B = sp.Matrix([1, 2, 1])

q_E = (G_B_E * q_B)
sp.pprint(sp.N(q_E))




