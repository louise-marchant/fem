from skfem import *
import numpy as np

# Material properties
E = 210e9                   # Young modulus [GPa]
v = 0.3                     # Poison's ratio [-]
rho = 7800                  # Volumic mass [kg/m^3]
lam = (E * v) / (1 - v**2)  # Plane stress - First Lame parameter
# lam = (E * v) / ((1 + v) * (1 - 2 * v))  # Plane strain / 3D - First Lame parameter
mu = E / (2 * (1 + v))      # Second Lame parameter 

L = 0.7                     # Beam length [m]
h = 0.020                   # Beam height ([m])
l = 0.002                   # Beam width ([m])

Iy = h * l**3 / 12

"""Plain stress assumption - Timoschenko Bar"""
m1 = np.linspace(0, L, 350)
m2 = np.linspace(0, l, 5)
m = MeshQuad.init_tensor(m1, m2).with_defaults()
# m = MeshTri.init_tensor(m1, m2).with_defaults()

e1 = ElementQuad2() # ElementQuad1()
# e1 = ElementTriP2()
e = ElementVector(e1)
basis = Basis(m, e, intorder=4)   # basis = Basis(m, e, intorder=2)
print(basis)