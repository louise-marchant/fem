from skfem import *
import numpy as np

# Material properties
E = 210e9                   # Young modulus [GPa]
v = 0.3                     # Poison's ratio [-]
rho = 7800                  # Volumic mass [kg/m^3]
lam = (E * v) / (1 - v**2)  # Plane stress - First Lame parameter
mu = E / (2 * (1 + v))            # Second Lame parameter
L = 0.7                     # Beam length [m]
h = 0.020                    # Beam height ([m])
l = 0.002                   # Beam width ([m])

"""Plain stress assumption - Timoschenko Bar"""
m1 = np.linspace(0, L, 25)
m2 = np.linspace(0, l, 4)
m = MeshQuad.init_tensor(m1, m2).with_defaults()

e1 = ElementQuad2() # ElementQuad1()
e = ElementVector(e1)
basis = Basis(m, e, intorder=4)   # basis = Basis(m, e, intorder=2)