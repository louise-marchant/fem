from skfem import *
import numpy as np
from ElementBeam import *

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

"""Plain stress assumption - Clamped-clamped beam"""
m1 = np.linspace(0, L, 50)
m2 = np.linspace(0, l, 2)
m = MeshQuad.init_tensor(m1, m2).with_defaults()

e1 = ElementQuad2()
e = ElementVector(e1)
basis = Basis(m, e, intorder=4)

F_spatial = np.zeros(basis.N)
F_node = np.where((np.isclose(m.p[0], L/2, rtol=0.05)) & (np.isclose(m.p[1], 0.0)))[0][0]

F_spatial[basis.nodal_dofs[1, F_node]] += 1

# Boundary conditions
D = m.boundary_nodes()
dofs = basis.get_dofs(lambda x: (x[0] == 0.) | (x[0] == L))
free_dofs = basis.complement_dofs(dofs)

""" Euler-Bernouilli Beam - Clamped-clamped beam"""
# m1 = np.linspace(0, L, 100)
# m = MeshLine.init_tensor(m1)

# e1 = ElementLineP1()        # dof u
# e2 = ElementLineHermite()   # dof v, theta
# e  = ElementComposite(e1, e2)
# basis = Basis(m, e)

# F_spatial = np.zeros(basis.N)
# F_node = np.where(np.isclose(m.p[0], L/2, rtol=0.05))[0][0]
# F_spatial[basis.nodal_dofs[1, F_node]] += 1

# # Boundary conditions
# D = m.boundary_nodes()
# dofs = basis.get_dofs(lambda x: (x[0] == 0.) | (x[0] == L))
# free_dofs = basis.complement_dofs(dofs)

"""3D model - Plate"""
# L = 0.5                     # Plate length [m]
# l = 0.5                     # Plate height ([m])
# h = 0.005                   # Plate width ([m])

# m1 = np.linspace(0, L, 15)
# m2 = np.linspace(0, l, 15)
# m3 = np.linspace(0, h, 3)
# m = MeshHex.init_tensor(m1, m2, m3).with_defaults()

# e1 = ElementHex2()
# e = ElementVector(e1)
# basis = Basis(m, e, intorder=3)

# F_spatial = np.zeros(basis.N)
# F_node = np.where((np.isclose(m.p[0], L/2, rtol=0.05)) & (np.isclose(m.p[1], l/2, rtol= 0.05)) & (np.isclose(m.p[2], 0.0, rtol= 0.05)))[0][0]
# F_spatial[basis.nodal_dofs[2, F_node]] += 1

# # Boundary conditions
# D = m.boundary_nodes()
# dofs = basis.get_dofs(lambda x: (x[0] == 0.) | (x[0] == L) | (x[1] == 0.) | (x[1] == l))
# free_dofs = basis.complement_dofs(dofs)

print(f"Basis: {basis}")