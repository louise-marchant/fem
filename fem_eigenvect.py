"""
Validation of scikit-fem computation of eigenmodes and eigenvalues with NX simulation
Assumption of Timoschenko bar
"""
from skfem import *
from params import *
from skfem.helpers import dot, ddot, grad, sym_grad, eye, trace, transpose
from skfem.visuals.matplotlib import plot, draw
import numpy as np
from matplotlib.animation import FuncAnimation
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt

# Stress tensor
def C(T):
    return 2. * mu * T + lam * eye(trace(T), T.shape[0])

@BilinearForm
def stiffness(u, v, w):
    return ddot(C(sym_grad(u)), sym_grad(v))

@BilinearForm
def mass(u, v, w):
    return rho*dot(u, v)

"""Euler-Bernouilli assumption (valid as L/l >> 10)"""
# m1 = np.linspace(0, L, 700)
# m = MeshLine(m1)
# # Use Hermite elements (tracks displacement 'w' and slope 'dw/dx')
# e = ElementLineHermite()
# basis = Basis(m, e)

# I_zz = (l * h**3) / 12  # Second moment of area
# Area = l * h            # Cross-section area

# @BilinearForm
# def stiffness(u, v, w):
#     # Ddot of second derivatives (curvature)
#     return E * I_zz * u.grad[0][0] * v.grad[0][0] 

# @BilinearForm
# def mass(u, v, w):
#     # Standard L2 product multiplied by mass per unit length
#     return rho * Area * u * v

""" Eigenmodes computation """
# Matrices assembly
Klin = stiffness.assemble(basis)
M = mass.assemble(basis)

# Boundary conditions - clamped-clamped beam
D = m.boundary_nodes()
dofs = basis.get_dofs(lambda x: (x[0] == 0.) | (x[0] == L))
free_dofs = basis.complement_dofs(dofs)

Kc = Klin[free_dofs, :][:, free_dofs]  # Constrained stiffness
Mc = M[free_dofs, :][:, free_dofs]  # Constrained mass

# Computation of eigenmodes and eigenfrequencies
print('Computation of eigenvalues')
nmodes = 3
eigenvals, eigenvecs = eigsh(Kc, k=nmodes, M=Mc, sigma=0.0)
nat_freqs = np.sqrt(np.real(eigenvals))/2/np.pi
nat_freqs_NX = np.array([21.57, 59.48, 116.65])
print(f'Eigenvalues: {nat_freqs} Hz')
print(f'Error (validation with NX): {abs(nat_freqs-nat_freqs_NX)/nat_freqs_NX * 100} %')
# print(f'Error (validation with NX): {abs(nat_freqs-nat_freqs_NX)**2/nat_freqs_NX**2 * 100} %')

eigenvec_full = np.zeros(basis.N)
scalar_basis_plot = Basis(m, ElementQuad1())

for modeid in range(nmodes):
    eigenvec_full[free_dofs] = eigenvecs[:,modeid]
    u_x = eigenvec_full[basis.nodal_dofs[0]]
    u_y = eigenvec_full[basis.nodal_dofs[1]]

    scale_factor = (0.05 * L) / np.max(np.abs(u_y))
    deformed_coords = m.p.copy()
    deformed_coords[0, :] += scale_factor * u_x
    deformed_coords[1, :] += scale_factor * u_y

    fig, ax = plt.subplots(figsize=(10, 5))
    m_deformed = type(m)(deformed_coords, m.t)
    disp_magnitude = np.sqrt(u_x**2 + u_y**2)
    cb = plot(m_deformed, disp_magnitude, ax=ax, cmap='rainbow', shading='gouraud', zorder=-1) #, colorbar='Displacement Magnitude')
    # ax.set_title(f"Mode shape {modeid+1} - Frequency: {nat_freqs[modeid]:.2f} Hz")
    ax.set_aspect('equal')
    ax.set_xlabel("X coordinate [m]")
    # ax.set_ylabel("Y coordinate [m]")
    ax.spines[['top', 'bottom', 'left', 'right']].set_visible(False)
    ax.set_yticks([])

plt.tight_layout()
plt.show()