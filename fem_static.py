from skfem import *
from params import *
from skfem.helpers import dot, ddot, grad, sym_grad, eye, trace, transpose
from skfem.visuals.matplotlib import plot, draw
import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, root
from scipy.sparse.linalg import LinearOperator, splu

def func(x, a, b):
    return a*x + b*x**3

def C(T):
    return 2. * mu * T + lam * eye(trace(T), T.shape[0])

# Green-Lagrange deformation tensor
def E_GL(u):
    return sym_grad(u) + 0.5 * dot(transpose(grad(u)), grad(u))

# 2nd Piola-Kirchhoff stress
def S_PK(u):
    return C(E_GL(u))

# Internal force - Evaluated at state u_prev (v is the test function)
@LinearForm
def f_int(v, w):
    u_n = w['u_n']
    S = S_PK(u_n)
    # Variation of E in direction v, with u_n fixed
    # Linear term + large displacement term
    dE_v = sym_grad(v) + 0.5 * ( dot(transpose(grad(u_n)), grad(v)) + dot(transpose(grad(v)), grad(u_n)))
    return ddot(S, dE_v)

# Tangential matrix (derivative of f_int) — BilinearForm (K_mat + K_geo)
@BilinearForm
def K_tangent(du, v, w):
    u_n = w['u_n']
    S = S_PK(u_n)

    # Variation of E in direction du
    dE_du = sym_grad(du) + 0.5 * (
        dot(transpose(grad(u_n)), grad(du)) +
        dot(transpose(grad(du)),  grad(u_n))
    )
    # Variation of E in direction v (virtual)
    dE_dv = sym_grad(v) + 0.5 * (
        dot(transpose(grad(u_n)), grad(v)) +
        dot(transpose(grad(v)),   grad(u_n))
    )

    # Material stiffness
    K_mat = ddot(C(dE_du), dE_dv)

    # Geometrical stiffness (3D elements)
    # grad_outer = np.einsum('ki...,kj...->ij...', grad(v), grad(du))
    # K_geo = ddot(S, grad_outer)
    K_geo = np.einsum('ij..., kj..., ki... -> ...', S, grad(v), grad(du))           # OK FOR SYMMETRIC S TENSOR

    return K_mat + K_geo

@BilinearForm
def stiffness(u, v, w):
    return ddot( C(sym_grad(u)), sym_grad(v) )

@LinearForm
def distributed_load(v, w):
    f = np.array([0., 1.0]) * 1/(L * h)
    return dot(f, v)

def preconditioner_factory(u, f):
    # Calculate the exact Jacobian at the current step
    J = K_tangent.assemble(basis, u_n=u)
    
    # Incomplete LU factorization of the exact Jacobian
    #ilu = splu(J.tocsc())#, drop_tol=1e-3)
    
    # Return as a LinearOperator for the Krylov solver
    return J    # LinearOperator(J.shape, matvec=ilu.solve)

def residual(u, f_ext):
    Ku = f_int.assemble(basis, u_n=u)
    return (Ku - f_ext)

Klin = stiffness.assemble(basis)

# Boundary conditions - clamped-clamped beam
D = m.boundary_nodes()
dofs = basis.get_dofs(lambda x: (x[0] == 0.) | (x[0] == L))
free_dofs = basis.complement_dofs(dofs)
x = basis.zeros()

# Newton-Raphson parameters
max_iter = int(1e4)
tol_rel = 1e-4
tol_abs = 1e-6
u_nlc = basis.zeros()   # Displacement

# Visualization amplitude factor
amp = 1.0

# Excitation force
F_amps =  np.linspace(0.1, 10, 10)
bottom_facets = m.facets_satisfying(lambda x: np.isclose(x[1], 0.0))
facet_basis = FacetBasis(m, e, facets=bottom_facets)

F_spatial = distributed_load.assemble(facet_basis)

y_max_nl = []
y_max_lin = []
u_nlc = x.copy()

x_c = np.array([[L / 2], [l / 2]])

for F_exc in F_amps:
    F_ext = F_exc * F_spatial
    
    # u_lin = solve(*condense(Klin, F_ext, D=dofs))
    # u_nlc = u_lin
    # for _ in range(5):
    #     u_nlc = root(residual, u_nlc, args=(F_ext,), jac=preconditioner_factory, method='hybr', options={'disp': True}).x
    
    # Newton-Raphson iterations
    for it in range(max_iter):
        F_int_vec = f_int.assemble(basis, u_n=u_nlc)
        R = F_ext - F_int_vec # Residual
        res_norm = np.linalg.norm(R[free_dofs])
        ref_norm = np.linalg.norm(F_ext[free_dofs]) + 1e-12
        if res_norm / ref_norm < tol_rel or res_norm < tol_abs:
            break
        
        if it % 100 == 0:
            print(f'Excitation force {F_exc}')
            print(f"Rel. residual {res_norm / ref_norm} and abs. residual {res_norm} at iteration : {it} (={it/max_iter*100} %)")

        KT = K_tangent.assemble(basis, u_n=u_nlc)
        dR_dyn = KT

        du = solve(*condense(dR_dyn, R, D=dofs))
        u_nlc += du

    u_nl = u_nlc
    u_lin = solve(*condense(Klin, F_ext, D=dofs))

    unl_interp = basis.interpolator(u_nl)
    ulin_interp = basis.interpolator(u_lin)
    unl_c = unl_interp(x_c)[1, 0]       # Compute displacement in y direction at middle of the beam
    ulin_c = ulin_interp(x_c)[1, 0]

    y_max_nl.append(unl_c)
    y_max_lin.append(ulin_c)

if __name__ == "__main__":
    from skfem.visuals.matplotlib import plot, show

    M = MeshQuad(np.array(m.p + amp * u_nl[basis.nodal_dofs]), m.t)
    ax1 = draw(M)
    plot(M, u_nl[basis.nodal_dofs[1]], ax=ax1)
    ax1.set_aspect('auto')
    # ax1.figure.savefig('figs/fem_nonlin_beam.pdf')

    M = MeshQuad(np.array(m.p + amp * u_lin[basis.nodal_dofs]), m.t)
    ax = draw(M)
    plot(M, u_lin[basis.nodal_dofs[1]], ax=ax)
    ax.set_aspect('auto')
    # ax.figure.savefig('figs/fem_lin_beam.pdf')

    # popt, _ = curve_fit(func, y_max_nl, F_amps)
    # popt = np.polyfit(np.array(y_max_nl), F_amps, 3)
    # b = np.array(y_max_nl)
    # nl_fit = popt[0]*b**3 + popt[1]*b**2 + popt[2]*b + popt[3]
              
    plt.figure()
    plt.plot(y_max_nl, F_amps, color='tab:blue', label='Nonlinear')
    plt.plot(y_max_lin, F_amps, color='tab:blue', label='Linear', linestyle='dashed')
    # plt.plot(((F_amps/L)*L**4)/(384*E*Iy), F_amps, color='tab:cyan', label="Beam theory")
    # plt.plot(y_max_nl, nl_fit, color='tab:red', label='Polynomial fit')
    plt.xlabel('Max. displacement [m]')
    plt.ylabel('Restoring force [N]')
    # plt.xlim([y_max_nl[0], y_max_nl[-1]])
    plt.legend(frameon=False)
    # plt.savefig('figs/deformations.pdf')

    plt.show()
