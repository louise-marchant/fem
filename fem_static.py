from skfem import *
from params import *
from skfem.helpers import dot, ddot, grad, sym_grad, eye, trace, transpose
from skfem.visuals.matplotlib import plot, draw
import numpy as np
from matplotlib.animation import FuncAnimation
import os
os.environ['XDG_SESSION_TYPE'] = 'gnome'
import matplotlib.pyplot as plt
import matplotlib
from scipy.optimize import curve_fit, root
from scipy.sparse.linalg import LinearOperator, splu
from load_NX_data import load_NX_file

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
    dE_du = sym_grad(du) + 0.5 * (dot(transpose(grad(u_n)), grad(du)) + dot(transpose(grad(du)), grad(u_n)))
    # Variation of E in direction v (virtual)
    dE_dv = sym_grad(v) + 0.5 * (dot(transpose(grad(u_n)), grad(v)) + dot(transpose(grad(v)), grad(u_n)))
    # Material stiffness
    K_mat = ddot(C(dE_du), dE_dv)
    # Geometrical stiffness (3D elements)
    K_geo = np.einsum('ij..., kj..., ki... -> ...', S, grad(v), grad(du))

    return K_mat + K_geo

@BilinearForm
def stiffness(u, v, w):
    return ddot( C(sym_grad(u)), sym_grad(v) )

@LinearForm
def distributed_load(v, w):
    f = np.array([0., 1.0]) * 1/(L * h)
    return dot(f, v)

def residual(uc, f_ext, free_dofs):
    u = np.zeros_like(x)    # Full sol vector
    u[free_dofs] = uc       # Full sol vector where constraints dofs are null
    Ku = f_int.assemble(basis, u_n=u)
    R = Ku - f_ext
    return R[free_dofs]

# Linear stiffness matrix assembly
Klin = stiffness.assemble(basis)

# Boundary conditions - clamped-clamped beam
D = m.boundary_nodes()
dofs = basis.get_dofs(lambda x: (x[0] == 0.) | (x[0] == L))
free_dofs = basis.complement_dofs(dofs)
x = basis.zeros()

# Newton-Raphson parameters
max_iter = int(1e3)
rtol = 1e-5
atol = 1e-8
u_nlc = basis.zeros()   # Displacement

# Excitation force
f_amps =  np.linspace(0.5, 5, 20)
bottom_facets = m.facets_satisfying(lambda x: np.isclose(x[1], 0.0))
facet_basis = FacetBasis(m, e, facets=bottom_facets)

f_spatial = distributed_load.assemble(facet_basis)

y_max_nl = []
y_max_lin = []
x_c = np.array([[L / 2], [l / 2]])

for i, f_exc in enumerate(f_amps):
    print(f'F = {f_exc} N, it. number {i+1}/{len(f_amps)}')
    f_ext = f_exc * f_spatial
    
    u_lin = solve(*condense(Klin, f_ext, D=dofs))
    KT = K_tangent.assemble(basis, u_n=u_nlc)
    
    for it in range(max_iter):
        F_int_vec = f_int.assemble(basis, u_n=u_nlc)
        R = f_ext - F_int_vec
        
        res_norm = np.linalg.norm(R[free_dofs])
        ref_norm = np.linalg.norm(f_ext[free_dofs])

        if res_norm < (atol + rtol * ref_norm):
            print(f"    Converged in {it} iter with a rel. residual {res_norm / ref_norm:.2e}")
            break
        
        # For classical Newton-Raphason iterations, new computations of derivative of residual KT at each step (uncomment line)
        # For modified Newton-Raphson iterations, computation of derivative of residual KT only at new forcing steps (faster, but less robust if next line is commented)
        # KT = K_tangent.assemble(basis, u_n=u_nlc)
        
        du_condensed = solve(*condense(KT, R, D=dofs))[free_dofs]
        
        # Line search
        alpha = 1.0
        alpha_ref = 1.0
        min_res = res_norm

        while alpha > 0.05:
            u_it = u_nlc.copy()
            u_it[free_dofs] += alpha * du_condensed

            R_it = f_ext - f_int.assemble(basis, u_n=u_it)
            res_it = np.linalg.norm(R_it[free_dofs])
            
            if res_it < res_norm:
                alpha_ref = alpha
                break
                
            if res_it < min_res:
                min_res = res_it
                alpha_ref = alpha
            alpha *= 0.5

        if alpha_ref != 1.0:
            print(f"   [Line search] Damped step at alpha = {alpha_ref} (Residual: {res_norm:.2e} -> {min_res:.2e})")
        u_nlc[free_dofs] += alpha_ref * du_condensed
        
    else:
        print(f"Convergence issue at F = {f_exc} after {max_iter} iterations")
        u_nlc[free_dofs] = u_lin[free_dofs]

    u_nl = u_nlc.copy()
    unl_interp = basis.interpolator(u_nl)
    ulin_interp = basis.interpolator(u_lin)
    unl_c = unl_interp(x_c)[1, 0]       # Compute displacement in y direction at middle of the beam
    ulin_c = ulin_interp(x_c)[1, 0]

    y_max_nl.append(unl_c*1000)         # [mm] scaling
    y_max_lin.append(ulin_c*1000)       # [mm] scaling

if __name__ == "__main__":
    from skfem.visuals.matplotlib import plot, show

    # Visualization amplitude factor
    amp = 1.0

    M = MeshQuad(np.array(m.p + amp * u_nl[basis.nodal_dofs]), m.t)
    # M = MeshTri(np.array(m.p + amp * u_nl[basis.nodal_dofs]), m.t)
    # ax1 = draw(M)
    # plot(M, u_nl[basis.nodal_dofs[1]], ax=ax1)
    # ax1.set_aspect('auto')
    # ax1.figure.savefig('figs/fem_nonlin_beam.pdf')

    # M = MeshQuad(np.array(m.p + amp * u_lin[basis.nodal_dofs]), m.t)
    # ax = draw(M)
    # plot(M, u_lin[basis.nodal_dofs[1]], ax=ax)
    # ax.set_aspect('auto')
    # ax.figure.savefig('figs/fem_lin_beam.pdf')

    # popt, _ = curve_fit(func, y_max_nl, f_amps)
    popt = np.polyfit(np.array(y_max_nl), f_amps, 3)
    b = np.array(y_max_nl)
    nl_fit = popt[0]*b**3 + popt[1]*b**2 + popt[2]*b + popt[3]
    print(f'Polynomial fit: {popt[0]}*b**3 + {popt[1]}*b**2 + {popt[2]}*b + {popt[3]}')

    iter_num = np.linspace(1, len(f_amps), len(f_amps))
    iter_num_nx, y_max_nl_nx = load_NX_file("NX_data/max_disp_10N.csv")
    plt.figure()
    plt.plot(y_max_nl, f_amps, color='tab:blue', label='Nonlinear', linewidth=2)
    plt.plot(y_max_lin, f_amps, color='tab:blue', label='Linear', linestyle='dashed', linewidth=2)
    plt.plot(y_max_nl_nx, f_amps, color='tab:orange', label="Nonlinear (NX)", linestyle='-.', linewidth=2)
    # plt.plot(y_max_nl, nl_fit, color='tab:red', label="Polynomial fit", linestyle=':', linewidth=2)
    plt.xlabel('Max. displacement [mm]')
    plt.ylabel('Forcing amplitude [N]')
    plt.legend(frameon=False)
    plt.xlim([np.maximum(y_max_nl[0], y_max_lin[0]), np.minimum(y_max_nl[-1], y_max_lin[-1])])

    plt.figure()
    plt.plot(f_amps, (np.abs(y_max_nl-y_max_nl_nx)/y_max_nl_nx)*100, color='tab:blue', linewidth=2)
    plt.ylabel('Error [%]')
    plt.xlabel('Forcing amplitude [N]')
    plt.show()
