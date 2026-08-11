from skfem import *
from params import *
from skfem.helpers import dot, ddot, grad, sym_grad, eye, trace, transpose
from skfem.visuals.matplotlib import plot, draw
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numba
from scipy.sparse.linalg import eigsh

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

@BilinearForm
def mass(u, v, w):
    return rho*dot(u, v)

@LinearForm
def distributed_load(v, w):
    # Applies a uniform downward force density in the y-direction
    f = np.array([0., -1.0])  # Fixed load vector
    return dot(f, v)

u_curr = basis.zeros()

# Amplification factor for visualization 
F_amp = 20 # Ponctual force
F_spatial = np.zeros(basis.N)
F_node = np.where((np.isclose(m.p[0], L/2, rtol=0.05)) & (np.isclose(m.p[1], 0.0)))[0][0]
F_spatial[basis.nodal_dofs[1, F_node]] += 1
# F_spatial = distributed_load.assemble(basis)

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
nat_freqs = np.sqrt(np.real(eigenvals))
print(f'Eigenvalues: {nat_freqs/2/np.pi} Hz')

w1, w2  = nat_freqs[0], nat_freqs[1]    # Two first natural freqs [rad/s]
zeta1, zeta2 = 0.02, 0.02               # Two first damping ratios [-]

ar = 2 * w1 * w2 * (zeta1 * w2 - zeta2 * w1) / (w2**2 - w1**2)
br = 2 * (zeta2 * w2 - zeta1 * w1) / (w2**2 - w1**2)

Cd = ar * M + br * Klin

# Dynamic Integration Parameters (Newmark-beta)
T = 20            # Sine period [s]
NFT = 64          # Temporal discretization
t_it = np.linspace(1, NFT, NFT)   # Time step size [s]
dt = T / NFT
beta = 0.25
gamma = 0.5

# Initialize kinematic vectors
u = basis.zeros()   # Displacement
v = basis.zeros()   # Velocity
a = basis.zeros()   # Acceleration

# Newton-Raphson parameters
max_iter = int(1e2)
tol = 1e-10

# Time loop
u_history = []
KT = K_tangent.assemble(basis, u_n=u_curr)
for t in t_it:
   
    u_old = u.copy()
    v_old = v.copy()
    a_old = a.copy()

    # Calculate harmonic time-dependent force
    F_ext = F_amp * F_spatial * np.sin((2*np.pi / T ) * t*T / NFT)
    
    # Newton-Raphson iteration loop for implicit step
    u_curr = u_old.copy() # Initial guess

    for it in range(max_iter):
        print(f"Iteration: {it} at time {t*T/NFT} over {T} (={t/NFT*100} %)")
        if it >= 5:
            KT = K_tangent.assemble(basis, u_n=u_curr)

        # Compute acceleration based on current displacement guess
        a_curr = (u_curr - u_old) / (beta * dt**2) - v_old / (beta * dt) - (1.0 / (2.0 * beta) - 1.0) * a_old
        v_curr = v_old + (1.0 - gamma) * dt * a_old + gamma * dt * a_curr

        F_int_vec = f_int.assemble(basis, u_n=u_curr)
        R = (F_ext - M @ a_curr - Cd @ v_curr - F_int_vec) # Residual
        res_norm = np.linalg.norm(R[free_dofs])
        ref_norm = np.linalg.norm(F_ext[free_dofs]) + 1e-12
        if res_norm / ref_norm < tol:
            break

        dR_dyn = (1.0 / (beta*dt**2)) * M + (gamma / (beta*dt)) * Cd + KT
        
        du = solve(*condense(dR_dyn, R, D=dofs))
        if np.linalg.norm(du) < tol:
            break
        u_curr += du
        
    u = u_curr
    a = (u - u_old) / (beta * dt**2) - v_old / (beta * dt) - (1.0 / (2.0 * beta) - 1.0) * a_old
    v = v_old + (1.0 - gamma) * dt * a_old + gamma * dt * a
    u_history.append(u.copy())

# print('Linear stiffness ', Klin @ u)
# print(f'Eigenvalues: {nat_freqs/2/np.pi} Hz')

if __name__ == "__main__":
    from skfem.visuals.matplotlib import plot, show
    amp = 5     # Amplification factor for plotting

    deformed_coords = m.p + amp * u[basis.nodal_dofs]
    y_min, y_max = deformed_coords[1, :].min(), deformed_coords[1, :].max()
    x_min, x_max = deformed_coords[0, :].min(), deformed_coords[0, :].max()
    # M = MeshQuad(np.array(m.p + amp * u[basis.nodal_dofs]), m.t)
    # ax1 = draw(M)
    # plot(M, u[basis.nodal_dofs[1]], ax=ax1)
    margin_x = 0.05 * (x_max - x_min)
    margin_y = 0.05 * (y_max - y_min)
    # ax1.set_xlim(x_min - margin_x, x_max + margin_x)
    # ax1.set_ylim(y_min - margin_y, y_max + margin_y)
    # ax1.set_aspect('auto')

    plt.figure()
    plt.plot(t_it * T / NFT, F_amp * np.sin((2*np.pi/T) * t_it * T / NFT))
    plt.xlim([t_it[0] * T / NFT, t_it[-1] * T / NFT])
    plt.xlabel('Time (s)')
    plt.ylabel('Max displacement (m)')  
    # plt.savefig('figs/temporal_deformation.pdf')


    fig, ax = plt.subplots(figsize=(9,5))
    disp_values = np.abs(np.concatenate([u_t[basis.nodal_dofs[1]] for u_t in u_history]))
    vmin = 0.0
    vmax = np.max(disp_values) if disp_values.size else 1.0
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap('viridis')

    def update(frame):
        ax.clear()

        u_t = u_history[frame]
        M_deformed = MeshQuad(np.array(m.p + amp * u_t[basis.nodal_dofs]), m.t)

        draw(M_deformed, ax=ax)
        plot(M_deformed, np.abs(u_t[basis.nodal_dofs[1]]), ax=ax, cmap=cmap, vmin=vmin, vmax=vmax)

        ax.set_xlim(x_min - margin_x, x_max + margin_x)
        ax.set_ylim(-0.002, 0.005)  # y_min - margin_y, y_max + margin_y)
        ax.set_title(f"Time : {frame * dt:.3f} s (amp x{amp})")

    anim = FuncAnimation(fig, update, frames=len(u_history), interval=50)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label='|u_y|')
    anim.save('figs/beam_deformation.gif', writer='pillow', fps=10)

    show()
