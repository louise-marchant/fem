from skfem import *
from skfem.helpers import dot, ddot, grad, sym_grad, eye, trace, transpose
from skfem.visuals.matplotlib import plot, draw
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.sparse.linalg import eigsh

# Material properties
E = 210e9                   # Young modulus [GPa]
v = 0.3                     # Poison's ratio [-]
rho = 7800                  # Volumic mass [kg/m^3]
lam = (E*v)/((1+v)*(1-2*v)) # First Lame parameter
mu = E/(2*(1+v))            # Second Lame parameter
L = 5.                      # Beam length [m]
h = 0.01                   # Beam height ([m])

# Mesh creation (or import)
m1 = MeshLine(np.linspace(0, L, 50))
m2 = MeshLine(np.linspace(0, h, 10))
m = (m1 * m2).with_defaults()

# Element creation (or import)
e1 = ElementQuad1()
e = ElementVector(e1)
basis = Basis(m, e, intorder=2)

# Stress tensor
def C(T):
    return 2. * mu * T + lam * eye(trace(T), T.shape[0])

@BilinearForm
def nlstiffness(u, v, w):
    u_prev = w['u_prev']
    C_arg = sym_grad(u) + 0.5*dot( transpose(grad(u_prev)), grad(u) )
    return ddot(C_arg, sym_grad(v))

@BilinearForm
def stiffness(u, v, w):
    return ddot( C(sym_grad(u)), sym_grad(v) )

@BilinearForm
def mass(u, v, w):
    return rho*dot(u, v)

@BilinearForm
def jacobian(du, v, w):
    u_n = w['u_n']
    geom = 0.5 * (dot(transpose(grad(u_n)), grad(du)) + dot(transpose(grad(du)), grad(u_n)))
    return ddot(C(sym_grad(du) + geom), sym_grad(v))    

# @LinearForm
# def force(v, w):
#     x = w.x[0]
#     f = v*0
#     f[1] = -1*x/L
#     return dot(f, v)

@LinearForm
def force(v, w):
    f = v*0
    return dot(f, v)

u_curr = basis.zeros()

# Ponctual force
# F_spatial = force.assemble(basis)
F_spatial = basis.point_source(x=np.array([L/2, h]))
# mesh_x = m.p[0, :]
# mesh_y = m.p[1, :]
# node_idx = np.argmin(np.abs(mesh_x - L/2) + np.abs(mesh_y - h/2))
# dof_y = basis.nodal_dofs[1, node_idx]
# F_spatial[dof_y] = 0.001  # Shaker amplitude force [N]

# Matrices assembly
K = nlstiffness.assemble(basis, u_prev=u_curr)
Klin = stiffness.assemble(basis)
M = mass.assemble(basis)

# Boundary conditions - clamped-clamped beam
D = m.boundary_nodes()
dofs = basis.get_dofs(lambda x: (x[0] == 0.) | (x[0] == L))
free_dofs = basis.complement_dofs(dofs)

Kc = K[free_dofs, :][:, free_dofs]  # Constrained stiffness
Mc = M[free_dofs, :][:, free_dofs]  # Constrained mass

# Computation of eigenmodes and eigenfrequencies
nmodes = 3
eigenvals, eigenvecs = eigsh(Kc, k=nmodes, M=Mc, sigma=0.0)
nat_freqs = np.sqrt(np.real(eigenvals))
print(f'Eigenvalues: {nat_freqs/2/np.pi} Hz')

w1, w2  = nat_freqs[0], nat_freqs[1]    # Two first natural freqs [rad/s]
zeta1, zeta2 = 0.02, 0.02               # Two first damping ratios [-]

ar = 2 * w1 * w2 * (zeta1 * w2 - zeta2 * w1) / (w2**2 - w1**2)
br = 2 * (zeta2 * w2 - zeta1 * w1) / (w2**2 - w1**2)

Cd = ar * M + br * Klin

# Show modeid
modeid = 0
mode = basis.zeros()
mode[free_dofs] = eigenvecs[:, modeid]
mode_y = mode[1::2]
basis_scalar = CellBasis(m, ElementQuad1())

fig, ax = plt.subplots(figsize=(7,5))
plot(basis_scalar, mode_y, ax=ax, shading='gouraud', cmap='viridis')
img = ax.collections[0]
fig.colorbar(img, ax=ax, label="Mode displacement")
ax.set_title(f"Mode shape {modeid+1}")

# Dynamic Integration Parameters (Newmark-beta)
dt = 2*np.pi/(30*w1)  # Time step size [s]
t_max = 0.2           # Total simulation time [s]
beta = 0.25
gamma = 0.5
freq = 15.0           # Forcing frequency [Hz]
omega = 2.0 * np.pi * freq

# Initialize kinematic vectors
u = basis.zeros()   # Displacement
v = basis.zeros()   # Velocity
a = basis.zeros()   # Acceleration

# Newton-Raphson parameters
max_iter = int(1e2)
tol = 1e-6

# Time loop
t = 0.0
u_history = [] # List to save sols at each time step
while t < t_max:
    t += dt

    # Predictor / Constants for current step
    u_old = u.copy()
    v_old = v.copy()
    a_old = a.copy()

    # Calculate harmonic time-dependent force
    F_ext = F_spatial * np.sin(omega * t)
    
    # Newton-Raphson iteration loop for implicit step
    u_curr = u_old.copy() # Initial guess for u_{t+dt}

    for it in range(max_iter):
        if it % 100:
            print(f"Iteration : {it} at time {t} over {t_max} (={t/t_max*100} %)")

        # Compute acceleration based on current displacement guess
        a_curr = (u_curr - u_old) / (beta * dt**2) - v_old / (beta * dt) - (1.0 / (2.0 * beta) - 1.0) * a_old
        v_curr = v_old + (1.0 - gamma) * dt * a_old + gamma * dt * a_curr

        K_nl = nlstiffness.assemble(basis, u_prev=u_curr)

        # Residual
        R = F_ext - M @ a_curr - Cd @ v_curr - K_nl @ u_curr
        res_norm = np.linalg.norm(R[free_dofs])
        if res_norm/np.linalg.norm(F_spatial + 1e-6) < tol:
            break

        dR_static = jacobian.assemble(basis, u_n=u_curr)   # Jacobian of static motion
        dR_dyn = (1.0 / (beta * dt**2)) * M + gamma/(beta * dt) * Cd + dR_static # Complete Jacobian
        du = solve(*condense(dR_dyn, R, D=dofs))
        u_curr += du

    u = u_curr
    a = (u - u_old) / (beta * dt**2) - v_old / (beta * dt) - (1.0 / (2.0 * beta) - 1.0) * a_old
    v = v_old + (1.0 - gamma) * dt * a_old + gamma * dt * a
    u_history.append(u.copy())

print('Nonlinear stiffness ', K @ u - Klin @ u)
print('Linear stiffness ', Klin @ u)

if __name__ == "__main__":
    from skfem.visuals.matplotlib import plot, show
    amp = 1e8 # Amplification factor

    M = MeshQuad(np.array(m.p + amp * u[basis.nodal_dofs]), m.t)
    ax1 = draw(M)
    plot(M, u[basis.nodal_dofs[1]], ax=ax1)

    fig, ax = plt.subplots(figsize=(7,5))
    
    def update(frame):
        ax.clear()
        
        u_t = u_history[frame]
        M_deformed = MeshQuad(np.array(m.p + amp * u_t[basis.nodal_dofs]), m.t)
        
        draw(M_deformed, ax=ax)
        plot(M_deformed, u_t[basis.nodal_dofs[1]], ax=ax)
        
        ax.set_xlim(-0.5, L+0.5)
        ax.set_ylim(-(2*h+0.1), 2*h+0.1)
        ax.set_title(f"Time : {frame * dt:.3f} s (amp x{amp})")

    anim = FuncAnimation(fig, update, frames=len(u_history), interval=100)

    show()
